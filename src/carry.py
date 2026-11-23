"""
carry.py
========
Planning with carried-over local work (experiment E13, November 2026).

With ``GIPA_LOCAL_CARRY`` set (see ``gipa_core``), a vehicle's on-board
processor carries unfinished local work into the next 20 ms slot, so a task
kept local waits for the earlier local tasks of the same vehicle. This module
gives the planners the resulting local cost.

Local queue of one vehicle
--------------------------
One task arrives at the start of every slot. With probability f the plan
sends it to an RSU; otherwise it runs locally and adds its service time L to
the backlog W, which drains by one slot per slot:
    W' = max(W + L 1{local} - 20, 0).
In 'own' mode a task whose predicted local latency W + L exceeds its
deadline D is sent to an RSU instead (the vehicle knows W without any
communication). :func:`local_chain` solves the stationary distribution of W
on a 1 ms lattice (all regime latencies are whole milliseconds) and returns
the share of tasks run locally, their mean latency, the local miss
probability and whether the queue is stable.

Planning
--------
:func:`carry_plan` wraps any of the symmetric planners. It replaces each
class's local latency L_k by the mean latency of its local tasks under the
current plan and re-plans until the plan stops changing (damped fixed
point). In 'own' mode the planner's offload share is the *effective* share
the RSUs receive; the vehicle's sampling probability is set lower, so that
sampling plus the redirect of tasks predicted to miss locally gives that
share (:func:`sampling_share`). Redirects alone send a class share
1 - share_local(f = 0) to the RSUs (1/11 of Regime A background tasks, 1/3
of Regime B background tasks, every task of a class with L_k > D_k); the
planners receive it as a minimum offload share (``MIN_OFF_S``, ``MIN_OFF_B``
in the regime dict) and only search plans that respect it. Unstable local
queues are given a local cost just below the deadline, which pushes the
planner towards offloading more. The plan is infeasible (None) if a class's
local queue is still unstable at the fixed point, or if redirects would send
that class to the RSUs more often than the plan allows.

Own mode with a small sampling probability gives a (nearly) periodic chain;
:func:`local_chain` then uses the average of the iterates. Checked against
a direct simulation of the recursion by ``experiments/check_carry.py``.
"""

import numpy as np

from gipa_core import CFG, LOCAL_CARRY, SLOT_MS

_CACHE = {}


def local_chain(f, L, D, own, slot=SLOT_MS, wmax=1500, iters=6000, tol=1e-12):
    """Stationary local-queue statistics for offload probability f.

    Returns dict(share_local, lat_local, miss_local, stable, mean_wait).
    """
    key = (round(float(f), 9), float(L), float(D), bool(own), float(slot))
    if key in _CACHE:
        return _CACHE[key]
    Li, Si = int(round(L)), int(round(slot))
    if f >= 1 - 1e-12:
        out = dict(share_local=0.0, lat_local=float(L), miss_local=0.0, stable=True, mean_wait=0.0)
        _CACHE[key] = out
        return out
    if Li <= Si:
        # The backlog never builds up: every local task finds the processor idle.
        ok = (L <= D) or not own
        share = (1 - f) if ok else 0.0
        out = dict(share_local=float(share), lat_local=float(L), miss_local=float(L > D) if share else 0.0,
                   stable=True, mean_wait=0.0)
        _CACHE[key] = out
        return out
    if not own and (1 - f) * L >= slot:
        out = dict(share_local=1 - f, lat_local=np.inf, miss_local=1.0, stable=False, mean_wait=np.inf)
        _CACHE[key] = out
        return out
    n = wmax + 1
    W = np.arange(n)
    up = np.minimum(W + Li - Si, wmax).clip(0)          # local task run
    dn = np.maximum(W - Si, 0)                           # task offloaded (or redirected)
    run_local = np.ones(n, bool)
    if own:
        run_local = W + L <= D
    p = np.zeros(n)
    p[0] = 1.0
    # Own mode with small f is (nearly) periodic, so plain power iteration may not settle:
    # if it has not converged, use the average of the iterates over the second half (Cesaro mean).
    acc, n_acc, converged = np.zeros(n), 0, False
    for it in range(iters):
        q = np.zeros(n)
        np.add.at(q, dn, p * (f + (1 - f) * (~run_local)))
        np.add.at(q, up, p * (1 - f) * run_local)
        if np.abs(q - p).sum() < tol:
            p, converged = q, True
            break
        p = q
        if own and it >= iters // 2:
            acc += p
            n_acc += 1
    if own and not converged:
        p = acc / n_acc
    loc_w = p * (1 - f) * run_local
    share = float(loc_w.sum())
    lat = float((loc_w * (W + L)).sum() / share) if share > 0 else float(L)
    miss = float((loc_w * (W + L > D)).sum() / share) if share > 0 else 0.0
    out = dict(share_local=share, lat_local=lat, miss_local=miss, stable=bool(p[-50:].sum() < 1e-6),
               mean_wait=lat - L)
    _CACHE[key] = out
    return out


def sampling_share(target, L, D, own):
    """Sampling offload probability whose effective offload share is ``target``.

    Without own-queue redirects the two coincide. With them, the effective
    share 1 - share_local(f) is at least f, and bisection finds the smallest f
    that reaches the target.
    """
    if not own or target <= 0:
        return float(target)

    def eff(f):
        return 1.0 - local_chain(f, L, D, True)['share_local']
    if eff(0.0) >= target:
        return 0.0
    lo, hi = 0.0, float(target)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if eff(mid) < target:
            lo = mid
        else:
            hi = mid
    return hi


def carry_plan(planner, Nbar, R, *args, iters=40, damp=0.5, **kw):
    """Fixed-point planning with carried-over local work.

    ``planner(Nbar, R, *args, **kw)`` is one of the symmetric planners
    (returning x_S, x_B, ...) or the asymmetric one (returning pS, pB, ...).
    Returns the planner's dict with the sampling probabilities adjusted for
    own-queue redirects, plus 'carry' diagnostics, or None if infeasible.
    """
    own = LOCAL_CARRY == 'own'
    M = CFG['M']
    Reff = dict(R)
    if own:
        # Redirect floor: even a vehicle that never samples an RSU sends this share of the
        # class there, because its own queue predicts a local miss. The planners read it as
        # a minimum offload share (MIN_OFF_S, MIN_OFF_B).
        for k in ('S', 'B'):
            floor = 1.0 - local_chain(0.0, R['L_' + k], R['D_' + k], True)['share_local']
            Reff['MIN_OFF_' + k] = max(floor - 1e-9, 0.0)
    plan = None
    for _ in range(iters):
        plan = planner(Nbar, Reff, *args, **kw)
        if plan is None:
            return None
        offS, offB = _offload_shares(plan, M)
        new = {}
        for k, off in (('S', offS), ('B', offB)):
            f = sampling_share(off, R['L_' + k], R['D_' + k], own)
            st = local_chain(f, R['L_' + k], R['D_' + k], own)
            lat = st['lat_local'] if st['stable'] else np.inf
            # Kept below the deadline so the planner's local-deadline rule does not force full offload.
            new['L_' + k] = max(R['L_' + k], min(lat, R['D_' + k] - 1e-6))
        change = max(abs(new[k] - Reff[k]) for k in new)
        for k in new:
            Reff[k] = damp * Reff[k] + (1 - damp) * new[k]
        if change < 1e-4:
            break
    plan = dict(plan)
    offS, offB = _offload_shares(plan, M)
    diag = {}
    for k, off in (('S', offS), ('B', offB)):
        f = sampling_share(off, R['L_' + k], R['D_' + k], own)
        st = local_chain(f, R['L_' + k], R['D_' + k], own)
        diag[k] = dict(target_offload=off, sampling_offload=f, L_eff=Reff['L_' + k], **st)
        # Infeasible if the local queue is unstable (blind), or if the redirects alone
        # already send more tasks to the RSUs than the plan allows (own).
        if not st['stable'] or (own and 1.0 - st['share_local'] > off + 0.01):
            return None
    _set_sampling(plan, M, diag['S']['sampling_offload'], diag['B']['sampling_offload'])
    plan['carry'] = dict(mode=LOCAL_CARRY, **diag)
    return plan


def _offload_shares(plan, M):
    if 'pS' in plan:
        return 1 - plan['pS'][0], 1 - plan['pB'][0]
    return M * plan['x_S'], M * plan['x_B']


def _set_sampling(plan, M, fS, fB):
    # A class whose offload comes entirely from redirects gets a vanishing sampling
    # probability rather than zero, so that redirects (gipa_core.own_redirect) still
    # follow the plan's split across RSUs.
    offS, offB = _offload_shares(plan, M)
    fS = max(fS, 1e-9) if offS > 0 else fS
    fB = max(fB, 1e-9) if offB > 0 else fB
    if 'pS' in plan:
        for key, f in (('pS', fS), ('pB', fB)):
            p = np.asarray(plan[key], float)
            off = 1 - p[0]
            rs = p[1:] / off if off > 0 else np.ones(len(p) - 1) / (len(p) - 1)
            plan[key] = np.concatenate([[1 - f], f * rs]).tolist()
    else:
        plan['x_S'], plan['x_B'] = fS / M, fB / M
