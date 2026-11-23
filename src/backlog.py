"""
backlog.py
==========
Backlog-aware screening of the planned prior (experiment E17, November 2026).

With carried-over local work (``GIPA_LOCAL_CARRY=own``) a background task
(L_B = 22 ms > the 20 ms slot) leaves work on the vehicle's processor. The
carry-aware plan of ``carry.py`` samples a fixed offload share and offloads
by necessity only when the task would miss locally (W + L > D), whatever the
vehicle's backlog W. Here the per-task screening also uses W: a background
vehicle offloads

    with probability 1      if W >= theta  or  W + L_B > D_B,
    with probability q      otherwise,

so that vehicles with long queues offload first, while the fleet's offload
share is still the one the plan prices at the RSUs. The random part q keeps
vehicles from synchronising (a deterministic threshold would make identical
vehicles offload in lock-step from a common start, the herding the paper
avoids). The vehicle needs only its own backlog, as for the existing
redirect; no communication is added.

Planning
--------
For a target background offload share f, :func:`best_screen` considers
every threshold theta (whole milliseconds, 0 ... D_B - L_B, or none) and,
from a table of the exact stationary backlog chain over 401 values of q
(:func:`chain_stats`, solved as a linear system on a 1 ms lattice; all
regime latencies are whole milliseconds), interpolates the q that gives
the offload share f; it returns the policy with the lowest mean local
latency, among those with q >= q_min (thresholds needing a smaller random
part are skipped, which keeps vehicles desynchronised). :func:`backlog_plan`
then repeats the deadline-constrained grid search of
``planner._plan_allocation`` with the background local latency replaced by
that of the best screen for each candidate share (safety tasks, L_S = 10 ms <= slot, never queue). The
policy with no threshold is the carry-aware plan's own policy, so the
screened plan's planned cost is never higher.

Simulation: :func:`run_backlog` mirrors ``planner.run_planned`` (same seeds,
warm-up and metrics).
"""

import numpy as np

from gipa_core import CFG, Env, RNG_ACT_BASE, SLOT_MS, safe_mean, carry_backlog, LOCAL_CARRY

QGRID = np.concatenate([[1e-9], np.linspace(0.0025, 1.0, 400)])
_TABLES = {}


def chain_stats(theta, q, L, D, slot=SLOT_MS):
    """Stationary statistics of one vehicle's backlog under the screen (theta, q), 0 < q <= 1.

    theta = None means no threshold. The stationary distribution is solved
    exactly as a linear system (with 0 < q the backlog chain has a single
    recurrent class). Returns dict(off, share_local, lat_local).
    """
    Li, Si = int(round(L)), int(round(slot))
    n = int(D) + Li + 2
    W = np.arange(n)
    p_off = np.where(W + L > D, 1.0, q)
    if theta is not None:
        p_off = np.where(W >= theta, 1.0, p_off)
    up = np.minimum(np.maximum(W + Li - Si, 0), n - 1)
    dn = np.maximum(W - Si, 0)
    P = np.zeros((n, n))
    np.add.at(P, (W, dn), p_off)
    np.add.at(P, (W, up), 1 - p_off)
    A = np.vstack([P.T - np.eye(n), np.ones(n)])
    rhs = np.zeros(n + 1)
    rhs[-1] = 1.0
    pi = np.linalg.lstsq(A, rhs, rcond=None)[0]
    pi = np.clip(pi, 0, None)
    pi /= pi.sum()
    loc = pi * (1 - p_off)
    share = float(loc.sum())
    lat = float((loc * (W + L)).sum() / share) if share > 0 else float(L)
    return dict(off=1 - share, share_local=share, lat_local=lat)


def _table(L, D, slot):
    key = (L, D, slot)
    if key not in _TABLES:
        thetas = [None] + list(range(0, int(D - L) + 1))
        tab = []
        for th in thetas:
            st = [chain_stats(th, q, L, D, slot) for q in QGRID]
            tab.append((th, np.array([x['off'] for x in st]), np.array([x['lat_local'] for x in st])))
        _TABLES[key] = tab
    return _TABLES[key]


def best_screen(target_off, L, D, slot=SLOT_MS, q_min=0.0):
    """Screen (theta, q) with stationary offload share target_off and lowest local latency, or None.

    For each threshold the offload share rises with q; q is interpolated on a
    401-point grid and the local latency read at that q.
    """
    best = None
    for th, off, lat in _table(L, D, slot):
        if target_off < off[0] - 1e-6 or target_off > off[-1] + 1e-9:
            continue
        q = float(np.interp(target_off, off, QGRID))
        if q < q_min and th is not None:
            continue
        lt = float(np.interp(target_off, off, lat))
        if best is None or lt < best['lat_local'] - 1e-12:
            best = dict(theta=th, q=q, off=float(target_off), lat_local=lt)
    return best


def backlog_plan(Nbar, R, z=2.0, grid_s=61, grid_b=241, q_min=0.0):
    """Deadline-constrained plan whose background tasks are screened by backlog (own mode only)."""
    M = CFG['M']
    XI = CFG['CONG']
    NS = CFG['PI_S'] * Nbar
    NB = Nbar - NS
    if R['L_S'] > SLOT_MS:
        raise ValueError('backlog_plan assumes safety tasks never queue (L_S <= slot)')
    table = {}
    best = None
    for b in np.linspace(0, NB / M, grid_b):
        fB = M * b / NB
        if b not in table:
            table[b] = best_screen(fB, R['L_B'], R['D_B'], q_min=q_min)
        scr = table[b]
        if scr is None:
            continue
        for s in np.linspace(0, NS / M, grid_s):
            n = s + b
            ps, pb = s / NS, b / NB
            sd = np.sqrt(NS * ps * (1 - ps) + NB * pb * (1 - pb))
            n_hi = n + z * sd
            if s > 0 and R['E_S'] + XI * max(n_hi - 1, 0) > R['D_S']:
                continue
            if b > 0 and R['E_B'] + XI * max(n_hi - 1, 0) > R['D_B']:
                continue
            J = ((NS - M * s) * R['L_S'] + M * s * (R['E_S'] + XI * max(n - 1, 0))
                 + (NB - M * b) * scr['lat_local'] + M * b * (R['E_B'] + XI * max(n - 1, 0)))
            if best is None or J < best[0]:
                best = (J, s, b, scr)
    if best is None:
        return None
    J, s, b, scr = best
    return dict(x_S=s / NS, x_B=b / NB, s=float(s), b=float(b), J=float(J), Nbar=float(Nbar), z=float(z), q_min=float(q_min),
                theta=scr['theta'], q=float(scr['q']), lat_local_B=scr['lat_local'], off_B=scr['off'])


def run_backlog(N, seed, R, plan):
    """One episode of the backlog-screened plan; metrics as ``planner.run_planned``."""
    assert LOCAL_CARRY == 'own', 'run with GIPA_LOCAL_CARRY=own'
    M = CFG['M']
    env = Env(N, seed, R)
    rg = np.random.default_rng(RNG_ACT_BASE + seed)
    S = env.isS[:N]
    P = np.empty((N, M + 1))
    P[S] = [1 - M * plan['x_S']] + [plan['x_S']] * M
    P[~S] = [1 - M * plan['x_B']] + [plan['x_B']] * M
    out = dict(mS=[], mB=[], lat=[], fair=[], off=[], offS=[], offB=[], en=[], p95=[], maxl=[])
    for t in range(CFG['SLOTS']):
        u = rg.random((N, 2))
        W = carry_backlog(env, N)
        # safety vehicles: sample the plan (they never queue); background vehicles: screen by own backlog
        offS = u[:, 0] < M * plan['x_S']
        pB = np.where(W + env.loc[:N] > env.dl[:N], 1.0, plan['q'])
        if plan['theta'] is not None:
            pB = np.where(W >= plan['theta'], 1.0, pB)
        offB = u[:, 0] < pB
        off = np.where(S, offS, offB)
        j = np.minimum((u[:, 1] * M).astype(int), M - 1) + 1
        a = np.where(off, j, 0)
        cost, c = env.lat(a, N)
        e = env.en(a, N)
        if t >= CFG['WARM']:
            miss = cost > env.dl[:N]
            out['mS'].append(miss[S].mean() if S.any() else 0.)
            out['mB'].append(miss[~S].mean() if (~S).any() else 0.)
            out['lat'].append(cost.mean())
            out['p95'].append(np.percentile(cost, 95))
            rl = c[1:]
            out['fair'].append(np.nan if rl.sum() == 0 else rl.sum() ** 2 / (len(rl) * (rl ** 2).sum()))
            out['off'].append((a > 0).mean())
            out['offS'].append((a[S] > 0).mean() if S.any() else 0.)
            out['offB'].append((a[~S] > 0).mean() if (~S).any() else 0.)
            out['en'].append(e.mean())
            out['maxl'].append(rl.max())
    return {k: safe_mean(v) for k, v in out.items()}


def agg_backlog(N, R, plan):
    rs = [run_backlog(N, s, R, plan) for s in range(CFG['NSEED'])]
    return {k: safe_mean([r[k] for r in rs]) for k in rs[0]}, rs
