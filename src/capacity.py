"""
capacity.py
===========
RSU capacity, an admission-control layer that wraps any policy, and metrics
that separate what vehicles *selected* from what RSUs *admitted*.

STATUS: written for the first revision of manuscript ITS-2026-08-0579
(workplan experiment E3, answering Reviewer 2 comment 3: "clarify how
capacity is enforced and report RSU overload rates"). Not part of the
submitted v1.0.0 results.

Capacity
--------
With the simulator's affine model, an RSU task of class k finishes in
E_k + CONG * (n - 1) ms when n tasks share the RSU. The *deadline capacity*
of class k is the largest n that still meets that class's deadline:

    Q_k = floor((D_k - E_k) / CONG) + 1.

With the archived constants (CONG = 0.55 ms; E_S = 3, D_S = 12; E_B = 8,
D_B = 40) this gives Q_S = 17 and Q_B = 59 tasks per RSU per slot, in both
regimes. Because every co-located task delays every other task equally, an
RSU holding at least one safety task must stay within Q_S; an RSU holding
only background tasks must stay within Q_B. This mixed limit is called the
effective capacity, Q_eff.

Selection versus admission
--------------------------
* Selected load: the number of vehicles that chose RSU j in a slot (what
  the decision policy did).
* Admitted load: the number of those tasks the RSU accepts.

In *soft* mode (``ac=False``) every selected task is admitted, exactly as in
the v1.0.0 simulator; overload can occur. In *admission-control* mode
(``ac=True``) each RSU considers its selected tasks in a uniformly random
order and admits a task only if, after admitting it, every admitted task
still meets its deadline (the admitted count stays within Q_S when any
admitted task is safety-critical, otherwise within Q_B). A rejected task
falls back to local execution and pays a handshake penalty of
``REJECT_PENALTY_MS`` (1 ms) on top of its local latency. Admission uses
only the vehicle-to-RSU link that offloading already requires; vehicles
exchange no messages.

Applying the same admission layer to every policy means that differences
under admission control reflect selection quality alone.

Seeds
-----
Environment 1000 + s and action sampling 5000 + s, exactly as in
``gipa_core.run``, so results pair with the archived policies. The random
admission order uses a separate generator seeded 9000 + s, so it never
disturbs the action-sampling stream. In soft mode the admission generator
is not used and :func:`run_ac` reproduces ``gipa_core.run`` bit for bit.

Metrics returned by :func:`run_ac` (per-slot values averaged over the 45
evaluated slots, then over seeds by :func:`agg_ac`)
---------------------------------------------------------------------
Standard keys, as in ``gipa_core.run``, computed on the *realised* outcome:
    mS, mB, lat, p95, fair, off, offS, offB, en, maxl
    (``off*`` count admitted offloads; ``fair`` and ``maxl`` use admitted
    loads; a rejected task's energy is E_TX plus its local energy.)
Capacity keys:
    maxl_sel   largest selected RSU load in the slot (tasks)
    maxl_adm   largest admitted RSU load in the slot (tasks)
    over_sel   share of RSU-slots whose selected load exceeds Q_eff
    sev_sel    mean excess (selected load - Q_eff)+ over RSU-slots (tasks)
    over_adm   share of RSU-slots whose admitted load exceeds Q_eff
               (0 by construction under admission control)
    rej        rejected tasks / tasks that selected an RSU (0 in soft mode)
    rej_miss   share of rejected tasks that missed their deadline after
               falling back (NaN in slots without rejections)
"""

import numpy as np

from gipa_core import (CFG, Env, safe_mean, kappa, RNG_ACT_BASE, LOCAL_CARRY, carry_apply,
                       gipa, greedy, masked_softmax, price_rr, alllocal, RBR,
                       prior_heur)

RNG_ADM_BASE = 9000
REJECT_PENALTY_MS = 1.0


def capacities(R):
    """Deadline capacities (Q_S, Q_B) per RSU per slot for regime R.

    Q_k = floor((D_k - E_k) / CONG) + 1. Returns (17, 59) for both archived
    regimes. A small tolerance guards against floating-point round-off when
    (D_k - E_k) / CONG is an exact integer.
    """
    xi = CFG['CONG']
    q_s = int(np.floor((R['D_S'] - R['E_S']) / xi + 1e-9)) + 1
    q_b = int(np.floor((R['D_B'] - R['E_B']) / xi + 1e-9)) + 1
    return q_s, q_b


def effective_capacity(has_safety, q_s, q_b):
    """Q_S if any safety task is present at the RSU, otherwise Q_B."""
    return q_s if has_safety else q_b


def admit(env, a, N, R, rg_adm):
    """Deadline-safe random-order admission at every RSU.

    Parameters
    ----------
    env : gipa_core.Env
    a : ndarray of int, shape (N,)
        Selected actions (0 = local, j >= 1 = RSU j).
    N : int
    R : dict
        Regime (for the capacities).
    rg_adm : numpy.random.Generator
        Admission-order generator (seed 9000 + s).

    Returns
    -------
    ndarray of bool, shape (N,)
        True for tasks that selected an RSU and were rejected.
    """
    q_s, q_b = capacities(R)
    rejected = np.zeros(N, bool)
    isS = env.isS[:N]
    for j in range(1, CFG['M'] + 1):
        order = rg_adm.permutation(np.flatnonzero(a == j))
        n, has_s = 0, False
        for i in order:
            cap = effective_capacity(has_s or isS[i], q_s, q_b)
            if n + 1 <= cap:
                n += 1
                has_s = has_s or bool(isS[i])
            else:
                rejected[i] = True
    return rejected


def select(pol, env, q, cb, rg, N, state, **kw):
    """Dispatch to a decision policy and return its selected actions.

    Mirrors the dispatch in ``gipa_core.run`` so that soft mode reproduces
    it exactly, and adds 'plan' (the planned allocation from
    ``planner.py``; pass ``plan=<dict>``). ``state`` carries per-episode
    objects (the RBR instance).
    """
    if pol == 'gipa':
        return gipa(env, q, cb, rg, N, **kw)
    if pol == 'greedy':
        return greedy(env, q, cb, rg, N)
    if pol == 'indep':
        return masked_softmax(env, q, cb * 0, rg, N)
    if pol == 'msoft':
        return masked_softmax(env, q, cb, rg, N)
    if pol == 'msoft_prior':
        return masked_softmax(env, q, cb, rg, N, use_prior=True)
    if pol == 'entreg':
        return masked_softmax(env, q, cb, rg, N, use_prior=True, tau=kw.get('tau', 1.2))
    if pol == 'price':
        return price_rr(env, q, cb, rg, N)
    if pol == 'rbr':
        return state['rbr'].act(env, q, cb, rg, N)
    if pol == 'ql':
        return kw['ql'].act(env, q, cb, rg, N)
    if pol == 'local':
        return alllocal(env, q, cb, rg, N)
    if pol == 'plan':
        from gipa_core import samp, own_redirect
        from planner import planned_pmf      # imported here to keep planner optional
        if 'P' not in state:
            state['P'] = planned_pmf(env, N, kw['plan'])
        return own_redirect(env, samp(state['P'], rg), state['P'], N, rg)
    raise ValueError(f'unknown policy: {pol}')


def run_ac(pol, N, seed, R, cb=None, ac=False, ql_reset=False, **kw):
    """One episode of policy ``pol``, in soft mode or with admission control.

    Arguments and seeds as in ``gipa_core.run``; ``ac=True`` enables the
    admission layer. ``ql_reset=True`` clears the Q-learner's previous-slot
    load (``QL.mu``) at the start of the episode, so every episode starts
    from the same state; the archived code does not do this (see the known
    issues in docs/SCRIPTS.md). Returns the metric dictionary described in
    the module docstring.
    """
    if pol == 'ql' and ql_reset:
        kw['ql'].mu = None
    env = Env(N, seed, R)
    rg = np.random.default_rng(RNG_ACT_BASE + seed)
    rg_adm = np.random.default_rng(RNG_ADM_BASE + seed)
    cb = prior_heur(N) if cb is None else cb
    state = {'rbr': RBR()}
    q_s, q_b = capacities(R)
    M = CFG['M']
    keys = ['mS', 'mB', 'lat', 'fair', 'off', 'offS', 'offB', 'en', 'p95', 'maxl',
            'maxl_sel', 'maxl_adm', 'over_sel', 'sev_sel', 'over_adm', 'rej', 'rej_miss']
    out = {k: [] for k in keys}
    isS = env.isS[:N]

    for t in range(CFG['SLOTS']):
        q = env.q(N)
        a = select(pol, env, q, cb, rg, N, state, **kw)
        c_sel = np.bincount(a, minlength=M + 1)
        if ac:
            rejected = admit(env, a, N, R, rg_adm)
        else:
            rejected = np.zeros(N, bool)
        a_adm = np.where(rejected, 0, a)
        c_adm = np.bincount(a_adm, minlength=M + 1)

        # Realised latency: admitted RSU tasks share the admitted load;
        # rejected tasks run locally after a handshake penalty.
        cg = np.zeros(M + 1)
        cg[1:] = kappa(c_adm[1:])
        cost = np.where(a_adm == 0, env.loc[:N], env.e0[:N] + cg[a_adm])
        cost = np.where(rejected, env.loc[:N] + REJECT_PENALTY_MS, cost)
        if LOCAL_CARRY:
            # Rejected tasks also run on the vehicle's processor; each local task waits for the backlog.
            ran_local = (a_adm == 0)
            cost = carry_apply(env, np.where(ran_local, 0, 1), cost, env.loc[:N], N)
        el = np.where(isS, CFG['E_LS'], CFG['E_LB'])
        e = np.where(a_adm == 0, el, CFG['E_TX'])
        e = np.where(rejected, el + CFG['E_TX'], e)

        if t >= CFG['WARM']:
            S = isS
            miss = cost > env.dl[:N]
            out['mS'].append(miss[S].mean() if S.any() else 0.)
            out['mB'].append(miss[~S].mean() if (~S).any() else 0.)
            out['lat'].append(cost.mean())
            out['p95'].append(np.percentile(cost, 95))
            rl = c_adm[1:]
            out['fair'].append(np.nan if rl.sum() == 0
                               else rl.sum() ** 2 / (len(rl) * (rl ** 2).sum()))
            out['off'].append((a_adm > 0).mean())
            out['offS'].append((a_adm[S] > 0).mean() if S.any() else 0.)
            out['offB'].append((a_adm[~S] > 0).mean() if (~S).any() else 0.)
            out['en'].append(e.mean())
            out['maxl'].append(rl.max())

            # Capacity metrics per RSU.
            over_sel, sev_sel, over_adm = [], [], []
            for j in range(1, M + 1):
                q_sel = effective_capacity(bool((S & (a == j)).any()), q_s, q_b)
                q_adm = effective_capacity(bool((S & (a_adm == j)).any()), q_s, q_b)
                over_sel.append(c_sel[j] > q_sel)
                sev_sel.append(max(c_sel[j] - q_sel, 0))
                over_adm.append(c_adm[j] > q_adm)
            out['maxl_sel'].append(c_sel[1:].max())
            out['maxl_adm'].append(rl.max())
            out['over_sel'].append(np.mean(over_sel))
            out['sev_sel'].append(np.mean(sev_sel))
            out['over_adm'].append(np.mean(over_adm))
            n_sel = (a > 0).sum()
            out['rej'].append(rejected.sum() / n_sel if n_sel else 0.)
            out['rej_miss'].append(miss[rejected].mean() if rejected.any() else np.nan)

    return {k: safe_mean(v) for k, v in out.items()}


def agg_ac(pol, N, R, cb=None, ac=False, ql_reset=False, **kw):
    """Run :func:`run_ac` for seed indices 0..NSEED-1.

    Returns (across-seed means, list of per-seed dicts), mirroring
    ``gipa_core.agg``.
    """
    rs = [run_ac(pol, N, s, R, cb, ac, ql_reset, **kw) for s in range(CFG['NSEED'])]
    return {k: safe_mean([r[k] for r in rs]) for k in rs[0]}, rs
