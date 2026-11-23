"""
planner.py
==========
System-optimal planned allocation for communication-free offloading.

STATUS: exploratory, written for the first revision of manuscript
ITS-2026-08-0579. It is NOT part of the submitted v1.0.0 results.

Idea
----
GIPA's heuristic prior (``gipa_core.prior_heur``) makes each vehicle react to
the congestion it expects for itself. The fleet then settles near the
*user equilibrium*: every action in use costs about the same, which is why
GIPA's mean latency matches all-local execution in Regime A.

This module plans the *system optimum* instead. In Stage 0, offline, it
chooses how many safety and background tasks each RSU should receive in
expectation so that the fleet's total expected latency is minimal, subject to
deadlines. Stage 1 is unchanged in spirit: every vehicle samples its action
independently from the planned per-class PMF, with no inter-vehicle messages
and no runtime load information. The planning inputs are the same as those
of the existing prior: anticipated fleet size N_bar, class mix pi_S, the
number of RSUs M and the regime's latency constants.

Model and assumptions
---------------------
* The latency model is exactly the simulator's affine model
  (``gipa_core.kappa``): an RSU task costs E_k + CONG * (n - 1), where n is
  the number of tasks at that RSU in the slot.
* RSUs are identical, so the plan is symmetric: every RSU receives an
  expected s safety and b background tasks per slot.
* The plan uses expected counts; realised loads are random because vehicles
  sample independently. An optional margin of ``z`` standard deviations of
  the per-RSU load (binomial approximation) is added before the deadline
  check to protect against that randomness.
* Tasks left local must meet their local deadline. In Regime B this forces
  every safety task to the RSUs, and the plan is infeasible when the RSUs
  cannot absorb them within the safety deadline.

Optimisation
------------
Exhaustive grid search over s in [0, N_S / M] (``grid_s`` points) and
b in [0, N_B / M] (``grid_b`` points), with N_S = pi_S * N_bar and
N_B = N_bar - N_S. The objective is the expected total latency

    J(s, b) = (N_S - M s) L_S + M s (E_S + CONG max(n - 1, 0))
            + (N_B - M b) L_B + M b (E_B + CONG max(n - 1, 0)),   n = s + b.

Typical use
-----------
>>> from gipa_core import REG
>>> from planner import plan_allocation, agg_planned
>>> plan = plan_allocation(160, REG['A'])
>>> means, per_seed = agg_planned(160, REG['A'], plan)
"""

import numpy as np

from gipa_core import CFG, Env, samp, safe_mean, RNG_ACT_BASE, own_redirect


def plan_allocation(Nbar, R, z=0.0, grid_s=61, grid_b=241, xi=None):
    """System-optimal planned allocation (see :func:`_plan_allocation`).

    With GIPA_LOCAL_CARRY set, the plan accounts for carried-over local work
    (``carry.carry_plan``); otherwise it is :func:`_plan_allocation` exactly.
    """
    from gipa_core import LOCAL_CARRY
    if LOCAL_CARRY:
        from carry import carry_plan
        return carry_plan(_plan_allocation, Nbar, R, z=z, grid_s=grid_s, grid_b=grid_b, xi=xi)
    return _plan_allocation(Nbar, R, z=z, grid_s=grid_s, grid_b=grid_b, xi=xi)


def _plan_allocation(Nbar, R, z=0.0, grid_s=61, grid_b=241, xi=None):
    """Compute the deadline-constrained system-optimal symmetric allocation.

    Parameters
    ----------
    Nbar : float
        Anticipated fleet size used for planning (may differ from the true N).
    R : dict
        Regime from ``gipa_core.REG`` (deadlines, local and RSU latencies, ms).
    z : float, default 0.0
        Safety margin in standard deviations of the per-RSU load under
        independent sampling. 0 plans on expected loads only.
    grid_s, grid_b : int
        Grid resolution for safety and background tasks per RSU.
    xi : float, optional
        Congestion slope (ms per extra co-located task) assumed by the
        planner. Defaults to the simulator's true value CFG['CONG']; pass a
        different value to study a misestimated slope (experiment E1).

    Returns
    -------
    dict or None
        None if no allocation satisfies the constraints. Otherwise:
        ``x_S``, ``x_B`` : probability that a safety / background vehicle
        selects any one given RSU (the local probability is 1 - M x);
        ``s``, ``b`` : expected safety / background tasks per RSU;
        ``J`` : expected total latency of the fleet per slot (ms);
        ``Nbar``, ``z`` : the planning inputs.
    """
    M = CFG['M']
    XI = CFG['CONG'] if xi is None else xi
    NS = CFG['PI_S'] * Nbar
    NB = Nbar - NS
    best = None
    for s in np.linspace(0, NS / M, grid_s):
        for b in np.linspace(0, NB / M, grid_b):
            n = s + b
            # Binomial standard deviation of one RSU's load under independent sampling.
            ps = s / NS if NS else 0
            pb = b / NB if NB else 0
            sd = np.sqrt(NS * ps * (1 - ps) + NB * pb * (1 - pb))
            n_hi = n + z * sd

            def rsu_latency(e):
                return e + XI * max(n_hi - 1, 0)

            feasible = True
            if s > 0 and rsu_latency(R['E_S']) > R['D_S']:
                feasible = False
            if b > 0 and rsu_latency(R['E_B']) > R['D_B']:
                feasible = False
            # Tasks left on the vehicle must meet their local deadline.
            if R['L_S'] > R['D_S'] and NS - M * s > 1e-9:
                feasible = False
            if R['L_B'] > R['D_B'] and NB - M * b > 1e-9:
                feasible = False
            # Minimum offload shares (set only by carry.carry_plan in 'own' mode).
            if M * s < R.get('MIN_OFF_S', 0.0) * NS - 1e-9 or M * b < R.get('MIN_OFF_B', 0.0) * NB - 1e-9:
                feasible = False
            if not feasible:
                continue
            J = ((NS - M * s) * R['L_S'] + M * s * (R['E_S'] + XI * max(n - 1, 0))
                 + (NB - M * b) * R['L_B'] + M * b * (R['E_B'] + XI * max(n - 1, 0)))
            if best is None or J < best[0]:
                best = (J, s, b)
    if best is None:
        return None
    J, s, b = best
    return dict(x_S=s / NS if NS else 0.0, x_B=b / NB if NB else 0.0,
                s=float(s), b=float(b), J=float(J), Nbar=float(Nbar), z=float(z))


def planned_pmf(env, N, plan):
    """Per-vehicle action PMFs implied by a plan.

    Returns an (N, M + 1) row-stochastic matrix: a safety vehicle chooses each
    RSU with probability x_S and stays local with 1 - M x_S; likewise for
    background vehicles with x_B.
    """
    M = CFG['M']
    P = np.empty((N, M + 1))
    S = env.isS[:N]
    P[S] = [1 - M * plan['x_S']] + [plan['x_S']] * M
    P[~S] = [1 - M * plan['x_B']] + [plan['x_B']] * M
    return P


def run_planned(N, seed, R, plan):
    """One episode in which every vehicle samples from the planned PMF.

    Uses the same environment seed (1000 + seed), action seed (5000 + seed),
    slot count, warm-up and metric definitions as ``gipa_core.run``, so the
    results are directly comparable with, and paired to, the archived
    policies. See ``gipa_core.run`` for the metric keys.
    """
    env = Env(N, seed, R)
    rg = np.random.default_rng(RNG_ACT_BASE + seed)
    P = planned_pmf(env, N, plan)
    out = dict(mS=[], mB=[], lat=[], fair=[], off=[], offS=[], offB=[],
               en=[], p95=[], maxl=[])
    for t in range(CFG['SLOTS']):
        a = own_redirect(env, samp(P, rg), P, N, rg)
        cost, c = env.lat(a, N)
        e = env.en(a, N)
        if t >= CFG['WARM']:
            S = env.isS[:N]
            miss = cost > env.dl[:N]
            out['mS'].append(miss[S].mean() if S.any() else 0.)
            out['mB'].append(miss[~S].mean() if (~S).any() else 0.)
            out['lat'].append(cost.mean())
            out['p95'].append(np.percentile(cost, 95))
            rl = c[1:]
            out['fair'].append(np.nan if rl.sum() == 0
                               else rl.sum() ** 2 / (len(rl) * (rl ** 2).sum()))
            out['off'].append((a > 0).mean())
            out['offS'].append((a[S] > 0).mean() if S.any() else 0.)
            out['offB'].append((a[~S] > 0).mean() if (~S).any() else 0.)
            out['en'].append(e.mean())
            out['maxl'].append(rl.max())
    return {k: safe_mean(v) for k, v in out.items()}


def agg_planned(N, R, plan):
    """Run :func:`run_planned` for seed indices 0..NSEED-1 and average.

    Returns (across-seed means, list of per-seed dicts), mirroring
    ``gipa_core.agg``.
    """
    rs = [run_planned(N, s, R, plan) for s in range(CFG['NSEED'])]
    return {k: safe_mean([r[k] for r in rs]) for k in rs[0]}, rs
