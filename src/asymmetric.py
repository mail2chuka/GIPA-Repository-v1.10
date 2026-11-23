"""
asymmetric.py
=============
Asymmetric RSUs for workplan experiment E9: does GIPA avoid herding when one
RSU is clearly the best? This stress-tests the manuscript's core
("anti-herding") claim; with identical RSUs there is no obvious best RSU to
herd onto.

STATUS: written for the first revision of manuscript ITS-2026-08-0579. Not
part of the submitted v1.0.0 results. ``gipa_core`` is unchanged.

Scenario
--------
RSU j has a speed factor sp_j that multiplies everything that RSU does:
its uncongested latency and its congestion. The default scenario 'fast1',
sp = (0.7, 1, 1), makes RSU 1 30% faster (lower uncongested cost and lower
congestion slope) and leaves RSUs 2 and 3 as before. 'sym' (1, 1, 1) is the
archived symmetric system and reproduces ``gipa_core`` exactly.

Affine model (``AsymEnv``):
    latency at RSU j = sp_j * (E_k + CONG * max(n_j - 1, 0))
Eq. (5) model (``AsymPSEnv``, default calibration of E2):
    safety at RSU j     = sp_j * (T_S + a_S n_j)
    background at RSU j = sp_j * (T_B + a_S nS_j + a_B nB_j)
Intrinsic scores are q_{i,j} = -sp_j * e_k (uncongested), so a vehicle's
scores now rank the fast RSU first.

Priors (per-vehicle matrices, local column 0)
--------------------------------------------
'uniform'   the archived Eq. (13): the fleet is assumed to split evenly over
            the M + 1 actions and every RSU gets the same delay, ignoring
            speed (affine: CONG * max(N/(M+1) - 1, 0); Eq. 5: the E2
            class-aware prior). This is what GIPA would use unchanged.
'capacity'  recalibrated Eq. (12): the expected offloaded load
            N * M / (M + 1) is split in proportion to capacity
            w_j = 1 / sp_j, and each RSU's delay is computed with its own
            speed. With sp = (0.7, 1, 1) this roughly equalises the
            congestion delay across RSUs.

Planner (``plan_asym``)
-----------------------
System-optimal allocation with per-RSU expected safety and background
counts (s_j, b_j), found with SLSQP from several starts. Objective: the
expected total latency per slot under the chosen model. Constraints: each
class using RSU j meets its deadline at the load n_j + z * sd_j (binomial
sd), written in complementarity form x * (latency - deadline) <= 0; tasks
kept local must meet the local deadline (in Regime B every safety task
must be offloaded). Vehicles then sample from the per-class PMF
(1 - sum_j s_j / N_S, s_1 / N_S, ...). With sp = (1, 1, 1) it reproduces the
grid planner of ``planner.py`` (checked in :func:`check_planner`).

Herding metrics (per slot, then averaged over evaluated slots)
--------------------------------------------------------------
herd_fast   load of RSU 1 divided by its capacity-weighted fair share of
            the offloaded load, n_1 / (n_off * w_1 / sum w); 1 = fair,
            > 1 = herding onto the fast RSU (NaN when nothing is offloaded)
cv_raw      coefficient of variation of the RSU loads n_j
cv_cap      coefficient of variation of the capacity-normalised loads
            n_j / w_j (0 = load exactly proportional to capacity)
same_choice probability that two vehicles of the same class (and so the
            same scores) choose the same action, estimated within the slot
            as sum_a c_a (c_a - 1) / (n (n - 1)), averaged over the classes
swing       slot-to-slot load swing sum_j |n_j(t) - n_j(t - 1)| / N
overload    1 if any RSU has at least one deadline miss in the slot
plus the ``gipa_core.run`` metrics.
"""

import numpy as np

from gipa_core import (CFG, Env, RNG_ACT_BASE, alllocal, greedy, masked_softmax, safe_mean, own_redirect,
                       samp)

SCENARIOS = {'sym': (1.0, 1.0, 1.0), 'fast1': (0.7, 1.0, 1.0)}


# ----------------------------------------------------------------------
# Environments
# ----------------------------------------------------------------------
class AsymEnv(Env):
    """``gipa_core.Env`` with per-RSU speed factors (affine model)."""

    def __init__(self, N, seed, R, sp):
        super().__init__(N, seed, R)
        self.sp = np.asarray(sp, float)

    def q(self, N):
        q = np.empty((N, self.n))
        q[:, 0] = -self.loc[:N]
        q[:, 1:] = -self.e0[:N, None] * self.sp[None, :]
        return q

    def lat(self, a, N):
        c = np.bincount(a, minlength=self.n)
        spa = np.concatenate([[0.0], self.sp])[a]
        cong = CFG['CONG'] * np.maximum(c[a] - 1.0, 0.0)
        return np.where(a == 0, self.loc[:N], spa * (self.e0[:N] + cong)), c


class AsymPSEnv(Env):
    """Eq. (5) batch processor sharing with per-RSU speed factors."""

    def __init__(self, N, seed, R, sp, cal):
        super().__init__(N, seed, R)
        self.sp = np.asarray(sp, float)
        self.cal = cal
        self.e0 = np.where(self.isS, cal['e_S'], cal['e_B'])
        self.T = np.where(self.isS, cal['T_S'], cal['T_B'])

    def q(self, N):
        q = np.empty((N, self.n))
        q[:, 0] = -self.loc[:N]
        q[:, 1:] = -self.e0[:N, None] * self.sp[None, :]
        return q

    def lat(self, a, N):
        S = self.isS[:N]
        c = np.bincount(a, minlength=self.n)
        nS = np.bincount(a[S], minlength=self.n)
        nB = c - nS
        a_S, a_B = self.cal['a_S'], self.cal['a_B']
        comp = np.where(S, a_S * c[a], a_S * nS[a] + a_B * nB[a])
        spa = np.concatenate([[0.0], self.sp])[a]
        return np.where(a == 0, self.loc[:N], spa * (self.T[:N] + comp)), c


def make_env(N, seed, R, sp, model):
    if model == 'affine':
        return AsymEnv(N, seed, R, sp)
    from calibration import make_calibration
    return AsymPSEnv(N, seed, R, sp, make_calibration())


# ----------------------------------------------------------------------
# Priors
# ----------------------------------------------------------------------
def rsu_loads(N, sp, kind):
    """Anticipated tasks per RSU (length M) under the prior's assumption."""
    M = CFG['M']
    sp = np.asarray(sp, float)
    if kind == 'uniform':
        return np.full(M, N / (M + 1))
    if kind == 'capacity':
        w = 1.0 / sp
        return N * M / (M + 1) * w / w.sum()
    raise ValueError(kind)


def prior_asym(env, N, model, kind):
    """Per-vehicle prior matrix (N x (M + 1)); see the module docstring.

    'uniform' ignores speed in the delay as well as in the load, so with
    sp = (1, 1, 1) it equals ``gipa_core.prior_heur`` (affine) or the E2
    class-aware prior (Eq. 5).
    """
    M = CFG['M']
    S = env.isS[:N]
    nbar = rsu_loads(N, env.sp, kind)
    spd = np.ones(M) if kind == 'uniform' else env.sp
    C = np.zeros((N, M + 1))
    if model == 'affine':
        C[:, 1:] = (spd * CFG['CONG'] * np.maximum(nbar - 1, 0))[None, :]
    else:
        cal, pi = env.cal, CFG['PI_S']
        C[S, 1:] = spd * cal['a_S'] * np.maximum(nbar - 1, 0)
        C[~S, 1:] = spd * (cal['a_S'] * pi * nbar + cal['a_B'] * np.maximum((1 - pi) * nbar - 1, 0))
    return C


# ----------------------------------------------------------------------
# Planner
# ----------------------------------------------------------------------
def _plan_terms(x, NS, NB, R, sp, model, cal):
    """Expected per-RSU latency for each class given counts x = (s_1..s_M, b_1..b_M)."""
    M = len(sp)
    s, b = x[:M], x[M:]
    n = s + b
    if model == 'affine':
        cong = CFG['CONG'] * np.maximum(n - 1, 0)
        lS = sp * (R['E_S'] + cong)
        lB = sp * (R['E_B'] + cong)
    else:
        lS = sp * (cal['T_S'] + cal['a_S'] * n)
        lB = sp * (cal['T_B'] + cal['a_S'] * s + cal['a_B'] * b)
    return s, b, n, lS, lB


def plan_asym(Nbar, R, sp, model='affine', z=2.0, starts=12, seed=0):
    """Asymmetric planned allocation (see :func:`_plan_asym`); carry-aware when GIPA_LOCAL_CARRY is set."""
    from gipa_core import LOCAL_CARRY
    if LOCAL_CARRY:
        from carry import carry_plan
        return carry_plan(_plan_asym, Nbar, R, sp, model=model, z=z, starts=starts, seed=seed)
    return _plan_asym(Nbar, R, sp, model=model, z=z, starts=starts, seed=seed)


def _plan_asym(Nbar, R, sp, model='affine', z=2.0, starts=12, seed=0):
    """System-optimal asymmetric allocation; dict with per-RSU s, b, PMFs, J (or None)."""
    from scipy.optimize import minimize
    cal = None
    if model != 'affine':
        from calibration import make_calibration
        cal = make_calibration()
    sp = np.asarray(sp, float)
    M = len(sp)
    NS = CFG['PI_S'] * Nbar
    NB = Nbar - NS

    def hi_terms(x):
        s, b = x[:M], x[M:]
        ps, pb = np.clip(s / NS, 0, 1), np.clip(b / NB, 0, 1)
        sd = np.sqrt(NS * ps * (1 - ps) + NB * pb * (1 - pb))
        n = s + b
        k = np.where(n > 1e-9, (n + z * sd) / np.maximum(n, 1e-9), 1.0)
        return _plan_terms(np.concatenate([s * k, b * k]), NS, NB, R, sp, model, cal)

    def J(x):
        s, b, n, lS, lB = _plan_terms(x, NS, NB, R, sp, model, cal)
        return ((NS - s.sum()) * R['L_S'] + (s * lS).sum()
                + (NB - b.sum()) * R['L_B'] + (b * lB).sum())

    cons = [dict(type='ineq', fun=lambda x: -x[:M] * (hi_terms(x)[3] - R['D_S'])),
            dict(type='ineq', fun=lambda x: -x[M:] * (hi_terms(x)[4] - R['D_B'])),
            dict(type='ineq', fun=lambda x: NS - x[:M].sum()),
            dict(type='ineq', fun=lambda x: NB - x[M:].sum())]
    if R['L_S'] > R['D_S']:
        cons.append(dict(type='eq', fun=lambda x: x[:M].sum() - NS))
    if R['L_B'] > R['D_B']:
        cons.append(dict(type='eq', fun=lambda x: x[M:].sum() - NB))
    # Minimum offload shares (set only by carry.carry_plan in 'own' mode).
    if R.get('MIN_OFF_S', 0.0) > 0:
        cons.append(dict(type='ineq', fun=lambda x: x[:M].sum() - R['MIN_OFF_S'] * NS))
    if R.get('MIN_OFF_B', 0.0) > 0:
        cons.append(dict(type='ineq', fun=lambda x: x[M:].sum() - R['MIN_OFF_B'] * NB))
    bounds = [(0, NS)] * M + [(0, NB)] * M
    rg = np.random.default_rng(seed)
    w = (1 / sp) / (1 / sp).sum()
    x0s = [np.concatenate([f1 * NS * w, f2 * NB * w]) for f1 in (0.0, 0.3, 1.0) for f2 in (0.3, 0.6)]
    x0s += [np.concatenate([rg.random(M) * NS / M, rg.random(M) * NB / M]) for _ in range(starts)]
    best = None
    for x0 in x0s:
        r = minimize(J, x0, method='SLSQP', bounds=bounds, constraints=cons,
                     options=dict(maxiter=500, ftol=1e-10))
        x = np.clip(r.x, 0, None)
        ok = all((c['fun'](x) >= -1e-6).all() if c['type'] == 'ineq' else abs(c['fun'](x)) < 1e-6 for c in cons)
        if ok and (best is None or J(x) < best[0]):
            best = (J(x), x)
    if best is None:
        return None
    Jv, x = best
    s, b = x[:M], x[M:]
    pS = np.concatenate([[max(1 - s.sum() / NS, 0)], s / NS]) if NS else np.r_[1, np.zeros(M)]
    pB = np.concatenate([[max(1 - b.sum() / NB, 0)], b / NB]) if NB else np.r_[1, np.zeros(M)]
    return dict(s=s.tolist(), b=b.tolist(), pS=(pS / pS.sum()).tolist(), pB=(pB / pB.sum()).tolist(),
                J=float(Jv), Nbar=float(Nbar), z=float(z), sp=sp.tolist(), model=model)


def plan_pmf(env, N, plan):
    P = np.empty((N, CFG['M'] + 1))
    S = env.isS[:N]
    P[S] = plan['pS']
    P[~S] = plan['pB']
    return P


def check_planner(Ns=(80, 160, 240, 320)):
    """Max relative gap between plan_asym with sp = (1, 1, 1) and the grid planner (expected J)."""
    from gipa_core import REG
    from planner import plan_allocation
    gaps = []
    for reg in ['A', 'B']:
        for N in Ns:
            g = plan_allocation(N, REG[reg], z=2.0)
            a = plan_asym(N, REG[reg], (1, 1, 1), 'affine', z=2.0)
            if g is None or a is None:
                gaps.append(0.0 if (g is None) == (a is None) else np.inf)
                continue
            gaps.append((a['J'] - g['J']) / g['J'])
    return float(max(gaps)), gaps


# ----------------------------------------------------------------------
# Policies and episode runner
# ----------------------------------------------------------------------
def rbr_asym(env, q, mu, rg, N, model):
    """Reactive best response to last slot's shares, with each RSU's own slope."""
    slope = CFG['CONG'] if model == 'affine' else env.cal['a_S'] * CFG['PI_S'] + env.cal['a_B'] * (1 - CFG['PI_S'])
    cg = np.zeros(env.n)
    cg[1:] = env.sp * slope * np.maximum(mu[1:] * N - 1.0, 0.0)
    z = (q - cg[None, :]) / CFG['TAU']
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    p = p / p.sum(1, keepdims=True)
    return samp(p, rg)


POLICIES = ['local', 'greedy', 'indep', 'msoft', 'msoft_prior', 'entreg', 'gipa', 'gipa_nodiff',
            'rbr', 'plan', 'la']
PRIOR_POLICIES = ['msoft', 'msoft_prior', 'entreg', 'gipa', 'gipa_nodiff']


def run_asym(pol, N, seed, R, sp, model='affine', prior='capacity', plan=None, la_cfg=None,
             warm=CFG['WARM']):
    """One episode; returns the gipa_core.run metrics plus the herding metrics.

    ``warm`` learning/warm-up slots precede the 45 evaluated slots (the
    archived 25 for every policy except 'la', which E9 runs with 500).
    """
    from calibration import gipa_pv, msoft_pv
    env = make_env(N, seed, R, sp, model)
    rg = np.random.default_rng(RNG_ACT_BASE + seed)
    C = prior_asym(env, N, model, prior)
    q = env.q(N)
    M = CFG['M']
    w = 1.0 / env.sp
    mu = np.full(M + 1, 1.0 / (M + 1))
    P = plan_pmf(env, N, plan) if pol == 'plan' else None
    if pol == 'la':
        from learning_automata import Automata
        la = Automata(env, N, la_cfg['algo'], la_cfg['resp'], la_cfg['rate'], seed)
        la.activate(N)
    S = env.isS[:N]
    keys = ['mS', 'mB', 'lat', 'p95', 'fair', 'off', 'offS', 'offB', 'maxl',
            'herd_fast', 'cv_raw', 'cv_cap', 'same_choice', 'swing', 'overload', 'load1', 'load2', 'load3']
    out = {k: [] for k in keys}
    prev = None
    n_eval = CFG['SLOTS'] - CFG['WARM']
    for t in range(warm + n_eval):
        if pol == 'gipa':
            a = gipa_pv(env, q, C, rg, N)
        elif pol == 'gipa_nodiff':
            a = gipa_pv(env, q, C, rg, N, lam_on=False)
        elif pol == 'msoft':
            a = msoft_pv(env, q, C, rg, N)
        elif pol == 'msoft_prior':
            a = msoft_pv(env, q, C, rg, N, use_prior=True)
        elif pol == 'entreg':
            a = msoft_pv(env, q, C, rg, N, use_prior=True, tau=1.2)
        elif pol == 'indep':
            a = masked_softmax(env, q, np.zeros(M + 1), rg, N)
        elif pol == 'greedy':
            a = greedy(env, q, None, rg, N)
        elif pol == 'local':
            a = alllocal(env, q, None, rg, N)
        elif pol == 'rbr':
            a = rbr_asym(env, q, mu, rg, N, model)
        elif pol == 'plan':
            a = own_redirect(env, samp(P, rg), P, N, rg)
        elif pol == 'la':
            a = la.act(rg, N)
        else:
            raise ValueError(pol)
        cost, c = env.lat(a, N)
        if pol == 'la':
            la.update(a, cost, N)
        mu = c / N
        rl = c[1:].astype(float)
        if t >= warm:
            miss = cost > env.dl[:N]
            out['mS'].append(miss[S].mean() if S.any() else 0.)
            out['mB'].append(miss[~S].mean() if (~S).any() else 0.)
            out['lat'].append(cost.mean())
            out['p95'].append(np.percentile(cost, 95))
            out['fair'].append(np.nan if rl.sum() == 0 else rl.sum() ** 2 / (M * (rl ** 2).sum()))
            out['off'].append((a > 0).mean())
            out['offS'].append((a[S] > 0).mean() if S.any() else 0.)
            out['offB'].append((a[~S] > 0).mean() if (~S).any() else 0.)
            out['maxl'].append(rl.max())
            noff = rl.sum()
            out['herd_fast'].append(np.nan if noff == 0 else rl[0] / (noff * w[0] / w.sum()))
            out['cv_raw'].append(np.nan if noff == 0 else rl.std() / rl.mean())
            nl = rl / w
            out['cv_cap'].append(np.nan if noff == 0 else nl.std() / nl.mean())
            sc = []
            for cls in (S, ~S):
                n = cls.sum()
                if n > 1:
                    cc = np.bincount(a[cls], minlength=M + 1)
                    sc.append((cc * (cc - 1)).sum() / (n * (n - 1)))
            out['same_choice'].append(np.mean(sc))
            out['swing'].append(np.nan if prev is None else np.abs(rl - prev).sum() / N)
            out['overload'].append(float(any(miss[a == j].any() for j in range(1, M + 1))))
            for j in range(M):
                out[f'load{j + 1}'].append(rl[j])
        prev = rl
    return {k: safe_mean(v) for k, v in out.items()}


def agg_asym(pol, N, R, sp, seeds=None, **kw):
    seeds = range(CFG['NSEED']) if seeds is None else seeds
    rs = [run_asym(pol, N, s, R, sp, **kw) for s in seeds]
    return {k: safe_mean([r[k] for r in rs]) for k in rs[0]}, rs
