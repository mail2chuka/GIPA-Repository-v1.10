"""
learning_automata.py
====================
Learning-automata (LA) baselines for workplan experiment E6, answering
Reviewer 1 comment 3 (compare directly with learning-automata offloading,
which learns from feedback rather than from a prior).

STATUS: written for the first revision of manuscript ITS-2026-08-0579. Not
part of the submitted v1.0.0 results. Nothing in ``gipa_core`` is changed.

Model
-----
Every vehicle i carries its own probability vector p_i over the actions
{0 = local, 1..M = RSU j}. The vector persists from slot to slot (and, in
the density-change test, across changes in fleet size). Only actions in the
vehicle's feasible set are given probability; the set is fixed per vehicle
because intrinsic costs do not change. Each slot:

1. every active vehicle samples a_i ~ p_i with the archived action generator
   (``RNG_ACT_BASE + seed``, one uniform per vehicle, via ``gipa_core.samp``),
   so results are paired with every other policy;
2. the environment returns realised latencies;
3. each vehicle computes its own environment response beta_i from its own
   latency only (no communication, no load broadcast) and updates p_i.

Feasible set (``mask``)
    'deadline' (default): an action is feasible when its uncongested
    intrinsic latency meets the deadline (local always included unless it
    misses; if nothing is feasible, local). This is the same deadline-only
    mask as MAPPO's fair comparison at N >= 240.
    'prior': GIPA's prior-inclusive mask (intrinsic cost + heuristic prior),
    for reference. With it, every offload is masked at N >= 240 in Regime A,
    so LA cannot offload at all.

Environment response (``response``)
    'binary' (P-model, the workplan specification): beta = 1 if the realised
    latency meets the deadline and is no worse than the vehicle's local
    latency, else 0. Note that in Regime A the local action always earns
    beta = 1, so the binary response cannot tell "local" from "useful
    offload".
    'snorm' (S-model): beta = max(0, 1 - latency / deadline) if the deadline
    is met, else 0. Graded, so a faster action earns more reinforcement.

Update schemes (``algo``), rate a, chosen action k, r feasible actions
    'lri'   Linear reward-inaction L_R-I:
                p <- p + a * beta * (e_k - p)
            (P-model: p_k += a (1 - p_k), p_j *= (1 - a) on reward; no change
            on penalty.)
    'lrep'  Linear reward-epsilon-penalty L_R-eP, eps = 0.1:
                p <- p + a * beta * (e_k - p)
                       + eps * a * (1 - beta) * (u_k - p)
            where u_k is uniform over the r - 1 other feasible actions (on
            penalty p_k shrinks and the others move toward uniform).
    'pursuit' Pursuit LA: each vehicle keeps reward estimates d_i(j)
            (sample average with step max(1 / count, 0.05), so it can track
            change), and moves toward the action with the highest estimate:
                p <- (1 - a) p + a * e_best
            Untried feasible actions have an optimistic estimate of 2.
            Ties are broken by a fixed per-vehicle jitter (generator
            ``RNG_LA_JITTER + seed``), so vehicles do not all pick RSU 1.

Probabilities start uniform over the feasible set and are renormalised every
slot to remove rounding drift.

Episodes
--------
:func:`run_la` runs ``warm`` learning slots followed by the 45 evaluation
slots of the archived protocol (``CFG['SLOTS'] - CFG['WARM']``). Learning
continues during evaluation (online). ``warm = 25`` gives the archived
70-slot episode; ``warm = 500`` gives LA time to converge. Metrics match
``gipa_core.run``.

:func:`run_schedule` runs a fleet-size schedule (the non-stationary test):
the environment is drawn for the largest fleet and the first N(t) vehicles
are active in slot t. Arriving vehicles start with uniform probabilities;
leaving vehicles are dropped. The same runner also drives GIPA (with a prior
refreshed to the current N, or a stale prior), the planner (re-planned or
stale) and all-local, so the traces are paired.

Seeds
-----
Evaluation: archived seed indices 0..29 (environment 1000 + s, actions
5000 + s). Tuning: seed indices 4500..4509 (environment 5500.., actions
9500..), disjoint from evaluation, MAPPO (1000.., 2000..) and Q-learning
(3000..). Pursuit tie-break jitter: 9800 + seed index.
"""

import numpy as np

from gipa_core import CFG, Env, RNG_ACT_BASE, gipa, prior_heur, safe_mean, samp, own_redirect

RNG_LA_JITTER = 9800
TUNE_SEED_INDEX = 4500
N_TUNE_SEEDS = 10
EPS_PENALTY = 0.1
PURSUIT_STEP_FLOOR = 0.05
OPTIMISTIC = 2.0
ALGOS = ['lri', 'lrep', 'pursuit']
RESPONSES = ['binary', 'snorm']
RATES = [0.01, 0.05, 0.10]


def make_env(N, seed, R, model='affine'):
    """'affine' -> gipa_core.Env; 'ps' -> calibration.PSEnv (Eq. 5)."""
    if model == 'affine':
        return Env(N, seed, R)
    from calibration import PSEnv, make_calibration
    return PSEnv(N, seed, R, make_calibration())


def feasible_set(env, N, mask='deadline'):
    """Fixed per-vehicle feasible set, shape (N, M + 1), bool."""
    # Intrinsic cost per action from the environment's own scores, so RSUs
    # with different speeds (asymmetric.AsymEnv, E9) are handled; for Env and
    # PSEnv this is [loc, e0, e0, e0] exactly as before.
    cost = -env.q(N)
    if mask == 'prior':
        cost = cost + prior_heur(N)[None, :]
    elif mask != 'deadline':
        raise ValueError(mask)
    F = cost <= env.dl[:N, None]
    F[~F.any(1), 0] = True
    return F


def response(lat, env, N, kind):
    """Environment response beta in [0, 1] per vehicle (see module docstring)."""
    met = lat <= env.dl[:N]
    if kind == 'binary':
        return (met & (lat <= env.loc[:N] + 1e-12)).astype(float)
    if kind == 'snorm':
        return np.where(met, np.clip(1.0 - lat / env.dl[:N], 0.0, 1.0), 0.0)
    raise ValueError(kind)


class Automata:
    """A population of independent learning automata, one per vehicle.

    State is held for up to ``N_max`` vehicles; :meth:`activate` sets the
    active count and initialises any vehicle that has just arrived.
    """

    def __init__(self, env, N_max, algo, resp, rate, seed, mask='deadline', eps=EPS_PENALTY):
        if algo not in ALGOS:
            raise ValueError(algo)
        self.env, self.algo, self.resp, self.rate, self.eps = env, algo, resp, rate, eps
        self.F = feasible_set(env, N_max, mask)
        n = self.F.shape[1]
        self.P = np.zeros((N_max, n))
        self.D = np.zeros((N_max, n))
        self.cnt = np.zeros((N_max, n))
        self.jit = np.random.default_rng(RNG_LA_JITTER + seed).random((N_max, n)) * 1e-6
        self.active = 0

    def _init_rows(self, lo, hi):
        F = self.F[lo:hi]
        self.P[lo:hi] = F / F.sum(1, keepdims=True)
        self.D[lo:hi] = 0.0
        self.cnt[lo:hi] = 0.0

    def activate(self, N):
        """Make vehicles 0..N-1 active; vehicles N..active-1 leave (state dropped)."""
        if N > self.active:
            self._init_rows(self.active, N)
        self.active = N

    def act(self, rg, N):
        """Sample one action per active vehicle (one uniform each)."""
        return samp(self.P[:N], rg)

    def update(self, a, lat, N):
        """Apply the update scheme to vehicles 0..N-1 after realised latencies."""
        beta = response(lat, self.env, N, self.resp)
        P, F, rate = self.P[:N], self.F[:N], self.rate
        idx = np.arange(N)
        E = np.zeros_like(P)
        E[idx, a] = 1.0
        if self.algo in ('lri', 'lrep'):
            P += rate * beta[:, None] * (E - P)
            if self.algo == 'lrep':
                others = F & (E == 0)
                r1 = others.sum(1, keepdims=True)
                U = np.where(r1 > 0, others / np.maximum(r1, 1), P)
                P += self.eps * rate * (1.0 - beta)[:, None] * (U - P)
        else:
            self.cnt[idx, a] += 1.0
            step = np.maximum(1.0 / self.cnt[idx, a], PURSUIT_STEP_FLOOR)
            self.D[idx, a] += step * (beta - self.D[idx, a])
            est = np.where(self.cnt[:N] > 0, self.D[:N], OPTIMISTIC) + self.jit[:N]
            est = np.where(F, est, -np.inf)
            Eb = np.zeros_like(P)
            Eb[idx, est.argmax(1)] = 1.0
            P *= (1.0 - rate)
            P += rate * Eb
        P *= F
        P /= P.sum(1, keepdims=True)


def _new_out():
    return dict(mS=[], mB=[], lat=[], fair=[], off=[], offS=[], offB=[], en=[], p95=[], maxl=[])


def _record(out, env, a, cost, c, N):
    """Append the gipa_core.run metrics for one slot."""
    S = env.isS[:N]
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
    out['en'].append(env.en(a, N).mean())
    out['maxl'].append(rl.max())


def run_la(algo, resp, rate, N, seed, R, model='affine', warm=CFG['WARM'], mask='deadline',
           eps=EPS_PENALTY, curve_every=0):
    """One LA episode: ``warm`` learning slots, then 45 evaluated slots.

    Returns the metric dict of ``gipa_core.run``. With ``curve_every`` > 0
    the dict also has 'curve': mean latency of each block of that many slots
    over the whole episode (learning curve).
    """
    env = make_env(N, seed, R, model)
    rg = np.random.default_rng(RNG_ACT_BASE + seed)
    la = Automata(env, N, algo, resp, rate, seed, mask, eps)
    la.activate(N)
    n_eval = CFG['SLOTS'] - CFG['WARM']
    out = _new_out()
    lat_trace = []
    for t in range(warm + n_eval):
        a = la.act(rg, N)
        cost, c = env.lat(a, N)
        la.update(a, cost, N)
        if curve_every:
            lat_trace.append(cost.mean())
        if t >= warm:
            _record(out, env, a, cost, c, N)
    res = {k: safe_mean(v) for k, v in out.items()}
    if curve_every:
        lt = np.array(lat_trace)
        nb = len(lt) // curve_every
        res['curve'] = lt[:nb * curve_every].reshape(nb, curve_every).mean(1).tolist()
    return res


def agg_la(algo, resp, rate, N, R, seeds=None, **kw):
    """Run :func:`run_la` over seed indices (default 0..29); (means, per-seed list)."""
    seeds = range(CFG['NSEED']) if seeds is None else seeds
    rs = [run_la(algo, resp, rate, N, s, R, **kw) for s in seeds]
    keys = [k for k in rs[0] if k != 'curve']
    m = {k: safe_mean([r[k] for r in rs]) for k in keys}
    if 'curve' in rs[0]:
        m['curve'] = np.mean([r['curve'] for r in rs], 0).tolist()
    return m, rs


# ----------------------------------------------------------------------
# Fleet-size schedule (non-stationary test)
# ----------------------------------------------------------------------
def run_schedule(pol, schedule, seed, R, model='affine', la_cfg=None, plan_fn=None):
    """Run one policy through a fleet-size schedule and return per-slot traces.

    Parameters
    ----------
    pol : str
        'la' (needs ``la_cfg`` = dict(algo, resp, rate[, mask])),
        'gipa_fresh' (prior refreshed to the current N every slot),
        'gipa_stale' (prior for the initial N, never refreshed),
        'plan_fresh' / 'plan_stale' (needs ``plan_fn(N)`` returning a plan;
        re-planned on every change, or planned once for the initial N),
        or 'local'.
    schedule : list of int
        Active fleet size in each slot.

    Returns
    -------
    dict of lists, one value per slot: 'lat' (mean latency), 'miss' (miss
    rate over all active vehicles), 'mS', 'mB', 'off', 'maxl'.
    """
    from planner import planned_pmf
    N_max = max(schedule)
    env = make_env(N_max, seed, R, model)
    rg = np.random.default_rng(RNG_ACT_BASE + seed)
    if pol == 'la':
        la = Automata(env, N_max, la_cfg['algo'], la_cfg['resp'], la_cfg['rate'], seed,
                      la_cfg.get('mask', 'deadline'))
    N0 = schedule[0]
    plans = {}
    tr = dict(lat=[], miss=[], mS=[], mB=[], off=[], maxl=[])
    for N in schedule:
        if pol == 'la':
            la.activate(N)
            a = la.act(rg, N)
        elif pol in ('gipa_fresh', 'gipa_stale'):
            cb = prior_heur(N if pol == 'gipa_fresh' else N0)
            a = gipa(env, env.q(N), cb, rg, N)
        elif pol in ('plan_fresh', 'plan_stale'):
            Np = N if pol == 'plan_fresh' else N0
            if Np not in plans:
                plans[Np] = plan_fn(Np)
            P_ = planned_pmf(env, N, plans[Np])
            a = own_redirect(env, samp(P_, rg), P_, N, rg)
        elif pol == 'local':
            a = np.zeros(N, dtype=int)
        else:
            raise ValueError(pol)
        cost, c = env.lat(a, N)
        if pol == 'la':
            la.update(a, cost, N)
        S = env.isS[:N]
        miss = cost > env.dl[:N]
        tr['lat'].append(float(cost.mean()))
        tr['miss'].append(float(miss.mean()))
        tr['mS'].append(float(miss[S].mean()) if S.any() else 0.0)
        tr['mB'].append(float(miss[~S].mean()) if (~S).any() else 0.0)
        tr['off'].append(float((a > 0).mean()))
        tr['maxl'].append(float(c[1:].max()))
    return tr


# ----------------------------------------------------------------------
# Self-check
# ----------------------------------------------------------------------
def check_updates(tol=1e-12):
    """Check the update rules against hand-computed values on a toy case.

    One vehicle with four feasible actions, uniform start, rate 0.1:
    L_R-I after a reward on action 2 gives p = (0.225, 0.225, 0.325, 0.225);
    L_R-eP after a penalty on action 2 (eps 0.1) gives
    p_2 = 0.25 * 0.99 = 0.2475 and the others 0.25 + 0.01 * (1/3 - 0.25).
    Returns True if both match.
    """
    from gipa_core import REG
    env = Env(1, 0, REG['A'])
    env.dl = np.array([1e9])
    env.loc = np.array([5.0])
    ok = True
    la = Automata(env, 1, 'lri', 'binary', 0.1, 0)
    la.activate(1)
    la.update(np.array([2]), np.array([1.0]), 1)
    ok &= np.allclose(la.P[0], [0.225, 0.225, 0.325, 0.225], atol=tol)
    la = Automata(env, 1, 'lrep', 'binary', 0.1, 0)
    la.activate(1)
    la.update(np.array([2]), np.array([10.0]), 1)      # 10 > local 5 -> penalty
    o = 0.25 + 0.01 * (1 / 3 - 0.25)
    ok &= np.allclose(la.P[0], [o, o, 0.2475, o], atol=tol)
    return bool(ok)
