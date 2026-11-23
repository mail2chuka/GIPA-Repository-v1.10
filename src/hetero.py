"""
hetero.py
=========
Heterogeneous tasks, uneven links and partial coverage for workplan
experiment E12: can GIPA, operating on top of a planned allocation, beat the
plain planner when tasks differ? This tests "Framing A" of decision D5
(GIPA as the per-task, anti-herding executor of a planner-led method)
against "Framing B" (the planner alone, sampled per class).

STATUS: written for the first revision of manuscript ITS-2026-08-0579. Not
part of the submitted v1.0.0 results. Nothing in the existing modules is
changed.

Environment (physical calibration of E2, Regime A deadlines)
------------------------------------------------------------
Vehicle i keeps its class for the whole episode (drawn as in
``gipa_core.Env``, seed 1000 + s). Each slot it has a NEW task with
    compute c_i = c_k * h_c,  data size d_i = d_k * h_d,
where c_k = 10 / 22 Mcycles and d_k = 0.270 / 0.734 Mbit are the E2 class
means and h_c, h_d are independent mean-one lognormal factors with
log-standard deviation ``sigma_h`` (redrawn every slot). Each vehicle-RSU
pair has a link-rate factor g_ij (mean-one lognormal, ``sigma_r``, fixed for
the episode), and with ``coverage = 2`` each vehicle reaches only two of the
three RSUs (a random pair, fixed for the episode).
    local latency      L_i  = c_i / f,  f = 1 GHz
    transfer to RSU j  T_ij = d_i / (100 Mbit/s * g_ij)
    RSU compute        batch processor sharing at F = 33.45 GHz: task i
                       finishes after sum_l min(w_l, w_i), w = c / F, over the
                       tasks at that RSU (E2's model with per-task work)
    latency            T_ij + compute (offload) or L_i (local)
With sigma_h = sigma_r = 0 and full coverage this is exactly E2's
``calibration.PSEnv`` (checked in :func:`check_homogeneous`).
Feature draws use generator 12000 + s, link and coverage 12500 + s, and
prediction noise 13000 + s; actions use the archived 5000 + s, so all policies
see the same tasks and the same uniforms.

Predictions
-----------
Policies see predicted features c_hat = c * exp(e), d_hat = d * exp(e') with
e, e' ~ N(0, sigma_p^2) (``sigma_p = 0``: oracle), and the true link
factors and coverage (known from association). Realised latency uses true
features.

Policies
--------
'local'       all-local
'gipa'        GIPA as submitted: per-task predicted scores, E2's class-aware
              uniform-split prior, prior-inclusive mask with local fallback
'plan'        Framing B: E2 planner (``calibration.plan_allocation_ps``,
              z = 2) on class means; each vehicle samples its class PMF, with
              the mass of unreachable RSUs moved to its reachable RSUs
'plan_rank'   Framing B with per-task information, deterministic: the class
              offload fraction comes from the plan; a task offloads if its
              predicted benefit L - min_j (T_j + w) is above the class
              threshold that makes that fraction offload (computed in Stage 0
              by Monte Carlo from the known feature distributions), or if
              local would miss its deadline; it then picks one of its
              reachable RSUs at random
'gipa_plan'   Framing A: the GIPA operator anchored to the plan:
                1. mask actions whose predicted latency (uncongested cost +
                   planned congestion) times 1.1 exceeds the deadline
                   (10% guard band, from E1); local fallback;
                2. start from the plan's class PMF (restricted to the mask);
                3. K = 4 rounds of GIPA amplification on the task's score
                   ADVANTAGE over the class-average task, Delta = q - q_bar
                   (so an average task stays exactly on the plan);
                4. diffusion toward the plan, p <- (1 - lambda) p + lambda pi;
                5. sample.
              lambda = 1 is the plan restricted to each task's feasible set
              ('plan_mask'); it equals 'plan' draw for draw when the mask
              removes nothing (homogeneous tasks, full coverage).
'plan_mask'   Framing B with per-task feasibility only: the plan's class PMF
              restricted to the task's guarded deadline mask (gipa_plan at
              lambda = 1); isolates what amplification adds
              beta (amplification gain, default 0.4 as in Table II) and
              lambda are tuned per scenario on separate seeds.
'gipa_plan_abs'  the same but amplifying the absolute congestion-adjusted
              score (q - C) as GIPA does, instead of the advantage
'la'          per-vehicle learning automata (E6 tuned L_R-I for the Eq. 5
              model), 500 warm-up slots; binary response against the task's
              own local latency
"""

import numpy as np

from gipa_core import CFG, REG, Env, RNG_ACT_BASE, safe_mean, samp, sig, LOCAL_CARRY, carry_apply, carry_backlog

F_GHZ = 33.454545454545446
C_MEAN = (10.0, 22.0)          # Mcycles (safety, background)
D_MEAN = (0.27010869565217394, 0.7342391304347826)   # Mbit, gives T = 2.701 / 7.342 ms at 100 Mbit/s
RATE = 0.1                     # Mbit per ms (100 Mbit/s)
F_LOCAL = 1.0                  # GHz
GUARD = 0.10
RNG_FEAT, RNG_LINK, RNG_PRED, RNG_STAGE0 = 12000, 12500, 13000, 13500


def lognorm1(rg, sigma, size):
    """Mean-one lognormal factors."""
    if sigma == 0:
        return np.ones(size)
    return np.exp(sigma * rg.standard_normal(size) - sigma ** 2 / 2)


class HetEnv:
    """Heterogeneous environment for one seed; see the module docstring."""

    def __init__(self, N, seed, R, sigma_h=0.0, sigma_r=0.0, coverage=3, sigma_p=0.0, slots=None):
        base = Env(N, seed, R)
        self.N, self.R = N, R
        self.M = CFG['M']
        self.n = self.M + 1
        self.isS = base.isS[:N]
        self.dl = base.dl[:N]
        k = np.where(self.isS, 0, 1)
        self.k = k
        T = slots or CFG['SLOTS']
        rf = np.random.default_rng(RNG_FEAT + seed)
        self.c = np.array(C_MEAN)[k][None, :] * lognorm1(rf, sigma_h, (T, N))
        self.d = np.array(D_MEAN)[k][None, :] * lognorm1(rf, sigma_h, (T, N))
        rl = np.random.default_rng(RNG_LINK + seed)
        self.g = lognorm1(rl, sigma_r, (N, self.M))
        self.reach = np.ones((N, self.M), bool)
        if coverage < self.M:
            for i in range(N):
                drop = rl.choice(self.M, self.M - coverage, replace=False)
                self.reach[i, drop] = False
        rp = np.random.default_rng(RNG_PRED + seed)
        self.c_hat = self.c * lognorm1(rp, sigma_p, (T, N)) if sigma_p else self.c.copy()
        self.d_hat = self.d * lognorm1(rp, sigma_p, (T, N)) if sigma_p else self.d.copy()

    def cost(self, t, predicted=True):
        """Uncongested cost per action, (N, M + 1): local latency, then T_ij + w_i (inf if unreachable)."""
        c = self.c_hat[t] if predicted else self.c[t]
        d = self.d_hat[t] if predicted else self.d[t]
        out = np.empty((self.N, self.n))
        out[:, 0] = c / F_LOCAL
        out[:, 1:] = d[:, None] / (RATE * self.g) + (c / F_GHZ)[:, None]
        out[:, 1:][~self.reach] = np.inf
        if predicted and LOCAL_CARRY == 'own':
            out[:, 0] += carry_backlog(self, self.N)       # the vehicle knows its own backlog
        return out

    def lat(self, a, t):
        """Realised latency for joint action a in slot t; plus load vector."""
        c, d = self.c[t], self.d[t]
        w = c / F_GHZ
        lat = c / F_LOCAL
        cnt = np.bincount(a, minlength=self.n)
        out = lat.copy()
        for j in range(1, self.n):
            idx = np.flatnonzero(a == j)
            if not len(idx):
                continue
            wj = w[idx]
            order = np.sort(wj)
            cs = np.concatenate([[0.0], np.cumsum(order)])
            r = np.searchsorted(order, wj, side='right')      # number of works <= w_i
            comp = cs[r] + wj * (len(wj) - r)                 # sum_l min(w_l, w_i)
            out[idx] = d[idx] / (RATE * self.g[idx, j - 1]) + comp
        return carry_apply(self, a, out, lat, self.N), cnt


# ----------------------------------------------------------------------
# Stage 0 quantities
# ----------------------------------------------------------------------
def class_plan(N, R):
    """E2 planner on class means (z = 2); returns plan dict with PMFs per class."""
    from calibration import make_calibration, plan_allocation_ps
    cal = make_calibration()
    p = plan_allocation_ps(N, R, cal, z=2.0)
    M = CFG['M']
    pi = np.array([[1 - M * p['x_S']] + [p['x_S']] * M, [1 - M * p['x_B']] + [p['x_B']] * M])
    aS, aB = cal['a_S'], cal['a_B']
    s, b = p['s'], p['b']
    C = np.zeros((2, M + 1))
    C[0, 1:] = aS * max(s + b - 1, 0)
    C[1, 1:] = aS * s + aB * max(b - 1, 0)
    qbar = -np.array([[C_MEAN[k] / F_LOCAL] + [D_MEAN[k] / RATE + C_MEAN[k] / F_GHZ] * M for k in (0, 1)])
    return dict(plan=p, pi=pi, C=C, qbar=qbar)


def rank_thresholds(st, sigma_h, sigma_r, coverage, R, n=20000, tie_tol=1e-9):
    """Class benefit thresholds for 'plan_rank' from a synthetic population (Stage 0).

    Returns (tau, q_tie) per class: tasks with benefit above tau + tie_tol
    offload; tasks within tie_tol of tau offload with probability q_tie, so
    the planned offload fraction is met even when many tasks share the same
    benefit (e.g. homogeneous tasks, where 'plan_rank' then equals 'plan').
    """
    rg = np.random.default_rng(RNG_STAGE0)
    M = CFG['M']
    tau = []
    for k in (0, 1):
        c = C_MEAN[k] * lognorm1(rg, sigma_h, n)
        d = D_MEAN[k] * lognorm1(rg, sigma_h, n)
        g = lognorm1(rg, sigma_r, (n, M))
        e = d[:, None] / (RATE * g) + (c / F_GHZ)[:, None]
        if coverage < M:
            for i in range(n):
                e[i, rg.choice(M, M - coverage, replace=False)] = np.inf
        ben = c / F_LOCAL - e.min(1)
        frac = 1 - st['pi'][k, 0]
        if frac <= 0:
            tau.append((np.inf, 0.0))
            continue
        t_k = float(np.quantile(ben, 1 - frac))
        above = np.mean(ben > t_k + tie_tol)
        tie = np.mean(np.abs(ben - t_k) <= tie_tol)
        tau.append((t_k, float(np.clip((frac - above) / tie, 0, 1)) if tie > 0 else 0.0))
    return np.array(tau)


# ----------------------------------------------------------------------
# Policies
# ----------------------------------------------------------------------
def restrict(P, ok):
    """Restrict row PMFs to allowed actions and renormalise (uniform over allowed if no mass)."""
    Q = np.where(ok, P, 0.0)
    s = Q.sum(1, keepdims=True)
    U = ok / ok.sum(1, keepdims=True)
    return np.where(s > 1e-15, Q / np.maximum(s, 1e-300), U)


def plan_pmf(env, st):
    """Class PMF with unreachable-RSU mass moved to reachable RSUs (offload probability kept)."""
    pi = st['pi'][env.k]
    ok = np.concatenate([np.ones((env.N, 1), bool), env.reach], 1)
    off = pi[:, 1:].sum(1)
    rs = np.where(env.reach, pi[:, 1:], 0.0)
    rs = rs / np.maximum(rs.sum(1, keepdims=True), 1e-300) * off[:, None]
    P = np.concatenate([pi[:, :1], rs], 1)
    return np.where(ok, P, 0.0)


def gipa_plan_probs(env, t, st, lam, relative=True, beta=None):
    """Framing A operator for slot t; returns (N, M + 1) PMFs."""
    q = -env.cost(t, predicted=True)
    C = st['C'][env.k]
    D = env.dl[:, None]
    pred = -q + C
    m = np.isfinite(pred) & (pred * (1 + GUARD) <= D)
    m[~m.any(1), 0] = True
    pi = restrict(plan_pmf(env, st), m)
    p = pi.copy()
    if relative:
        qt = np.where(np.isfinite(q), q - st['qbar'][env.k], -np.inf)
    else:
        qt = np.where(np.isfinite(q), q - C, -np.inf)
    rm = np.where(m, qt, -np.inf).max(1, keepdims=True)
    for r in range(1, CFG['K'] + 1):
        th = CFG['TH0'] * CFG['GA'] ** r
        arg = np.where(m, (qt - rm + th) / CFG['NU'], 0.0)
        g = np.where(m, np.exp((CFG['BETA'] if beta is None else beta) * sig(arg)), 0.0)
        p = p * g
        p = p / p.sum(1, keepdims=True)
        p = (1 - lam) * p + lam * pi
    return p


def gipa_submitted_probs(env, t, N):
    """GIPA as submitted on per-task predicted scores (E2 class prior, local fallback)."""
    from calibration import make_calibration, prior_matrix
    cal = make_calibration()
    q = -env.cost(t, predicted=True)
    C = prior_matrix(env, N, cal, 'class')
    isS = env.isS
    lam = np.where(isS, CFG['LAM_S'], CFG['LAM_B'])[:, None]
    D = env.dl[:, None]
    pred = -q + C
    m = np.isfinite(pred) & (pred <= D)
    m[~m.any(1), 0] = True
    qt = np.where(np.isfinite(q), q - CFG['ETA'] * C, -np.inf)
    Fs = m.sum(1, keepdims=True)
    z = np.where(m, qt / CFG['TAU'], -np.inf)
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    p = p / p.sum(1, keepdims=True)
    rm = np.where(m, qt, -np.inf).max(1, keepdims=True)
    for r in range(1, CFG['K'] + 1):
        th = CFG['TH0'] * CFG['GA'] ** r
        arg = np.where(m, (qt - rm + th) / CFG['NU'], 0.0)
        g = np.where(m, np.exp(CFG['BETA'] * sig(arg)), 0.0)
        p = p * g
        p = p / p.sum(1, keepdims=True)
        p = (1 - lam) * p + lam * m / Fs
    return p


def plan_rank_actions(env, t, st, tau, rg):
    """Deterministic per-task ranking under the plan's class offload fraction.

    Consumes exactly one uniform per vehicle (for the RSU choice) so the
    action stream stays aligned with the other policies.
    """
    e = env.cost(t, predicted=True)
    u = rg.random(env.N)
    ben = e[:, 0] - e[:, 1:].min(1)
    local_bad = e[:, 0] > env.dl
    t_k, q_tie = tau[env.k, 0], tau[env.k, 1]
    above = ben > t_k + 1e-9
    tie = np.abs(ben - t_k) <= 1e-9
    tie_go = tie & (u < q_tie)
    go = above | tie_go | local_bad
    nreach = env.reach.sum(1)
    # One uniform per vehicle: reuse it for the RSU pick (rescaled within the tie draw).
    v = np.where(tie_go & ~above & ~local_bad, u / np.maximum(q_tie, 1e-12), u)
    pick = np.minimum((v * nreach).astype(int), nreach - 1)
    rsu = np.array([np.flatnonzero(env.reach[i])[pick[i]] + 1 for i in range(env.N)])
    return np.where(go, rsu, 0)


# ----------------------------------------------------------------------
# Episode runner
# ----------------------------------------------------------------------
def run_het(pol, N, seed, R, scen, lam=0.5, beta=None, la_cfg=None, warm=CFG['WARM']):
    """One episode; returns gipa_core.run-style metrics plus offload targeting."""
    T = warm + CFG['SLOTS'] - CFG['WARM']
    env = HetEnv(N, seed, R, scen['sigma_h'], scen['sigma_r'], scen['coverage'], scen.get('sigma_p', 0.0), slots=T)
    rg = np.random.default_rng(RNG_ACT_BASE + seed)
    st = class_plan(N, R) if pol != 'local' else None
    if pol == 'plan_rank':
        tau = rank_thresholds(st, scen['sigma_h'], scen['sigma_r'], scen['coverage'], R)
    if pol == 'la':
        from learning_automata import Automata

        class Shim:
            pass
        sh = Shim()
        sh.isS, sh.dl, sh.N = env.isS, env.dl, N
        sh.loc = np.zeros(N)
        reach_cost = np.concatenate([np.zeros((N, 1)), np.where(env.reach, 0.0, np.inf)], 1)
        sh.q = lambda n: -reach_cost[:n]
        la = Automata(sh, N, la_cfg['algo'], la_cfg['resp'], la_cfg['rate'], seed)
        la.activate(N)
    out = dict(mS=[], mB=[], lat=[], p95=[], off=[], offS=[], offB=[], maxl=[], miss=[], local_infeasible_rescued=[],
               ben_off=[], cv=[])
    for t in range(T):
        if pol == 'local':
            a = np.zeros(N, int)
        elif pol == 'plan':
            P = plan_pmf(env, st)
            a = samp(P, rg)
            if LOCAL_CARRY == 'own':
                # Local execution predicted to miss (own backlog): redraw among the plan's reachable RSUs.
                u2 = rg.random(N)
                bad = (a == 0) & (env.cost(t, predicted=True)[:, 0] > env.dl)
                if bad.any():
                    rs = P[:, 1:]
                    tot = rs.sum(1, keepdims=True)
                    unif = env.reach / env.reach.sum(1, keepdims=True)
                    rs = np.where(tot > 0, rs / np.maximum(tot, 1e-300), unif)
                    j = np.minimum((u2[:, None] > np.cumsum(rs, 1)).sum(1), rs.shape[1] - 1) + 1
                    a = np.where(bad, j, a)
        elif pol == 'gipa_plan':
            a = samp(gipa_plan_probs(env, t, st, lam, True, beta), rg)
        elif pol == 'gipa_plan_abs':
            a = samp(gipa_plan_probs(env, t, st, lam, False, beta), rg)
        elif pol == 'plan_mask':
            a = samp(gipa_plan_probs(env, t, st, 1.0, True), rg)
        elif pol == 'gipa':
            a = samp(gipa_submitted_probs(env, t, N), rg)
        elif pol == 'plan_rank':
            a = plan_rank_actions(env, t, st, tau, rg)
        elif pol == 'la':
            a = la.act(rg, N)
        else:
            raise ValueError(pol)
        cost, cnt = env.lat(a, t)
        if pol == 'la':
            sh.loc = env.c[t] / F_LOCAL
            la.update(a, cost, N)
        if t >= warm:
            S = env.isS
            miss = cost > env.dl
            out['mS'].append(miss[S].mean() if S.any() else 0.0)
            out['mB'].append(miss[~S].mean() if (~S).any() else 0.0)
            out['miss'].append(miss.mean())
            out['lat'].append(cost.mean())
            out['p95'].append(np.percentile(cost, 95))
            out['off'].append((a > 0).mean())
            out['offS'].append((a[S] > 0).mean() if S.any() else 0.0)
            out['offB'].append((a[~S] > 0).mean() if (~S).any() else 0.0)
            rl = cnt[1:]
            out['maxl'].append(rl.max())
            out['cv'].append(np.nan if rl.sum() == 0 else rl.std() / rl.mean())
            loc_bad = (env.c[t] / F_LOCAL) > env.dl
            out['local_infeasible_rescued'].append(np.nan if not loc_bad.any() else float((~miss[loc_bad]).mean()))
            e_true = env.cost(t, predicted=False)
            ben = e_true[:, 0] - e_true[:, 1:].min(1)
            out['ben_off'].append(np.nan if not (a > 0).any() else float(ben[a > 0].mean() - ben.mean()))
    return {k: safe_mean(v) for k, v in out.items()}


def check_homogeneous(N=160, seeds=range(3)):
    """sigma = 0, full coverage: 'plan' and 'gipa' reproduce E2 per-seed latencies exactly."""
    import json
    from pathlib import Path
    ref = json.load(open(Path(__file__).resolve().parents[1] / 'results' / 'res_calibration.json'))['per_seed']['A'][str(N)]
    scen = dict(sigma_h=0.0, sigma_r=0.0, coverage=3)
    worst = 0.0
    for s in seeds:
        worst = max(worst, abs(run_het('plan', N, s, REG['A'], scen)['lat'] - ref['plan_z2|class'][s]['lat']))
        worst = max(worst, abs(run_het('gipa', N, s, REG['A'], scen)['lat'] - ref['gipa|class'][s]['lat']))
    return worst
