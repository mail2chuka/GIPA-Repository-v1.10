"""
calibration.py
==============
Physical calibration and the computation-based latency model of manuscript
Eq. (5) with class-dependent compute (workplan experiment E2, answering
Reviewer 2 comment 2: "justify the affine model and validate GIPA under the
model defined in Equation (5)").

STATUS: written for the first revision of manuscript ITS-2026-08-0579. Not
part of the submitted v1.0.0 results.

Physical calibration (Regime A anchors)
---------------------------------------
Local latency is compute over vehicle speed (Eq. 3): L_k = c_k / f_i. With an
assumed vehicle CPU of f_i = 1 GHz, the archived local latencies (10 ms and
22 ms) give

    c_S = 10 Mcycles (safety),  c_B = 22 Mcycles (background),

a 2.2x ratio. The RSU capacity F_j is chosen so that the class-mix average
compute time per task equals the archived congestion step:

    (pi_S c_S + (1 - pi_S) c_B) / F_j = 0.55 ms  =>  F_j = 18.4 / 0.55 = 33.45 GHz.

The per-task RSU compute times are then a_S = c_S / F_j = 0.299 ms and
a_B = c_B / F_j = 0.658 ms, and the transfer (uplink + downlink) times are
what remains of the archived uncongested RSU latencies:
T_S = E_S - a_S = 2.701 ms, T_B = E_B - a_B = 7.342 ms. At a 100 Mbit/s
uplink with a negligible result size these correspond to about 0.27 Mbit
(safety) and 0.73 Mbit (background) of input data. f_i = 1 GHz and
100 Mbit/s are assumptions; everything else is derived. Regime B keeps the
archived local latencies (a slower vehicle, f_i ~ 0.71 GHz); its background
local latency would be 30.8 ms rather than 30 ms under this calibration,
a 3% discrepancy that is documented but not changed.

Latency model (Eq. 5 with exact processor sharing)
--------------------------------------------------
All tasks offloaded to an RSU in a slot start computing together and share
the processor equally (batch processor sharing). Task i then finishes after

    T_k(i) + sum_l min(c_l, c_i) / F_j,

where the sum runs over every task at that RSU, including i. For a safety
task this is T_S + a_S * n (n = all tasks at the RSU); for a background task
it is T_B + a_S * n_S + a_B * n_B. Two consequences:

* With class-independent compute (a_S = a_B = a), this reduces exactly to
  the affine model E_k + a (n - 1) used in the submitted experiments, which
  is therefore the special case c_S = c_B of Eq. (5). :func:`check_special_case`
  verifies this numerically.
* With class-dependent compute, safety tasks are delayed far less by
  co-located tasks (0.30 ms each) than background tasks are (0.30 ms per
  safety task, 0.66 ms per background task).

Upload-time differences are ignored for the start of computation; the
asynchronous model of experiment E5 removes that simplification.

Priors
------
'agnostic'  the submitted heuristic prior (Eq. 13): one value per RSU,
            a_mix * (N / (M + 1) - 1), identical for both classes; with the
            default calibration a_mix = 0.55 ms, so it equals
            gipa_core.prior_heur(N).
'class'     class-aware heuristic prior from the same calibration and the
            same uniform-split assumption, n_bar = N / (M + 1) tasks per RSU
            of which pi_S are safety:
                safety:     a_S * max(n_bar - 1, 0)
                background: a_S * pi_S * n_bar + a_B * max((1 - pi_S) * n_bar - 1, 0)

Policies that use the prior (masked softmax family, GIPA and its
ablations) are re-implemented here with a per-vehicle prior matrix
C (N x (M + 1)); with identical rows they reproduce gipa_core exactly. Policies
whose internal congestion model is the affine kappa (reactive best response,
congestion pricing) are unchanged: they keep a class-agnostic internal model
while the environment is class-dependent. Q-learning is not run here; it is
retrained under both latency models in experiment E4.

Seeds
-----
Environment 1000 + s and action sampling 5000 + s, as in gipa_core.run.
"""

import numpy as np

from gipa_core import (CFG, Env, safe_mean, sig, samp, RNG_ACT_BASE, RBR, own_redirect,
                       greedy, masked_softmax, price_rr, alllocal, prior_heur)

# ----------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------
F_I_GHZ = 1.0                 # assumed vehicle CPU (Regime A)
C_S_MCYC = 10.0               # safety compute demand, Mcycles
C_B_MCYC = 22.0               # background compute demand, Mcycles
UPLINK_MBPS = 100.0           # assumed uplink rate for the data-size check


def make_calibration(F_ghz=None, c_S=C_S_MCYC, c_B=C_B_MCYC, R=None):
    """Return a calibration dict for RSU capacity F_ghz (GHz).

    By default F_ghz is chosen so that the class-mix average compute time
    equals CFG['CONG'] (0.55 ms). Transfer times T_S, T_B are fixed by the
    default calibration and the archived uncongested latencies E_S = 3 ms,
    E_B = 8 ms; changing F_ghz changes only the compute part, so the
    uncongested RSU latency becomes e_k = T_k + c_k / F.

    Keys: F_ghz, c_S, c_B (Mcycles), a_S, a_B (ms per task), T_S, T_B (ms),
    e_S, e_B (uncongested RSU latency, ms), a_mix (class-mix average, ms),
    size_S_mbit, size_B_mbit (input size implied at UPLINK_MBPS).
    """
    pi = CFG['PI_S']
    c_mix = pi * c_S + (1 - pi) * c_B
    F_default = c_mix / CFG['CONG']               # Mcycles / ms = GHz
    T_S = 3.0 - c_S / F_default
    T_B = 8.0 - c_B / F_default
    F = F_default if F_ghz is None else F_ghz
    a_S, a_B = c_S / F, c_B / F
    return dict(F_ghz=F, c_S=c_S, c_B=c_B, a_S=a_S, a_B=a_B, T_S=T_S, T_B=T_B,
                e_S=T_S + a_S, e_B=T_B + a_B, a_mix=pi * a_S + (1 - pi) * a_B,
                size_S_mbit=T_S * UPLINK_MBPS / 1000, size_B_mbit=T_B * UPLINK_MBPS / 1000)


def class_independent_calibration():
    """Calibration with c_S = c_B (a_S = a_B = 0.55 ms): the affine special case."""
    a = CFG['CONG']
    return dict(F_ghz=None, c_S=None, c_B=None, a_S=a, a_B=a, T_S=3.0 - a, T_B=8.0 - a,
                e_S=3.0, e_B=8.0, a_mix=a, size_S_mbit=None, size_B_mbit=None)


# ----------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------
class PSEnv(Env):
    """``gipa_core.Env`` with Eq. (5) batch processor sharing at each RSU.

    The task classes, deadlines and local latencies are drawn exactly as in
    ``Env`` (same seed, same draws). The uncongested RSU latency per class
    is ``cal['e_S']`` / ``cal['e_B']`` (equal to the archived 3 and 8 ms for
    the default calibration), so intrinsic scores ``q`` are consistent with
    the RSU capacity.
    """

    def __init__(self, N, seed, R, cal):
        super().__init__(N, seed, R)
        self.cal = cal
        self.e0 = np.where(self.isS, cal['e_S'], cal['e_B'])
        self.T = np.where(self.isS, cal['T_S'], cal['T_B'])

    def lat(self, a, N):
        """Realised latency under batch processor sharing; plus the load vector."""
        M = CFG['M']
        S = self.isS[:N]
        c = np.bincount(a, minlength=M + 1)
        nS = np.bincount(a[S], minlength=M + 1)
        nB = c - nS
        a_S, a_B = self.cal['a_S'], self.cal['a_B']
        comp = np.where(S, a_S * c[a], a_S * nS[a] + a_B * nB[a])
        return np.where(a == 0, self.loc[:N], self.T[:N] + comp), c


def check_special_case(N=160, seeds=range(5), R=None, tol=1e-9):
    """Verify that Eq. (5) with class-independent compute equals the affine model.

    Runs the default GIPA and masked softmax in both environments and
    returns the largest absolute metric difference (should be ~1e-14).
    """
    from gipa_core import REG, run
    R = R or REG['A']
    cal = class_independent_calibration()
    worst = 0.0
    for pol in ['gipa', 'msoft', 'local', 'greedy']:
        for s in seeds:
            a = run(pol, N, s, R)
            b = run_ps(pol, N, s, R, cal, prior='agnostic')
            for k, v in a.items():
                w = b[k]
                if np.isnan(v) and np.isnan(w):
                    continue
                worst = max(worst, abs(v - w))
    return worst, worst < tol


# ----------------------------------------------------------------------
# Priors
# ----------------------------------------------------------------------
def prior_matrix(env, N, cal, kind='class', f=1.0):
    """Per-vehicle prior matrix C (N x (M + 1)), local column 0.

    kind 'agnostic': a_mix * max(n_bar - 1, 0) for every RSU and vehicle
    (equals gipa_core.prior_heur(N) for the default calibration).
    kind 'class': class-aware values from the module docstring.
    ``f`` scales the anticipated fleet (N_bar = f * N).
    """
    M = CFG['M']
    pi = CFG['PI_S']
    nbar = f * N / (M + 1)
    C = np.zeros((N, M + 1))
    S = env.isS[:N]
    if kind == 'agnostic':
        if f == 1.0 and abs(cal['a_mix'] - CFG['CONG']) < 1e-12:
            C[:, 1:] = prior_heur(N)[1:]            # bit-identical to the archive
        else:
            C[:, 1:] = cal['a_mix'] * max(nbar - 1, 0)
    elif kind == 'class':
        C[S, 1:] = cal['a_S'] * max(nbar - 1, 0)
        C[~S, 1:] = cal['a_S'] * pi * nbar + cal['a_B'] * max((1 - pi) * nbar - 1, 0)
    else:
        raise ValueError(kind)
    return C


# ----------------------------------------------------------------------
# Policies with a per-vehicle prior matrix
# ----------------------------------------------------------------------
def gipa_pv(env, q, C, rg, N, anneal=True, graded=True, lam_on=True,
            amp_on=True, mask_on=True, prior_on=True):
    """``gipa_core.gipa`` with a per-vehicle prior matrix C instead of a vector."""
    isS = env.isS[:N]
    lam = np.where(isS, CFG['LAM_S'], CFG['LAM_B'])[:, None]
    if not lam_on:
        lam = lam * 0
    D = env.dl[:N][:, None]
    pred = -q + C
    m = (pred <= D) if mask_on else np.ones_like(pred, bool)
    m[~m.any(1), 0] = True
    qt = q - (CFG['ETA'] * C if prior_on else 0)
    Fs = m.sum(1, keepdims=True)
    z = np.where(m, qt / CFG['TAU'], -np.inf)
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    p = p / p.sum(1, keepdims=True)
    rm = np.where(m, qt, -np.inf).max(1, keepdims=True)
    for t in range(1, CFG['K'] + 1):
        th = CFG['TH0'] * (CFG['GA'] ** t if anneal else 1.0)
        if amp_on:
            arg = (qt - rm + th) / CFG['NU']
            g = np.exp(CFG['BETA'] * (sig(arg) if graded else (arg > 0).astype(float)))
            g = np.where(m, g, 0.0)
            p = p * g
            p = p / p.sum(1, keepdims=True)
        p = (1 - lam) * p + lam * m / Fs
    return samp(p, rg)


def msoft_pv(env, q, C, rg, N, use_prior=False, tau=None):
    """``gipa_core.masked_softmax`` with a per-vehicle prior matrix."""
    tau = tau or CFG['TAU']
    D = env.dl[:N][:, None]
    m = (-q + C) <= D
    m[~m.any(1), 0] = True
    sc = q - (CFG['ETA'] * C if use_prior else 0)
    z = np.where(m, sc / tau, -np.inf)
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    p = p / p.sum(1, keepdims=True)
    return samp(p, rg)


# ----------------------------------------------------------------------
# Planner under Eq. (5)
# ----------------------------------------------------------------------
def plan_allocation_ps(Nbar, R, cal, z=0.0, grid_s=61, grid_b=241):
    """Eq. (5) planned allocation (see :func:`_plan_allocation_ps`); carry-aware when GIPA_LOCAL_CARRY is set."""
    from gipa_core import LOCAL_CARRY
    if LOCAL_CARRY:
        from carry import carry_plan
        return carry_plan(_plan_allocation_ps, Nbar, R, cal, z=z, grid_s=grid_s, grid_b=grid_b)
    return _plan_allocation_ps(Nbar, R, cal, z=z, grid_s=grid_s, grid_b=grid_b)


def _plan_allocation_ps(Nbar, R, cal, z=0.0, grid_s=61, grid_b=241):
    """Deadline-constrained system-optimal symmetric allocation under Eq. (5).

    Same search and return format as ``planner.plan_allocation``, but with
    class-dependent processor-sharing latencies:
        safety at an RSU:      e_S + a_S * (n - 1)
        background at an RSU:  e_B + a_S * s + a_B * (b - 1)
    where s, b are the expected safety and background tasks per RSU and
    n = s + b. The margin ``z`` scales both counts up by (n + z sd) / n.
    """
    M = CFG['M']
    a_S, a_B, e_S, e_B = cal['a_S'], cal['a_B'], cal['e_S'], cal['e_B']
    NS = CFG['PI_S'] * Nbar
    NB = Nbar - NS
    best = None
    for s in np.linspace(0, NS / M, grid_s):
        for b in np.linspace(0, NB / M, grid_b):
            n = s + b
            ps = s / NS if NS else 0
            pb = b / NB if NB else 0
            sd = np.sqrt(NS * ps * (1 - ps) + NB * pb * (1 - pb))
            k = (n + z * sd) / n if n > 0 else 1.0
            s_hi, b_hi, n_hi = s * k, b * k, n * k
            feasible = True
            if s > 0 and e_S + a_S * max(n_hi - 1, 0) > R['D_S']:
                feasible = False
            if b > 0 and e_B + a_S * s_hi + a_B * max(b_hi - 1, 0) > R['D_B']:
                feasible = False
            if R['L_S'] > R['D_S'] and NS - M * s > 1e-9:
                feasible = False
            if R['L_B'] > R['D_B'] and NB - M * b > 1e-9:
                feasible = False
            # Minimum offload shares (set only by carry.carry_plan in 'own' mode).
            if M * s < R.get('MIN_OFF_S', 0.0) * NS - 1e-9 or M * b < R.get('MIN_OFF_B', 0.0) * NB - 1e-9:
                feasible = False
            if not feasible:
                continue
            J = ((NS - M * s) * R['L_S'] + M * s * (e_S + a_S * max(n - 1, 0))
                 + (NB - M * b) * R['L_B'] + M * b * (e_B + a_S * s + a_B * max(b - 1, 0)))
            if best is None or J < best[0]:
                best = (J, s, b)
    if best is None:
        return None
    J, s, b = best
    return dict(x_S=s / NS if NS else 0.0, x_B=b / NB if NB else 0.0,
                s=float(s), b=float(b), J=float(J), Nbar=float(Nbar), z=float(z))


# ----------------------------------------------------------------------
# Episode runner
# ----------------------------------------------------------------------
PRIOR_POLICIES = ['msoft', 'msoft_prior', 'entreg', 'gipa', 'plan']
"""Policies whose decisions depend on the prior (or, for 'plan', on the
latency model used for planning)."""


def run_ps(pol, N, seed, R, cal, prior='class', plan=None, **kw):
    """One episode under Eq. (5) processor sharing.

    Parameters
    ----------
    pol : str
        'greedy', 'indep', 'rbr', 'price', 'msoft', 'msoft_prior', 'entreg',
        'local', 'gipa' (ablation switches via **kw) or 'plan'.
    cal : dict
        From :func:`make_calibration` or :func:`class_independent_calibration`.
    prior : {'class', 'agnostic'}
        Prior used by the prior-dependent policies.
    plan : dict, optional
        Planned allocation for 'plan' (from :func:`plan_allocation_ps` or
        ``planner.plan_allocation``).

    Returns the metric dict of ``gipa_core.run``.
    """
    env = PSEnv(N, seed, R, cal)
    rg = np.random.default_rng(RNG_ACT_BASE + seed)
    C = prior_matrix(env, N, cal, prior)
    cb0 = C[0].copy()                 # vector form for policies that take one
    rbr = RBR()
    P = None
    if pol == 'plan':
        from planner import planned_pmf
        P = planned_pmf(env, N, plan)
    out = dict(mS=[], mB=[], lat=[], fair=[], off=[], offS=[], offB=[],
               en=[], p95=[], maxl=[])
    for t in range(CFG['SLOTS']):
        q = env.q(N)
        if pol == 'gipa':
            a = gipa_pv(env, q, C, rg, N, **kw)
        elif pol == 'greedy':
            a = greedy(env, q, cb0, rg, N)
        elif pol == 'indep':
            a = masked_softmax(env, q, cb0 * 0, rg, N)
        elif pol == 'msoft':
            a = msoft_pv(env, q, C, rg, N)
        elif pol == 'msoft_prior':
            a = msoft_pv(env, q, C, rg, N, use_prior=True)
        elif pol == 'entreg':
            a = msoft_pv(env, q, C, rg, N, use_prior=True, tau=kw.get('tau', 1.2))
        elif pol == 'price':
            a = price_rr(env, q, cb0, rg, N)
        elif pol == 'rbr':
            a = rbr.act(env, q, cb0, rg, N)
        elif pol == 'local':
            a = alllocal(env, q, cb0, rg, N)
        elif pol == 'plan':
            a = own_redirect(env, samp(P, rg), P, N, rg)
        else:
            raise ValueError(f'unknown policy: {pol}')
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


def agg_ps(pol, N, R, cal, prior='class', plan=None, **kw):
    """Run :func:`run_ps` for seed indices 0..NSEED-1; returns (means, per-seed list)."""
    rs = [run_ps(pol, N, s, R, cal, prior, plan, **kw) for s in range(CFG['NSEED'])]
    return {k: safe_mean([r[k] for r in rs]) for k in rs[0]}, rs
