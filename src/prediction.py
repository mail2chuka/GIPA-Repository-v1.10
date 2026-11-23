"""
prediction.py
=============
Replace the oracle intrinsic scores with *predicted* latencies and measure
how prediction error affects offloading decisions, latency and deadline
misses (workplan experiment E1, answering Reviewer 2 comment 1).

STATUS: written for the first revision of manuscript ITS-2026-08-0579. Not
part of the submitted v1.0.0 results.

Why the policies are re-implemented here
----------------------------------------
In ``gipa_core`` every policy derives both its scores and its feasibility
mask from one matrix ``q``. To isolate error sources, the functions below
take two matrices: ``qs`` drives the scores and ``qm`` drives the deadline
mask. They are line-for-line copies of the ``gipa_core`` policies otherwise.
With ``qs = qm = q`` and no guard band, :func:`run_pred` reproduces
``gipa_core.run`` exactly (verified for all policies except Q-learning).

Error arms (one predicted matrix, used where the arm says)
----------------------------------------------------------
'none'   oracle: scores and mask use the true intrinsic latency
'score'  (a) scores use the prediction; the mask uses the truth
'mask'   (b) the mask uses the prediction; scores use the truth
'both'   combined: one predictor feeds both, as in a real deployment

Prior error, arm (c), is applied separately by passing a misspecified prior
(:func:`prior_misspecified`): wrong anticipated fleet size N_bar = f * N
and/or a wrong congestion slope xi_hat = xi_mult * 0.55 ms.

Truth models
------------
'static' the archived simulator: each task's intrinsic latency is a fixed
         constant (local L_k, RSU E_k).
'ar1'    each vehicle's intrinsic latency for each action drifts over time:
         l_t = l_0 * max(1 + eta_t, 0.1), with eta following a stationary
         AR(1) process (correlation ``rho`` = 0.9 per slot, standard deviation
         ``sigma_eta``). This models slowly varying channel and compute
         conditions, and gives a history-based predictor something to track.

Predictors
----------
'oracle'    l_hat = l_t (no error).
'synthetic' l_hat = l_t * (1 + eps), eps = max(b + sigma * Z, -0.9),
            Z ~ N(0, 1) i.i.d. per vehicle, action and slot. ``bias`` b < 0
            means optimistic (under-predicted) latency, the dangerous case.
'ewma'      each vehicle keeps an exponentially weighted average of the
            intrinsic latency it observed for each action, updated only for
            the action it took (smoothing ``alpha`` = 0.2), initialised at
            the nominal latency. Actions not taken recently go stale. It uses
            only the vehicle's own observations, so it is communication-free.
            What the vehicle observes is set by ``obs`` (E1b, Nov 2026):
              'breakdown'       the uncongested latency of the action it took
                                (the RSU returns its timing breakdown with the
                                result: transfer and isolated service time).
                                This is the E1 default.
              'total'           only the realised latency, congestion
                                included, used as is (naive: the prior's
                                congestion is then counted twice).
              'total_debiased'  the realised latency minus the congestion the
                                broadcast prior anticipates for that action
                                (heuristic prior c_bar, or the planned load for
                                the planned policies). Needs no breakdown and
                                no communication.

Planned policies that use per-task predictions (E1b)
---------------------------------------------------
'plan'       samples the plan; uses no per-task prediction (E1).
'plan_mask'  the plan restricted to the task's predicted-feasible actions
             (guarded mask (8) with the plan's anticipated congestion),
             renormalised; local fallback if nothing is feasible.
'gipa_plan'  GIPA with planned prior as in Algorithm 2: mask as above,
             start from the restricted plan (18), K rounds of amplification
             on the task's predicted advantage over the class average (16)
             and diffusion towards the restricted plan with lambda_k (22).
             With oracle predictions and identical tasks both reproduce
             'plan' draw for draw.

Guard band
----------
A predicted action is feasible only if (predicted latency + prior) *
(1 + guard) <= deadline. ``guard`` = 0 is the archived mask.

Realised latency and misses always come from the truth, never from the
prediction.

Seeds
-----
Environment 1000 + s and action sampling 5000 + s (paired with the archived
runs); all prediction-noise and AR(1) draws use a separate generator seeded
7000 + s. Draws are made every slot whatever the noise level, so the same
underlying random numbers are used at every sigma and bias (common random
numbers).

Metrics returned by :func:`run_pred`
------------------------------------
Standard keys as in ``gipa_core.run`` (mS, mB, lat, p95, fair, off, offS,
offB, en, maxl), plus:
    over_sel   share of RSU-slots whose selected load exceeds the effective
               capacity (see capacity.py)
    pred_err   mean absolute relative prediction error |l_hat - l_t| / l_t
               over all vehicles and actions (0 for the oracle)
"""

import numpy as np

from gipa_core import (CFG, Env, safe_mean, kappa, samp, sig, RNG_ACT_BASE, RBR, LOCAL_CARRY, carry_apply,
                       carry_backlog, own_redirect,
                       prior_heur)
from capacity import capacities, effective_capacity

RNG_NOISE_BASE = 7000
EPS_FLOOR = -0.9


# ----------------------------------------------------------------------
# Priors with arm (c) misspecification
# ----------------------------------------------------------------------
def plan_congestion(plan, z=0.0):
    """Congestion the plan anticipates for a task at each action (affine model).

    With ``z`` > 0 the load is the planned mean plus z standard deviations of
    the independent-sampling load (the margin of P1), so the mask treats an
    RSU as feasible only if it meets the deadline at the margin load.
    """
    c = np.zeros(CFG['M'] + 1)
    n = plan['s'] + plan['b']
    if z:
        NS = CFG['PI_S'] * plan['Nbar']
        NB = plan['Nbar'] - NS
        var = NS * plan['x_S'] * (1 - plan['x_S']) + NB * plan['x_B'] * (1 - plan['x_B'])
        n = n + z * np.sqrt(var)
    c[1:] = kappa(n)
    return c


def restrict(P, ok):
    """Restrict row PMFs to allowed actions and renormalise (uniform over allowed if no mass)."""
    Q = np.where(ok, P, 0.0)
    s = Q.sum(1, keepdims=True)
    U = ok / ok.sum(1, keepdims=True)
    return np.where(s > 1e-15, Q / np.maximum(s, 1e-300), U)


def plan_mask_probs(env, qm, Cp, P, N, guard=0.0):
    """The plan restricted to the predicted-feasible actions."""
    return restrict(P, _mask(env, qm, Cp, N, guard))


def gipa_plan_probs(env, qs, qm, Cp, P, qbar, N, guard=0.0, beta=None):
    """GIPA with planned prior (Algorithm 2) on predicted scores qs and mask qm."""
    lam = np.where(env.isS[:N], CFG['LAM_S'], CFG['LAM_B'])[:, None]
    m = _mask(env, qm, Cp, N, guard)
    u = restrict(P, m)
    qt = qs - qbar
    rm = np.where(m, qt, -np.inf).max(1, keepdims=True)
    p = u.copy()
    for t in range(1, CFG['K'] + 1):
        th = CFG['TH0'] * CFG['GA'] ** t
        g = np.where(m, np.exp((CFG['BETA'] if beta is None else beta) * sig(np.where(m, (qt - rm + th) / CFG['NU'], 0.0))), 0.0)
        p = p * g
        p = p / p.sum(1, keepdims=True)
        p = (1 - lam) * p + lam * u
    return p


def prior_misspecified(N, f=1.0, xi_mult=1.0):
    """Heuristic prior (Eq. 13) built for N_bar = f * N and slope xi_mult * 0.55 ms.

    With f = 1 and xi_mult = 1 this returns ``gipa_core.prior_heur(N)``
    exactly.
    """
    if f == 1.0 and xi_mult == 1.0:
        return prior_heur(N)
    M = CFG['M']
    c = np.zeros(M + 1)
    c[1:] = xi_mult * CFG['CONG'] * np.maximum(f * N / (M + 1) - 1.0, 0.0)
    return c


# ----------------------------------------------------------------------
# Policies with separate score (qs) and mask (qm) inputs
# ----------------------------------------------------------------------
def _mask(env, qm, cb, N, guard):
    """Prior-inclusive deadline mask with guard band, local fallback if empty."""
    D = env.dl[:N][:, None]
    m = ((-qm + cb[None, :]) * (1.0 + guard)) <= D
    m[~m.any(1), 0] = True
    return m


def gipa_sm(env, qs, qm, cb, rg, N, guard=0.0):
    """GIPA (``gipa_core.gipa`` with default switches), scores from qs, mask from qm."""
    isS = env.isS[:N]
    lam = np.where(isS, CFG['LAM_S'], CFG['LAM_B'])[:, None]
    m = _mask(env, qm, cb, N, guard)
    qt = qs - CFG['ETA'] * cb[None, :]
    Fs = m.sum(1, keepdims=True)
    z = np.where(m, qt / CFG['TAU'], -np.inf)
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    p = p / p.sum(1, keepdims=True)
    rm = np.where(m, qt, -np.inf).max(1, keepdims=True)
    for t in range(1, CFG['K'] + 1):
        th = CFG['TH0'] * CFG['GA'] ** t
        arg = (qt - rm + th) / CFG['NU']
        g = np.exp(CFG['BETA'] * sig(arg))
        g = np.where(m, g, 0.0)
        p = p * g
        p = p / p.sum(1, keepdims=True)
        p = (1 - lam) * p + lam * m / Fs
    return samp(p, rg)


def msoft_sm(env, qs, qm, cb, rg, N, use_prior=False, tau=None, guard=0.0):
    """Masked-softmax family (``gipa_core.masked_softmax``), scores qs, mask qm."""
    tau = tau or CFG['TAU']
    m = _mask(env, qm, cb, N, guard)
    sc = qs - (CFG['ETA'] * cb[None, :] if use_prior else 0)
    z = np.where(m, sc / tau, -np.inf)
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    p = p / p.sum(1, keepdims=True)
    return samp(p, rg)


def price_sm(env, qs, qm, rg, N, guard=0.0):
    """Congestion pricing (``gipa_core.price_rr``), scores qs, mask qm."""
    mu = np.zeros(CFG['M'] + 1)
    for _ in range(30):
        m = _mask(env, qm, mu, N, guard)
        z = np.where(m, (qs - mu[None, :]) / CFG['TAU'], -np.inf)
        z = z - z.max(1, keepdims=True)
        p = np.exp(z)
        p = p / p.sum(1, keepdims=True)
        load = p.sum(0)
        new = np.zeros(CFG['M'] + 1)
        new[1:] = kappa(load[1:])
        mu = 0.5 * mu + 0.5 * new
    m = _mask(env, qm, mu, N, guard)
    z = np.where(m, (qs - mu[None, :]) / CFG['TAU'], -np.inf)
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    p = p / p.sum(1, keepdims=True)
    return samp(p, rg)


POLICIES = ['greedy', 'indep', 'rbr', 'price', 'msoft', 'msoft_prior', 'entreg',
            'local', 'gipa', 'plan']
PLAN_PRED = ['plan_mask', 'gipa_plan']
"""Planned policies that use per-task predictions (E1b)."""
"""Policies supported by :func:`run_pred`. Q-learning is excluded: it never
uses latency predictions (its state is class and load bucket only)."""

PRIOR_POLICIES = ['msoft', 'msoft_prior', 'entreg', 'gipa', 'plan']
"""Policies whose decisions depend on the congestion prior (arm c)."""


def select_pred(pol, env, qs, qm, cb, rg, N, state, guard=0.0, **kw):
    """Dispatch a policy with separate score and mask inputs."""
    if pol == 'gipa':
        return gipa_sm(env, qs, qm, cb, rg, N, guard)
    if pol == 'greedy':
        return qs.argmax(1)
    if pol == 'indep':
        return msoft_sm(env, qs, qm, cb * 0, rg, N, guard=guard)
    if pol == 'msoft':
        return msoft_sm(env, qs, qm, cb, rg, N, guard=guard)
    if pol == 'msoft_prior':
        return msoft_sm(env, qs, qm, cb, rg, N, use_prior=True, guard=guard)
    if pol == 'entreg':
        return msoft_sm(env, qs, qm, cb, rg, N, use_prior=True,
                        tau=kw.get('tau', 1.2), guard=guard)
    if pol == 'price':
        return price_sm(env, qs, qm, rg, N, guard)
    if pol == 'rbr':
        return state['rbr'].act(env, qs, cb, rg, N)
    if pol == 'local':
        return np.zeros(N, int)
    if pol in ('plan', 'plan_mask', 'gipa_plan'):
        from planner import planned_pmf      # imported here to keep planner optional
        if 'P' not in state:
            state['P'] = planned_pmf(env, N, kw['plan'])
            state['Cp'] = plan_congestion(kw['plan'], kw.get('mask_z', 0.0))
            R0 = kw.get('R0')                            # Stage 0's latency model (default: the truth)
            if R0 is None:
                state['qbar'] = env.q(N)                 # class-average intrinsic scores (identical tasks)
            else:
                M = CFG['M']
                rS = -np.array([R0['L_S']] + [R0['E_S']] * M)
                rB = -np.array([R0['L_B']] + [R0['E_B']] * M)
                state['qbar'] = np.where(env.isS[:N, None], rS[None, :], rB[None, :])
        if pol == 'plan':
            return own_redirect(env, samp(state['P'], rg), state['P'], N, rg)
        if pol == 'plan_mask':
            return samp(plan_mask_probs(env, qm, state['Cp'], state['P'], N, guard), rg)
        return samp(gipa_plan_probs(env, qs, qm, state['Cp'], state['P'], state['qbar'], N, guard), rg)
    raise ValueError(f'unknown policy: {pol}')


# ----------------------------------------------------------------------
# Episode runner
# ----------------------------------------------------------------------
def run_pred(pol, N, seed, R, cb=None, arm='none', predictor='synthetic',
             sigma=0.0, bias=0.0, guard=0.0, truth='static', sigma_eta=0.0,
             rho=0.9, alpha=0.2, obs='breakdown', **kw):
    """One episode with predicted latencies.

    Parameters
    ----------
    pol : str
        One of :data:`POLICIES`.
    N, seed, R, cb
        As in ``gipa_core.run`` (``cb`` defaults to the heuristic prior).
    arm : {'none', 'score', 'mask', 'both'}
        Where the prediction is used (see module docstring).
    predictor : {'synthetic', 'oracle', 'ewma'}
    sigma, bias : float
        Synthetic relative-error standard deviation and mean.
    guard : float
        Guard-band fraction for the deadline mask.
    truth : {'static', 'ar1'}
    sigma_eta, rho : float
        AR(1) drift standard deviation and per-slot correlation.
    alpha : float
        EWMA smoothing factor.
    obs : {'breakdown', 'total', 'total_debiased'}
        What the EWMA predictor observes (see the module docstring).
    **kw
        ``plan=<dict>`` for the planned policies; ``R0=<regime dict>`` for the
        latency model Stage 0 used (class averages of 'gipa_plan'; default
        the truth); ``tau`` for 'entreg'.

    Returns
    -------
    dict of float
        Metrics described in the module docstring.
    """
    env = Env(N, seed, R)
    rg = np.random.default_rng(RNG_ACT_BASE + seed)
    rg_n = np.random.default_rng(RNG_NOISE_BASE + seed)
    cb = prior_heur(N) if cb is None else cb
    state = {'rbr': RBR()}
    M = CFG['M']
    q_s, q_b = capacities(R)
    isS = env.isS[:N]
    rows = np.arange(N)

    l0 = -env.q(N)                                    # nominal intrinsic latency (ms)
    eta = rg_n.standard_normal((N, M + 1)) * sigma_eta if truth == 'ar1' else None
    l_hat_ewma = l0.copy()
    keys = ['mS', 'mB', 'lat', 'fair', 'off', 'offS', 'offB', 'en', 'p95', 'maxl',
            'over_sel', 'pred_err']
    out = {k: [] for k in keys}

    for t in range(CFG['SLOTS']):
        # True intrinsic latency this slot.
        if truth == 'ar1':
            eta = rho * eta + np.sqrt(1 - rho ** 2) * sigma_eta * rg_n.standard_normal((N, M + 1))
            l_t = l0 * np.maximum(1.0 + eta, 0.1)
        else:
            l_t = l0
        # Prediction (a synthetic draw is made every slot: common random numbers).
        if predictor == 'synthetic':
            eps = np.maximum(bias + sigma * rg_n.standard_normal((N, M + 1)), EPS_FLOOR)
            l_hat = l_t * (1.0 + eps)
        elif predictor == 'ewma':
            l_hat = l_hat_ewma.copy()
        else:
            l_hat = l_t
        l_dec = l_t
        if LOCAL_CARRY == 'own':
            # The vehicle knows its own local backlog exactly (no communication).
            Wn = carry_backlog(env, N)
            l_dec = l_t.copy()
            l_dec[:, 0] += Wn
            l_hat = l_hat.copy()
            l_hat[:, 0] += Wn
        q_true, q_pred = -l_dec, -l_hat
        qs = q_pred if arm in ('score', 'both') else q_true
        qm = q_pred if arm in ('mask', 'both') else q_true

        a = select_pred(pol, env, qs, qm, cb, rg, N, state, guard, **kw)

        c = np.bincount(a, minlength=M + 1)
        cg = np.zeros(M + 1)
        cg[1:] = kappa(c[1:])
        cost = np.where(a == 0, l_t[:, 0], l_t[rows, a] + cg[a])
        w_seen = np.where(a == 0, carry_backlog(env, N), 0.0) if LOCAL_CARRY else 0.0
        cost = carry_apply(env, a, cost, l_t[:, 0], N)
        e = env.en(a, N)
        if predictor == 'ewma':
            if obs == 'breakdown':
                seen = l_t[rows, a]
            elif obs == 'total':
                seen = cost - w_seen                 # the vehicle knows its own wait
            elif obs == 'total_debiased':
                ant = plan_congestion(kw['plan']) if pol in ('plan', 'plan_mask', 'gipa_plan') else cb
                seen = np.maximum(cost - w_seen - ant[a], 0.1)
            else:
                raise ValueError(obs)
            l_hat_ewma[rows, a] = (1 - alpha) * l_hat_ewma[rows, a] + alpha * seen

        if t >= CFG['WARM']:
            S = isS
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
            over = [c[j] > effective_capacity(bool((S & (a == j)).any()), q_s, q_b)
                    for j in range(1, M + 1)]
            out['over_sel'].append(np.mean(over))
            out['pred_err'].append(float(np.mean(np.abs(l_hat - l_dec) / l_dec)))

    return {k: safe_mean(v) for k, v in out.items()}


def agg_pred(pol, N, R, cb=None, **kw):
    """Run :func:`run_pred` for seed indices 0..NSEED-1.

    Returns (across-seed means, list of per-seed dicts).
    """
    rs = [run_pred(pol, N, s, R, cb, **kw) for s in range(CFG['NSEED'])]
    return {k: safe_mean([r[k] for r in rs]) for k in rs[0]}, rs
