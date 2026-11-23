"""
gipa_core.py
============
Shared simulator, GIPA decision operator, baseline policies, congestion
priors and episode runners for the GIPA evaluation (IET ITS manuscript
ITS-2026-08-0579).

Every experiment script imports from this module, so the environment, the
policies and the metric definitions exist in exactly one place.

What the simulator is (and is not)
----------------------------------
This is a *controlled abstract-congestion* environment, not a packet- or
cycle-level network simulator. All latencies are fixed millisecond constants:

* Local execution of a task costs a class-specific constant
  (``REG[r]['L_S']`` or ``REG[r]['L_B']``).
* Offloading to an RSU costs a class-specific uncongested constant
  (``E_S`` or ``E_B``) plus ``CFG['CONG']`` (0.55 ms) for every *other*
  task offloaded to the same RSU in the same slot, regardless of that
  task's class (see :func:`kappa`). This is the affine model of manuscript
  Section VI-A, not the processor-sharing model of Eq. (5).
* There are no CPU frequencies, cycle counts, data sizes or channel models.
* Each vehicle's task class is drawn once per episode and repeats in every
  slot, so the tasks are identical across the 70 slots of an episode; only
  the action sampling changes from slot to slot.

Units
-----
Latencies, deadlines, congestion priors and intrinsic scores are in
milliseconds. Energy values are dimensionless "stylized" units.

Seeds
-----
environment RNG : 1000 + seed index      (seed indices 0..NSEED-1)
action RNG      : 5000 + seed index
Q-learning      : 5999 (exploration during training); training environments
                  cycle through seed indices 0..6 (``step % 7``)

Reproducibility
---------------
Every random draw uses ``numpy.random.default_rng`` with an enumerated seed,
and the order of draws is fixed, so results reproduce bit for bit. The code
in this file is behaviourally identical to the v1.0.0 archive; this revision
adds documentation only.

History
-------
This module replaces an earlier pattern in which diag.py and final.py
re-executed fragments of exp.py via exec(open(...).read().split(...)). That
pattern caused a NameError in diag.py because the split fragment did not
contain the self-consistent-prior functions. Everything shared is now defined
here once and imported normally.
"""

import os

import numpy as np

# ----------------------------------------------------------------------
# Carried-over local work (experiment E13, November 2026)
# ----------------------------------------------------------------------
# The slotted model gives every slot's task an idle on-board processor. With
# the environment variable GIPA_LOCAL_CARRY set, each vehicle's processor
# instead carries unfinished local work into the next slot:
#     W_i(t + 1) = max(W_i(t) + s_i(t) - SLOT_MS, 0),
# where s_i(t) is the local service time of the task executed locally in
# slot t (0 if it was offloaded), and a local task's latency is W_i(t) plus
# its service time. Tasks arrive at the start of each 20 ms slot.
#   GIPA_LOCAL_CARRY=blind  decisions unchanged (the vehicle ignores its own
#                           backlog); only realised latency changes
#   GIPA_LOCAL_CARRY=own    the vehicle also uses its own backlog, which needs
#                           no communication: the local intrinsic score becomes
#                           -(W_i + service), so the deadline mask (8) and
#                           every score-based policy see it, and the planned
#                           policies redraw among the RSUs of the plan when
#                           local execution is predicted to miss its deadline
# Unset (the default), every result reproduces the archived values.
LOCAL_CARRY = os.environ.get('GIPA_LOCAL_CARRY', '').strip().lower()
if LOCAL_CARRY not in ('', 'blind', 'own'):
    raise ValueError(f'GIPA_LOCAL_CARRY must be blind or own, not {LOCAL_CARRY!r}')
SLOT_MS = 20.0


def carry_backlog(env, N):
    """Current local backlog W_i (ms) of the first N vehicles of ``env``."""
    W = getattr(env, '_W', None)
    if W is None:
        W = env._W = np.zeros(env.N)
    return W[:N]


def carry_apply(env, a, cost, service, N):
    """Add each local task's wait to its latency and advance the backlog by one slot.

    ``service`` is the local service time per vehicle (ms). Returns the new
    latency vector; does nothing when GIPA_LOCAL_CARRY is unset.
    """
    if not LOCAL_CARRY:
        return cost
    W = carry_backlog(env, N).copy()
    loc = a == 0
    cost = np.where(loc, cost + W, cost)
    work = np.zeros(env.N)
    work[:N] = np.where(loc, service, 0.0)
    env._W = np.maximum(env._W + work - SLOT_MS, 0.0)
    return cost


def own_redirect(env, a, P, N, rg):
    """Planned policies, own mode: move a task whose local execution is predicted to miss to an RSU of the plan.

    The redraw uses the plan's RSU probabilities renormalised (uniform if the
    plan offloads nothing). Draws one uniform per vehicle from ``rg`` only in
    own mode, so the default stream is unchanged.
    """
    if LOCAL_CARRY != 'own':
        return a
    W = carry_backlog(env, N)
    bad = (a == 0) & (W + env.loc[:N] > env.dl[:N])
    u = rg.random(N)
    if not bad.any():
        return a
    rs = P[:, 1:]
    tot = rs.sum(1, keepdims=True)
    rs = np.where(tot > 0, rs / np.maximum(tot, 1e-300), 1.0 / rs.shape[1])
    j = np.minimum((u[:, None] > np.cumsum(rs, 1)).sum(1), rs.shape[1] - 1) + 1
    return np.where(bad, j, a)


def _carry_wrap(cls):
    """Wrap a class's own ``lat`` and ``q`` so that every Env subclass carries local work."""
    if 'lat' in cls.__dict__ and not getattr(cls.__dict__['lat'], '_carry', False):
        orig_lat = cls.__dict__['lat']

        def lat(self, a, N, _f=orig_lat):
            cost, c = _f(self, a, N)
            return carry_apply(self, a, cost, self.loc[:N], N), c
        lat._carry = True
        lat.__doc__ = orig_lat.__doc__
        cls.lat = lat
    if 'q' in cls.__dict__ and not getattr(cls.__dict__['q'], '_carry', False):
        orig_q = cls.__dict__['q']

        def q(self, N, _f=orig_q):
            out = _f(self, N)
            if LOCAL_CARRY == 'own':
                out = out.copy()
                out[:, 0] -= carry_backlog(self, N)
            return out
        q._carry = True
        q.__doc__ = orig_q.__doc__
        cls.q = q
    return cls


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
# Operator parameters correspond to manuscript Table II.
CFG = dict(
    M=3,            # number of RSUs
    CONG=0.55,      # ms of added edge delay per co-located offloader (xi_j)
    PI_S=0.30,      # fraction of tasks that are safety critical (pi_S)
    TAU=0.4,        # softmax temperature (tau, Eq. 15)
    BETA=0.4,       # amplification gain (beta, Eq. 18)
    NU=0.3,         # grading width (nu, Eq. 18)
    TH0=1.5,        # anneal start (theta_0, Eq. 16)
    GA=0.6,         # anneal decay (gamma_theta, Eq. 16)
    K=4,            # amplification-diffusion rounds (K)
    ETA=1.0,        # prior weight (eta, Eq. 14)
    LAM_S=0.35,     # diffusion (floor) rate, safety class (lambda_S, Eq. 19)
    LAM_B=0.60,     # diffusion (floor) rate, background class (lambda_B)
    SLOTS=70,       # slots per episode
    WARM=25,        # warm-up slots discarded
    NSEED=30,       # independent seeds
    E_LS=2.6,       # relative energy, safety local
    E_LB=5.4,       # relative energy, background local
    E_TX=1.2,       # relative energy per offload transmit
)

# Two workload regimes (manuscript Section VI-B). All values in ms.
#   D_S, D_B : class deadlines
#   L_S, L_B : local execution latency per class
#   E_S, E_B : uncongested RSU latency per class
#   A: local execution meets every deadline (the manuscript's primary regime)
#   B: safety local execution is INFEASIBLE (14 > 12 ms), so the edge tier
#      has a purpose; used as a diagnostic stress test
REG = {
    'A': dict(D_S=12., D_B=40., L_S=10., L_B=22., E_S=3., E_B=8.),
    'B': dict(D_S=12., D_B=40., L_S=14., L_B=30., E_S=3., E_B=8.),
}

RNG_ENV_BASE = 1000
RNG_ACT_BASE = 5000
RNG_QL_TRAIN = 5999


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def sig(z):
    """Numerically guarded logistic function sigma(z) = 1 / (1 + exp(-z)).

    The argument is clipped to [-40, 40] to avoid overflow in ``exp``.
    """
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def kappa(load):
    """Affine slot-local congestion delay in ms, first occupant free.

    kappa(n) = CONG * max(n - 1, 0), with CONG = 0.55 ms (manuscript
    Section IV-B and VI-A). ``load`` may be a scalar or an array of task
    counts; non-integer (expected) counts are allowed and are used by the
    priors. The delay is the same for every co-located task, whatever its
    class.
    """
    return CFG['CONG'] * np.maximum(load - 1.0, 0.0)


def samp(p, rg):
    """Draw one action per row of a probability matrix by inverse CDF.

    Parameters
    ----------
    p : ndarray, shape (N, n_actions)
        Row-stochastic matrix; row i is vehicle i's action PMF.
    rg : numpy.random.Generator
        Action-sampling generator. Exactly one uniform draw is consumed per
        row, so the draw order is fixed and reproducible.

    Returns
    -------
    ndarray of int, shape (N,)
        Sampled action indices (0 = local, j >= 1 = RSU j). The index is
        capped at n_actions - 1 to guard against rounding in the cumulative
        sum.
    """
    c = np.cumsum(p, 1)
    u = rg.random((p.shape[0], 1))
    return np.minimum((u > c).sum(1), p.shape[1] - 1)


def safe_mean(values):
    """Mean that returns NaN for an all-NaN input without emitting a warning.

    Replaces a bare np.nanmean call that raised 'Mean of empty slice' when
    every seed produced an undefined Jain index (no RSU received a task).
    The returned value is unchanged; only the warning is removed.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float('nan')
    return float(np.nanmean(arr))


# ----------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------
class Env:
    """Controlled abstract-congestion VEC environment with two task classes.

    Action space: 0 = local execution, j = 1..M = offload the whole task to
    RSU j. Coverage is universal (every vehicle reaches every RSU) and there
    is no mobility, dwell time or handover.

    Parameters
    ----------
    N : int
        Number of active vehicles (fleet size).
    seed : int
        Seed index; the environment RNG uses ``RNG_ENV_BASE + seed``.
    R : dict
        A workload regime from :data:`REG`.

    Attributes
    ----------
    isS : ndarray of bool, shape (N,)
        True for safety-critical vehicles. Drawn once, Bernoulli(PI_S); the
        class is fixed for the whole episode.
    dl : ndarray, shape (N,)
        Per-vehicle deadline in ms (D_S or D_B).
    loc : ndarray, shape (N,)
        Per-vehicle local latency in ms (L_S or L_B).
    e0 : ndarray, shape (N,)
        Per-vehicle uncongested RSU latency in ms (E_S or E_B).
    n : int
        Number of actions, M + 1.
    """

    def __init__(self, N, seed, R):
        self.N = N
        self.R = R
        self.n = CFG['M'] + 1
        rg = np.random.default_rng(RNG_ENV_BASE + seed)
        self.isS = rg.random(N) < CFG['PI_S']
        self.dl = np.where(self.isS, R['D_S'], R['D_B'])
        self.loc = np.where(self.isS, R['L_S'], R['L_B'])
        self.e0 = np.where(self.isS, R['E_S'], R['E_B'])

    def q(self, N):
        """Oracle intrinsic action scores q_{i,a}: negative intrinsic cost in ms.

        Higher is better. Column 0 is -local latency; columns 1..M are
        -uncongested RSU latency. No congestion and no prediction error are
        included: these are the 'oracle intrinsic scores' of manuscript
        Section VI-A, shared by every policy.

        Returns
        -------
        ndarray, shape (N, M + 1)
        """
        q = np.empty((N, self.n))
        q[:, 0] = -self.loc[:N]
        q[:, 1:] = -self.e0[:N, None]
        return q

    def lat(self, a, N):
        """Realised latency for a joint action, plus the per-action load vector.

        Local tasks cost their local latency. An offloaded task costs its
        uncongested RSU latency plus kappa(n_j), where n_j is the number of
        tasks sent to the same RSU in this slot (affine model).

        Parameters
        ----------
        a : ndarray of int, shape (N,)
            Chosen action per vehicle.
        N : int
            Number of active vehicles.

        Returns
        -------
        latency : ndarray, shape (N,)
            Realised end-to-end latency per vehicle in ms.
        c : ndarray of int, shape (M + 1,)
            Number of vehicles choosing each action; c[0] is local, c[1:]
            are the RSU loads n_j(t).
        """
        c = np.bincount(a, minlength=self.n)
        cg = np.zeros(self.n)
        cg[1:] = kappa(c[1:])
        return np.where(a == 0, self.loc[:N], self.e0[:N] + cg[a]), c

    def en(self, a, N):
        """Stylized energy per vehicle for a joint action (dimensionless units).

        Local execution costs E_LS or E_LB by class; any offload costs E_TX.
        This is the 'stylized energy model' of the manuscript, not a
        physical energy measurement.
        """
        el = np.where(self.isS[:N], CFG['E_LS'], CFG['E_LB'])
        return np.where(a == 0, el, CFG['E_TX'])

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        _carry_wrap(cls)


_carry_wrap(Env)


# ----------------------------------------------------------------------
# Priors (Stage 0)
# ----------------------------------------------------------------------
def prior_heur(N):
    """Stage 0 heuristic congestion prior, manuscript Eq. (13).

    Assumes the anticipated fleet splits uniformly over all M + 1 actions,
    so each RSU expects N / (M + 1) tasks, and returns the corresponding
    affine delay kappa(N / (M + 1)) for every RSU. The local action has
    prior 0.

    Returns
    -------
    ndarray, shape (M + 1,)
        Prior vector c_bar in ms. With M = 3: 10.45, 21.45, 32.45 and
        43.45 ms at N = 80, 160, 240 and 320.
    """
    c = np.zeros(CFG['M'] + 1)
    c[1:] = kappa(N / (CFG['M'] + 1))
    return c


def prior_workload(N, R, roff=(0.5, 0.5)):
    """Count-based workload prior, a simplified form of manuscript Eq. (12).

    The expected offloaded share is sum_k pi_k * r_k, where ``roff`` gives
    the planned offload propensity r_k for (safety, background). Offloaded
    tasks are assumed to split uniformly across the M RSUs. ``R`` is
    accepted for interface symmetry but not used.
    """
    pk = (CFG['PI_S'], 1 - CFG['PI_S'])
    exp_off = sum(p * r for p, r in zip(pk, roff))
    c = np.zeros(CFG['M'] + 1)
    c[1:] = kappa(N * exp_off / CFG['M'])
    return c


def prior_class(N, frac):
    """Class-aware prior computed from one class's load only.

    ``frac`` is the fraction of the fleet assumed to offload, split
    uniformly across the M RSUs. Not used by the principal experiment.
    """
    c = np.zeros(CFG['M'] + 1)
    c[1:] = kappa(N * frac / CFG['M'])
    return c


# ----------------------------------------------------------------------
# GIPA operator (Stage 1) and ablations
# ----------------------------------------------------------------------
def gipa(env, q, cb, rg, N, anneal=True, graded=True, lam_on=True,
         amp_on=True, mask_on=True, prior_on=True):
    """One slot of the GIPA decision operator for all N vehicles (Algorithm 1).

    Steps and the manuscript equations they implement:

    1. Predicted-feasibility mask (Eq. 7): an action is feasible when its
       predicted latency, intrinsic cost plus prior, is within the vehicle's
       deadline. Coverage and dwell time are not modelled here. If no action
       is feasible, local execution is forced (the forced-local safeguard
       of Proposition 2 / Assumption A1).
    2. Congestion-adjusted scores q_tilde = q - eta * c_bar (Eq. 14).
    3. Masked-softmax initial PMF with temperature tau (Eq. 15).
    4. K rounds of annealed graded reweighting (Eqs. 16-18) followed by
       class-specific diffusion toward the uniform PMF over the feasible
       set (Eq. 19).
    5. One categorical draw per vehicle (Eq. 20).

    Every vehicle uses only its own scores and the shared prior; no load
    information or inter-vehicle messages are used.

    Parameters
    ----------
    env : Env
    q : ndarray, shape (N, M + 1)
        Intrinsic scores from :meth:`Env.q`.
    cb : ndarray, shape (M + 1,)
        Congestion prior c_bar in ms (e.g. :func:`prior_heur`).
    rg : numpy.random.Generator
        Action-sampling generator.
    N : int
    anneal, graded, lam_on, amp_on, mask_on, prior_on : bool
        Ablation switches: annealed offset (False = fixed theta_0),
        graded weights (False = binary threshold), diffusion,
        amplification, deadline mask and prior adjustment of the scores.
        Note that ``prior_on=False`` removes the prior from the scores only;
        the mask still uses it.

    Returns
    -------
    ndarray of int, shape (N,)
        Sampled actions.
    """
    isS = env.isS[:N]
    lam = np.where(isS, CFG['LAM_S'], CFG['LAM_B'])[:, None]
    if not lam_on:
        lam = lam * 0
    D = env.dl[:N][:, None]

    # Step 1: predicted latency = intrinsic cost + prior; mask on the deadline.
    pred = -q + cb[None, :]
    m = (pred <= D) if mask_on else np.ones_like(pred, bool)
    m[~m.any(1), 0] = True                       # local fallback (Assumption A1)

    # Step 2: congestion-adjusted scores (Eq. 14).
    qt = q - (CFG['ETA'] * cb[None, :] if prior_on else 0)
    Fs = m.sum(1, keepdims=True)                 # |F_i(t)|, feasible-set size

    # Step 3: masked softmax initialisation (Eq. 15).
    z = np.where(m, qt / CFG['TAU'], -np.inf)
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    p = p / p.sum(1, keepdims=True)

    # Step 4: K rounds of reweighting and diffusion (Eqs. 16-19).
    rm = np.where(m, qt, -np.inf).max(1, keepdims=True)   # m_F,i (Eq. 17)
    for t in range(1, CFG['K'] + 1):
        th = CFG['TH0'] * (CFG['GA'] ** t if anneal else 1.0)    # theta_r (Eq. 16)
        if amp_on:
            arg = (qt - rm + th) / CFG['NU']
            g = np.exp(CFG['BETA'] * (sig(arg) if graded else (arg > 0).astype(float)))  # Eq. 18
            g = np.where(m, g, 0.0)
            p = p * g
            p = p / p.sum(1, keepdims=True)                    # normalised reweighting
        p = (1 - lam) * p + lam * m / Fs                       # diffusion (Eq. 19)

    # Step 5: sample (Eq. 20).
    return samp(p, rg)


# ----------------------------------------------------------------------
# Baseline policies
# ----------------------------------------------------------------------
def masked_softmax(env, q, cb, rg, N, use_prior=False, tau=None):
    """Masked softmax baselines (manuscript Eqs. 45-46).

    The mask is the same prior-inclusive deadline mask GIPA uses. With
    ``use_prior=False`` the scores are the raw intrinsic scores (masked
    softmax); with ``use_prior=True`` they are prior-adjusted (prior-adjusted
    masked softmax). Passing ``tau=1.2`` with ``use_prior=True`` gives the
    entropy-regularised baseline. Calling it with a zero prior (``cb * 0``)
    gives the 'independent softmax' baseline, whose mask is then effectively
    deadline-only on uncongested costs.
    """
    tau = tau or CFG['TAU']
    D = env.dl[:N][:, None]
    m = (-q + cb[None, :]) <= D
    m[~m.any(1), 0] = True
    sc = q - (CFG['ETA'] * cb[None, :] if use_prior else 0)
    z = np.where(m, sc / tau, -np.inf)
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    p = p / p.sum(1, keepdims=True)
    return samp(p, rg)


def price_rr(env, q, cb, rg, N):
    """Congestion pricing with randomised rounding (manuscript Eq. 47).

    Runs 30 damped fixed-point iterations (damping 0.5) in which the price
    mu_j is kappa of the *fleet-wide* expected load implied by the current
    masked softmax. The iteration therefore uses aggregate information about
    all vehicles' PMFs, i.e. it is a centralised computation. ``cb`` is not
    used; the price replaces the prior both in the mask and in the scores.
    """
    mu = np.zeros(CFG['M'] + 1)
    D = env.dl[:N][:, None]
    for _ in range(30):
        m = (-q + mu[None, :]) <= D
        m[~m.any(1), 0] = True
        z = np.where(m, (q - mu[None, :]) / CFG['TAU'], -np.inf)
        z = z - z.max(1, keepdims=True)
        p = np.exp(z)
        p = p / p.sum(1, keepdims=True)
        load = p.sum(0)
        new = np.zeros(CFG['M'] + 1)
        new[1:] = kappa(load[1:])
        mu = 0.5 * mu + 0.5 * new
    m = (-q + mu[None, :]) <= D
    m[~m.any(1), 0] = True
    z = np.where(m, (q - mu[None, :]) / CFG['TAU'], -np.inf)
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    p = p / p.sum(1, keepdims=True)
    return samp(p, rg)


def greedy(env, q, cb, rg, N):
    """Deterministic greedy: each vehicle picks its highest intrinsic score.

    With identical RSUs, ties resolve to the lowest index, so every vehicle
    that offloads picks RSU 1. No randomness is consumed.
    """
    return q.argmax(1)


def alllocal(env, q, cb, rg, N):
    """All-local baseline: every task executes on its own vehicle."""
    return np.zeros(N, int)


class RBR:
    """Reactive best response to the live per-RSU load (not fictitious play).

    Each slot, every vehicle softmax-responds to congestion computed from the
    *previous* slot's realised load shares, which the policy is assumed to
    observe (a per-slot load broadcast that GIPA does not receive). It keeps
    no belief over past actions. No feasibility mask is applied. The first
    slot of an episode assumes uniform shares.
    """

    def __init__(self):
        self.mu = None

    def act(self, env, q, cb, rg, N):
        """Sample actions given last slot's load shares; update the shares."""
        mu = self.mu if self.mu is not None else np.full(env.n, 1. / env.n)
        cg = np.zeros(env.n)
        cg[1:] = kappa(mu[1:] * N)
        z = (q - cg[None, :]) / CFG['TAU']
        z = z - z.max(1, keepdims=True)
        p = np.exp(z)
        p = p / p.sum(1, keepdims=True)
        a = samp(p, rg)
        self.mu = np.bincount(a, minlength=env.n) / N
        return a


class QL:
    """Tabular learner over (class, load bucket) with a shaped reward.

    Specification (exactly as implemented):

    * One Q-table, shape (2 classes, 4 load buckets, M + 1 actions), shared
      by all vehicles. Class index 0 = safety, 1 = background.
    * State bucket = min(int(4 * max_j share of the fleet at RSU j in the
      previous slot), 3).
    * Reward per vehicle = -(latency / deadline) - pen * miss, pen = 3.0.
    * Update: Q <- Q + lr * (r - Q). There is no bootstrapped next-state
      term, so the discount factor is effectively 0 (a contextual bandit
      with running-average values).
    * One training step = one slot in which all N vehicles act; their N
      updates are applied in sequence. Default 15,000 steps.
    * Exploration: epsilon = max(0.05, 1 - step / (0.7 * steps)), i.e.
      linear from 1 to 0.05 over the first 70% of training.
    * Learning rate 0.10, constant. Exploration seed RNG_QL_TRAIN = 5999.
    * At evaluation: greedy action plus 2% uniformly random actions.

    Known issues, to be fixed in the revision (they are documented here and
    deliberately NOT changed in this file, so that v1.0.0 results still
    reproduce bit for bit):

    * Training environments use ``Env(N, st % 7, R)``, i.e. seed indices
      0-6, which are also evaluation seed indices 0-6 (train/test overlap).
    * Only one training run (one exploration seed).
    * ``self.mu`` (the previous-slot load) is not reset between evaluation
      episodes, so each episode starts from the last state of the previous
      one.
    """

    def __init__(self, n):
        self.Q = np.zeros((2, 4, n))
        self.n = n
        self.mu = None

    def bucket(self):
        """Load bucket 0-3 from the previous slot's largest RSU share."""
        return 0 if self.mu is None else min(int(self.mu[1:].max() * 4), 3)

    def train(self, N, R, steps=15000, pen=3.0, lr=0.10):
        """Train the shared table at fleet size N in regime R (see class docstring)."""
        rg = np.random.default_rng(RNG_QL_TRAIN)
        self.mu = np.full(self.n, 1. / self.n)
        for st in range(steps):
            eps = max(0.05, 1 - st / (0.7 * steps))
            env = Env(N, st % 7, R)
            q = env.q(N)
            b = self.bucket()
            cls = (~env.isS[:N]).astype(int)
            greedy_a = self.Q[cls, b].argmax(1)
            rnd = rg.integers(0, self.n, N)
            a = np.where(rg.random(N) < eps, rnd, greedy_a)
            lat, c = env.lat(a, N)
            miss = lat > env.dl[:N]
            rew = -(lat / env.dl[:N]) - pen * miss
            for i in range(N):
                self.Q[cls[i], b, a[i]] += lr * (rew[i] - self.Q[cls[i], b, a[i]])
            self.mu = c / N

    def act(self, env, q, cb, rg, N):
        """Greedy action from the table with 2% random exploration."""
        b = self.bucket()
        cls = (~env.isS[:N]).astype(int)
        a = self.Q[cls, b].argmax(1)
        f = rg.random(N) < 0.02
        a[f] = rg.integers(0, self.n, f.sum())
        self.mu = np.bincount(a, minlength=self.n) / N
        return a


# ----------------------------------------------------------------------
# Self-consistent prior (defined here so every script can import it)
# ----------------------------------------------------------------------
def Phi(cb, N, R, seed=0):
    """Induced-congestion map on the deployed K-round operator output.

    Given a candidate prior ``cb``, runs the GIPA PMF construction (without
    sampling) for one environment draw, sums the final PMFs into an expected
    load per RSU, and returns kappa of that load. This is the count-based
    map audited in manuscript Appendix G.3, not the full heterogeneous
    workload map of Eq. (42). The PMF steps duplicate those in :func:`gipa`.
    """
    env = Env(N, seed, R)
    q = env.q(N)
    isS = env.isS[:N]
    lam = np.where(isS, CFG['LAM_S'], CFG['LAM_B'])[:, None]
    D = env.dl[:N][:, None]
    m = (-q + cb[None, :]) <= D
    m[~m.any(1), 0] = True
    qt = q - CFG['ETA'] * cb[None, :]
    Fs = m.sum(1, keepdims=True)
    z = np.where(m, qt / CFG['TAU'], -np.inf)
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    p = p / p.sum(1, keepdims=True)
    rm = np.where(m, qt, -np.inf).max(1, keepdims=True)
    for t in range(1, CFG['K'] + 1):
        th = CFG['TH0'] * CFG['GA'] ** t
        g = np.exp(CFG['BETA'] * sig((qt - rm + th) / CFG['NU']))
        g = np.where(m, g, 0.)
        p = p * g
        p = p / p.sum(1, keepdims=True)
        p = (1 - lam) * p + lam * m / Fs
    load = p.sum(0)
    out = np.zeros(env.n)
    out[1:] = kappa(load[1:])
    return out


def sc_prior(N, R, d=0.5, iters=400, tol=1e-6):
    """Damped iteration toward a self-consistent prior (Appendix G.3).

    cb <- (1 - d) * cb + d * Phi(cb), starting from zero, until the
    sup-norm change falls below ``tol`` (ms) or ``iters`` is reached.

    Returns (prior, iterations_used, final_residual). The iteration is an
    empirical procedure with no convergence guarantee; it fails to converge
    at low density by oscillating across a feasibility-mask boundary, which
    the caller must check via the returned residual.
    """
    cb = np.zeros(CFG['M'] + 1)
    k = 0
    r = float('inf')
    for k in range(iters):
        nx = (1 - d) * cb + d * Phi(cb, N, R)
        r = float(np.abs(nx - cb).max())
        cb = nx
        if r < tol:
            break
    return cb, k + 1, r


# ----------------------------------------------------------------------
# Episode runner and aggregation
# ----------------------------------------------------------------------
def run(pol, N, seed, R, cb=None, **kw):
    """Run one episode of one policy and return per-metric means.

    An episode has CFG['SLOTS'] = 70 slots; the first CFG['WARM'] = 25 are
    warm-up and are not recorded, so 45 slots are evaluated. Each metric is
    computed per slot and then averaged over the evaluated slots:

    mS, mB   : deadline-miss rate of safety / background tasks (Eq. 8)
    lat      : mean realised latency over all vehicles, ms
    p95      : 95th percentile of latency across vehicles *within each
               slot*, then averaged over slots (note: this is a per-slot
               p95, not a pooled per-run p95)
    fair     : Jain index over RSU loads (Eq. 10); NaN in a slot where no
               task is offloaded, and NaN slots are skipped in the average
    off      : overall offload ratio; offS, offB : per-class offload ratio (Eq. 9)
    en       : mean stylized energy per vehicle
    maxl     : largest RSU load in the slot (tasks), averaged over slots

    Parameters
    ----------
    pol : str
        One of 'gipa', 'greedy', 'indep', 'msoft', 'msoft_prior', 'entreg',
        'price', 'rbr', 'ql', 'local'.
    N : int
        Fleet size.
    seed : int
        Seed index (environment seed 1000 + seed, action seed 5000 + seed).
    R : dict
        Regime from :data:`REG`.
    cb : ndarray, optional
        Congestion prior; defaults to :func:`prior_heur` (N).
    **kw
        Passed to :func:`gipa` (ablation switches); ``tau`` for 'entreg';
        ``ql`` (a trained :class:`QL`) is required for 'ql'.

    Returns
    -------
    dict of float
    """
    env = Env(N, seed, R)
    rg = np.random.default_rng(RNG_ACT_BASE + seed)
    cb = prior_heur(N) if cb is None else cb
    rbr = RBR()
    out = dict(mS=[], mB=[], lat=[], fair=[], off=[], offS=[], offB=[],
               en=[], p95=[], maxl=[])

    for t in range(CFG['SLOTS']):
        q = env.q(N)
        if pol == 'gipa':
            a = gipa(env, q, cb, rg, N, **kw)
        elif pol == 'greedy':
            a = greedy(env, q, cb, rg, N)
        elif pol == 'indep':
            a = masked_softmax(env, q, cb * 0, rg, N)
        elif pol == 'msoft':
            a = masked_softmax(env, q, cb, rg, N)
        elif pol == 'msoft_prior':
            a = masked_softmax(env, q, cb, rg, N, use_prior=True)
        elif pol == 'entreg':
            a = masked_softmax(env, q, cb, rg, N, use_prior=True, tau=kw.get('tau', 1.2))
        elif pol == 'price':
            a = price_rr(env, q, cb, rg, N)
        elif pol == 'rbr':
            a = rbr.act(env, q, cb, rg, N)
        elif pol == 'ql':
            a = kw['ql'].act(env, q, cb, rg, N)
        elif pol == 'local':
            a = alllocal(env, q, cb, rg, N)
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
            # Jain index is undefined when no RSU receives a task.
            out['fair'].append(np.nan if rl.sum() == 0
                               else rl.sum() ** 2 / (len(rl) * (rl ** 2).sum()))
            out['off'].append((a > 0).mean())
            out['offS'].append((a[S] > 0).mean() if S.any() else 0.)
            out['offB'].append((a[~S] > 0).mean() if (~S).any() else 0.)
            out['en'].append(e.mean())
            out['maxl'].append(rl.max())

    return {k: safe_mean(v) for k, v in out.items()}


def agg(pol, N, R, cb=None, **kw):
    """Run :func:`run` for seed indices 0..NSEED-1 and average.

    Returns
    -------
    (dict, list of dict)
        The across-seed mean of each metric (NaN-aware), and the per-seed
        result dictionaries in seed order.
    """
    rs = [run(pol, N, s, R, cb, **kw) for s in range(CFG['NSEED'])]
    return {k: safe_mean([r[k] for r in rs]) for k in rs[0]}, rs
