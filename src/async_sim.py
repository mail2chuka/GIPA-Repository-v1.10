"""
async_sim.py
============
Event-driven simulator with asynchronous task arrivals and persistent RSU
queues for workplan experiment E5, answering Reviewer 1 comment 6 ("the
system appears synchronous; tasks arrive over time") and the request to
carry unfinished work forward.

STATUS: written for the first revision of manuscript ITS-2026-08-0579. Not
part of the submitted v1.0.0 results. ``gipa_core`` is unchanged.

Model
-----
* N vehicles, classes drawn as in ``gipa_core.Env`` (environment seed
  1000 + s). Each vehicle generates tasks as a Poisson process with mean
  one task per 20 ms decision slot, so the fleet generates N tasks per 20 ms
  on average. Arrival times and vehicle ids come from a separate generator
  (11000 + s) and are identical for every policy (paired runs).
* Each task is decided at its own arrival time.
* Local execution (``local='none'``, the E5 default): the task takes the
  archived local latency L_k, as in the synchronous model (no local
  queue: the synchronous model also gives every slot's task a fresh local
  CPU). With ``local='fifo'`` (experiment E5b) each vehicle has one
  on-board processor that serves its local tasks in arrival order and
  carries unfinished work forward, so a local task waits for the earlier
  local tasks of the same vehicle; ``local='ps'`` shares the on-board
  processor among them instead (sensitivity). A background vehicle
  generates on average one 22 ms task every 20 ms (Regime A), so its
  processor is overloaded (utilisation 1.1) unless at least 1/11 of its
  tasks are offloaded.
* Offloading: the task spends the transfer time T_k of the E2 calibration
  (2.70 ms safety, 7.34 ms background) and then joins RSU j's queue, which
  is served by egalitarian processor sharing at the E2 capacity: a task
  needs a_k ms of service when alone (0.299 ms safety, 0.658 ms
  background). RSU queues persist for the whole run: unfinished work
  carries forward and is never reset at slot boundaries.
* Latency = completion time - arrival time; a miss is latency > D_k.
* Own-backlog awareness (``+own`` policies, E5b only): the vehicle knows
  its own local queue, which needs no communication. It predicts the
  latency of local execution as the time until its processor is free plus
  L_k (exact under FIFO), uses that prediction as the local score and in
  the deadline mask of GIPA, and for the plan redraws among the RSUs when
  local execution is predicted to miss the deadline.
* Optional admission control (``ac=True``, the asynchronous rule of E3): on
  arrival at the RSU a task is admitted only if T_k + a_k (n + 1) <= D_k,
  with n the tasks then in service there; otherwise it runs locally after a
  1 ms handshake penalty (latency T_k + 1 + L_k).

Processor sharing is simulated exactly with a virtual-time engine
(:class:`PSQueue`): virtual time advances at rate 1 / n(t), and a task
arriving at virtual time V with work w departs when virtual time reaches
V + w.

Policies and the information they use
-------------------------------------
Static (no runtime information; one per-class PMF for the whole run):
  'local', 'greedy' (best intrinsic score: RSU 1), 'indep' (masked softmax,
  no prior), 'msoft_prior_*' and 'gipa_*' with a prior, and 'plan'.
  Priors (extra delay over the uncongested RSU cost, ms):
    'slot'      the synchronous Stage 0 prior of E2 (class-aware Eq. 13,
                n_bar = N / (M + 1) co-located tasks); GIPA unchanged
    'soj_heur'  backlog-aware: processor-sharing mean sojourn
                a_k / (1 - rho) - a_k with the utilisation rho implied by
                Stage 0's uniform split (each RSU receives N / (M + 1) tasks
                per slot); infinite if rho >= 1
    'soj_plan'  the same formula with the utilisation of the asynchronous
                plan (:func:`plan_async`)
Broadcast (state of every RSU sampled every ``period`` ms and sent to all
vehicles; the same information for every policy that uses it):
  'rbr_b20', 'rbr_b100'    reactive best response: softmax (tau 0.4) on
                           q minus a_mix * n_j(broadcast)
  'gipa_b20', 'gipa_b100'  GIPA with prior a_k * n_j(broadcast)
  'jsq_live'               shortest expected delay with the RSU backlog
                           refreshed every 1 ms (near-live information;
                           a centralised-information reference)
Feedback:
  'la'  learning automata (E6 best configuration); each vehicle updates its
        own probability vector when one of its tasks completes. Decisions
        and updates are processed in 20 ms chunks.

Seeds
-----
Classes 1000 + s (as ``Env``), arrivals 11000 + s, decisions 5000 + s (one
uniform per task, in arrival order, shared by all policies).

Local-queue analysis (E5b planner)
----------------------------------
With independent sampling, the local tasks of a class-k vehicle form a
Poisson stream of rate (1 - f_k) / 20 per ms, where f_k = M x_k is the
offloaded share, served in L_k ms: an M/D/1 queue. Its mean sojourn is
L_k + rho L_k / (2 (1 - rho)) with rho = (1 - f_k) L_k / 20, and its
waiting time has the exact distribution of :func:`md1_wait_cdf`
(Erlang's formula). The planner adds both to :func:`plan_async`.

Validation
----------
:func:`check_table3` rebuilds the manuscript's Table III processor-sharing
check (5 ms tasks arriving over the first quarter of a 20 ms slot, 4,000
replications, seed 0). :func:`check_batch` checks that simultaneous
arrivals reproduce the E2 batch formula exactly. :func:`check_mm1` checks
the M/M/1-PS mean sojourn 1 / (mu - lambda).
"""

import heapq
import math

import numpy as np

from gipa_core import CFG, Env, RNG_ACT_BASE, sig

RNG_ARR_BASE = 11000
SLOT_MS = 20.0
HANDSHAKE_MS = 1.0
STATIC = ['local', 'greedy', 'indep', 'msoft_prior_slot', 'msoft_prior_soj_heur', 'msoft_prior_soj_plan',
          'gipa_slot', 'gipa_soj_heur', 'gipa_soj_plan', 'gipa_nodiff_slot', 'plan']
BROADCAST = {'rbr_b20': 20.0, 'rbr_b100': 100.0, 'gipa_b20': 20.0, 'gipa_b100': 100.0, 'jsq_live': 1.0}
OWN = ['gipa_slot+own', 'plan+own']          # own-backlog-aware variants (E5b)


# ----------------------------------------------------------------------
# Processor-sharing queue
# ----------------------------------------------------------------------
class PSQueue:
    """Egalitarian processor-sharing server with a virtual-time engine.

    Tasks are added with :meth:`add` (arrival time, work, id) in any order;
    :meth:`advance` processes arrivals and departures up to a time.
    """

    def __init__(self, admit=None):
        self.t = 0.0
        self.V = 0.0
        self.active = []          # heap of (finish virtual time, id)
        self.pending = []         # heap of (arrival time, id, work, extra)
        self.done = {}            # id -> departure time
        self.rejected = set()
        self.busy = 0.0
        self.admit = admit        # callable(n_in_service, extra) -> bool, or None

    def add(self, t, work, jid, extra=None):
        heapq.heappush(self.pending, (t, jid, work, extra))

    @property
    def n(self):
        return len(self.active)

    def advance(self, t_target):
        """Process every event with time <= t_target and move the clock there."""
        act, pend = self.active, self.pending
        while True:
            n = len(act)
            t_arr = pend[0][0] if pend else math.inf
            t_dep = self.t + (act[0][0] - self.V) * n if n else math.inf
            if n and t_dep <= t_arr and t_dep <= t_target:
                dt = max(t_dep - self.t, 0.0)
                self.V = act[0][0]
                self.busy += dt
                self.t = t_dep
                _, jid = heapq.heappop(act)
                self.done[jid] = t_dep
            elif pend and t_arr <= t_target:
                if n:
                    self.V += (t_arr - self.t) / n
                    self.busy += t_arr - self.t
                self.t = t_arr
                _, jid, work, extra = heapq.heappop(pend)
                if self.admit is not None and not self.admit(n, extra):
                    self.rejected.add(jid)
                    continue
                heapq.heappush(act, (self.V + work, jid))
            else:
                if math.isinf(t_target):
                    return                      # drained: nothing left to process
                if n and t_target > self.t:
                    self.V += (t_target - self.t) / n
                    self.busy += t_target - self.t
                self.t = max(self.t, t_target)
                return


class FIFOQueue:
    """Single-server first-in-first-out queue with the interface of :class:`PSQueue`.

    Under FIFO a task's departure is fixed when it arrives, so ``done`` holds
    the departure time as soon as :meth:`advance` passes the arrival; callers
    that need completed tasks compare ``done[jid]`` with the current time.
    """

    def __init__(self):
        self.t = 0.0
        self.free = 0.0           # time at which the server finishes its current work
        self.pending = []         # heap of (arrival time, id, work)
        self.done = {}
        self.rejected = set()     # never used; kept for interface compatibility
        self.busy = 0.0
        self.deps = []            # heap of departure times of tasks still in the system

    def add(self, t, work, jid, extra=None):
        heapq.heappush(self.pending, (t, jid, work))

    @property
    def n(self):
        return len(self.deps)

    def advance(self, t_target):
        pend = self.pending
        while pend and pend[0][0] <= t_target:
            t, jid, w = heapq.heappop(pend)
            start = max(t, self.free)
            self.free = start + w
            self.busy += w
            self.done[jid] = self.free
            heapq.heappush(self.deps, self.free)
        while self.deps and self.deps[0] <= t_target:
            heapq.heappop(self.deps)
        self.t = max(self.t, t_target)

    def predict(self, t, w):
        """Latency of a task of work ``w`` joining at time ``t`` (exact for FIFO)."""
        self.advance(t)
        return max(self.free - t, 0.0) + w


def ps_predict(q, t, w):
    """Latency of a task of work ``w`` joining PS queue ``q`` at ``t`` if nothing else arrives."""
    q.advance(t)
    rem = [fv - q.V for fv, _ in q.active]
    return sum(min(r, w) for r in rem) + w


def md1_wait_cdf(t, lam, s):
    """P(W <= t) for the FIFO M/D/1 queue (arrival rate lam, service time s), Erlang's formula.

    P(W <= t) = (1 - rho) sum_{k=0}^{floor(t/s)} [lam (k s - t)]^k / k! exp(-lam (k s - t)),
    with rho = lam s < 1. Exact; the sum has floor(t/s) + 1 terms.
    """
    rho = lam * s
    if rho >= 1:
        return 0.0
    if t < 0:
        return 0.0
    tot = 0.0
    for k in range(int(math.floor(t / s)) + 1):
        x = lam * (k * s - t)
        tot += x ** k / math.factorial(k) * math.exp(-x)
    return min(max((1 - rho) * tot, 0.0), 1.0)


def local_queue_stats(f, L, D, slot=SLOT_MS):
    """(rho, mean sojourn, miss probability) of a vehicle's local M/D/1 queue when a share f is offloaded."""
    lam = (1 - f) / slot
    rho = lam * L
    if f >= 1:
        return 0.0, L, 0.0
    if rho >= 1:
        return rho, math.inf, 1.0
    soj = L + rho * L / (2 * (1 - rho))
    miss = 1.0 - md1_wait_cdf(D - L, lam, L) if D >= L else 1.0
    return rho, soj, miss


def local_admit_stats(f, L, D, slot=SLOT_MS):
    """Local queue of an own-backlog-aware vehicle: (admitted share, mean local latency).

    The vehicle keeps a task local only if its predicted wait (the time until
    its processor is free) is at most theta = D - L; otherwise the task goes
    to an RSU. For theta < L every admission lifts the workload above theta,
    so admissions form a renewal process: after an admission with wait v the
    workload falls to theta after v + L - theta, and the next local candidate
    arrives Exp(lam) later and is admitted with wait max(theta - E, 0). Hence
    E[v] = theta - (1 - exp(-lam theta)) / lam and the admitted share is
    1 / (lam (E[v] + L - theta) + 1). Exact for FIFO and Poisson candidates;
    with theta = 0 it reduces to the M/D/1/1 loss system.
    """
    lam = (1 - f) / slot
    theta = D - L
    if f >= 1 or lam <= 0:
        return 0.0, L
    if theta < 0:
        return 0.0, L
    if theta >= L:
        raise ValueError('local_admit_stats assumes D - L < L')
    Ev = theta - (1 - math.exp(-lam * theta)) / lam if theta > 0 else 0.0
    adm = 1.0 / (lam * (Ev + L - theta) + 1.0)
    return adm, Ev + L


def check_admit(f=0.2, L=22.0, D=40.0, n=400000, seed=0):
    """Simulated admitted share and mean latency of :func:`local_admit_stats` (theory, simulation)."""
    lam = (1 - f) / SLOT_MS
    rg = np.random.default_rng(seed)
    arr = np.cumsum(rg.exponential(1 / lam, n))
    free, adm, lat = 0.0, 0, 0.0
    for t in arr:
        w = max(free - t, 0.0)
        if w <= D - L:
            free = t + w + L
            adm += 1
            lat += w + L
    th = local_admit_stats(f, L, D)
    return dict(admit_theory=th[0], admit_sim=adm / n, lat_theory=th[1], lat_sim=lat / adm)


def check_md1(lam=0.04, s=22.0, n=200000, seed=0):
    """FIFO M/D/1 against theory: (mean wait theory, simulated; P(W <= 18) theory, simulated)."""
    rg = np.random.default_rng(seed)
    arr = np.cumsum(rg.exponential(1 / lam, n))
    q = FIFOQueue()
    for i in range(n):
        q.add(arr[i], s, i)
    q.advance(math.inf)
    wait = np.array([q.done[i] - arr[i] - s for i in range(n)])[n // 10:]
    rho = lam * s
    return dict(mean_theory=rho * s / (2 * (1 - rho)), mean_sim=float(wait.mean()),
                cdf18_theory=md1_wait_cdf(18.0, lam, s), cdf18_sim=float((wait <= 18.0).mean()),
                cdf40_theory=md1_wait_cdf(40.0, lam, s), cdf40_sim=float((wait <= 40.0).mean()))


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
def check_table3(counts=(5, 10, 20, 40), reps=4000, seed=0, work=5.0, spread=5.0):
    """Mean PS sojourn for ``count`` tasks of ``work`` ms arriving uniformly in [0, spread] ms.

    Returns {count: (closed-form count * work, discrete-event mean)}. The
    manuscript's Table III reports 21.64, 45.90, 95.48 and 195.25 ms; the
    original script is not in the repository, so the random stream differs
    and agreement is expected to Monte-Carlo precision only.
    """
    rg = np.random.default_rng(seed)
    out = {}
    for c in counts:
        tot = 0.0
        for _ in range(reps):
            arr = rg.uniform(0, spread, c)
            q = PSQueue()
            for i, a in enumerate(arr):
                q.add(a, work, i)
            q.advance(math.inf)
            tot += np.mean([q.done[i] - arr[i] for i in range(c)])
        out[c] = (c * work, tot / reps)
    return out


def check_batch(seed=0, tol=1e-9):
    """Simultaneous arrivals reproduce the batch formula sum_l min(w_l, w_i)."""
    rg = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(200):
        n = rg.integers(1, 40)
        w = np.where(rg.random(n) < 0.3, 0.299, 0.658)
        q = PSQueue()
        for i in range(n):
            q.add(0.0, w[i], i)
        q.advance(math.inf)
        ref = np.array([np.minimum(w, w[i]).sum() for i in range(n)])
        got = np.array([q.done[i] for i in range(n)])
        worst = max(worst, np.abs(got - ref).max())
    return worst <= tol, worst


def check_mm1(lam=0.8, mu=1.0, n=200000, seed=0):
    """M/M/1-PS mean sojourn against 1 / (mu - lambda); returns (theory, simulated)."""
    rg = np.random.default_rng(seed)
    arr = np.cumsum(rg.exponential(1 / lam, n))
    w = rg.exponential(1 / mu, n)
    q = PSQueue()
    for i in range(n):
        q.add(arr[i], w[i], i)
    q.advance(math.inf)
    soj = np.array([q.done[i] - arr[i] for i in range(n)])
    return 1 / (mu - lam), float(soj[n // 10:].mean())


# ----------------------------------------------------------------------
# Probabilities for static policies (per class)
# ----------------------------------------------------------------------
def gipa_probs(q, C, dl, lam, mask_on=True, lam_on=True):
    """GIPA PMF for rows of scores q and priors C (same steps as calibration.gipa_pv)."""
    lam = lam[:, None] * (1.0 if lam_on else 0.0)
    D = dl[:, None]
    m = ((-q + C) <= D) if mask_on else np.ones_like(q, bool)
    m[~m.any(1), 0] = True
    qt = q - CFG['ETA'] * C
    Fs = m.sum(1, keepdims=True)
    z = np.where(m, qt / CFG['TAU'], -np.inf)
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    p = p / p.sum(1, keepdims=True)
    rm = np.where(m, qt, -np.inf).max(1, keepdims=True)
    for t in range(1, CFG['K'] + 1):
        th = CFG['TH0'] * CFG['GA'] ** t
        g = np.exp(CFG['BETA'] * sig((qt - rm + th) / CFG['NU']))
        g = np.where(m, g, 0.0)
        p = p * g
        p = p / p.sum(1, keepdims=True)
        p = (1 - lam) * p + lam * m / Fs
    return p


def msoft_probs(q, C, dl, use_prior=True, tau=None):
    """Masked softmax PMF (calibration.msoft_pv)."""
    tau = tau or CFG['TAU']
    m = (-q + C) <= dl[:, None]
    m[~m.any(1), 0] = True
    sc = q - (CFG['ETA'] * C if use_prior else 0)
    z = np.where(m, sc / tau, -np.inf)
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def check_gipa_probs(N=160, seeds=range(3)):
    """gipa_probs + samp reproduces calibration.gipa_pv draws exactly."""
    from gipa_core import REG, samp
    from calibration import PSEnv, gipa_pv, make_calibration, prior_matrix
    cal = make_calibration()
    for s in seeds:
        env = PSEnv(N, s, REG['A'], cal)
        q = env.q(N)
        C = prior_matrix(env, N, cal, 'class')
        lam = np.where(env.isS[:N], CFG['LAM_S'], CFG['LAM_B'])
        a1 = samp(gipa_probs(q, C, env.dl[:N], lam), np.random.default_rng(s))
        a2 = gipa_pv(env, q, C, np.random.default_rng(s), N)
        if not np.array_equal(a1, a2):
            return False
    return True


# ----------------------------------------------------------------------
# Planner for the asynchronous model
# ----------------------------------------------------------------------
def plan_async(N, R, cal, rho_max=0.9, kappa_dl=3.0, grid=201, local='none', eps_loc=0.05, rho_loc_max=0.95):
    """Symmetric per-class offload probabilities minimising expected latency.

    Arrival rates: lam_k = pi_k N / 20 tasks per ms. Each RSU receives
    lam_k x_k of class k (x_k = probability of choosing that RSU), with
    utilisation rho = sum_k lam_k x_k a_k. Expected RSU latency for class k
    is T_k + a_k / (1 - rho) (M/G/1-PS mean sojourn, insensitive to the
    service distribution). Constraints: rho <= rho_max; for each class that
    offloads, T_k + kappa_dl * a_k / (1 - rho) <= D_k (a margin for the
    sojourn spread); tasks kept local must meet their local deadline.
    With ``local`` other than 'none' (E5b), local execution is the M/D/1
    queue of :func:`local_queue_stats`: its mean sojourn enters the
    objective, its utilisation must not exceed ``rho_loc_max`` and its
    deadline-miss probability must not exceed ``eps_loc`` for any class
    that keeps tasks local.
    Returns dict(x_S, x_B, rho, J) or None.
    """
    M = CFG['M']
    lS = CFG['PI_S'] * N / SLOT_MS
    lB = (1 - CFG['PI_S']) * N / SLOT_MS
    best = None
    for xs in np.linspace(0, 1 / M, grid):
        if R['L_S'] > R['D_S'] and xs < 1 / M - 1e-12:
            continue
        for xb in np.linspace(0, 1 / M, grid):
            if R['L_B'] > R['D_B'] and xb < 1 / M - 1e-12:
                continue
            rho = lS * xs * cal['a_S'] + lB * xb * cal['a_B']
            if rho > rho_max:
                continue
            sS = cal['a_S'] / (1 - rho)
            sB = cal['a_B'] / (1 - rho)
            if xs > 0 and cal['T_S'] + kappa_dl * sS > R['D_S']:
                continue
            if xb > 0 and cal['T_B'] + kappa_dl * sB > R['D_B']:
                continue
            if local == 'none':
                locS, locB = R['L_S'], R['L_B']
            elif local == 'fifo_own':
                # Own-backlog-aware vehicles: tasks predicted to miss locally move to the RSUs.
                aS_, locS = local_admit_stats(M * xs, R['L_S'], R['D_S'])
                aB_, locB = local_admit_stats(M * xb, R['L_B'], R['D_B'])
                gS = M * xs + (1 - M * xs) * (1 - aS_)          # effective offloaded shares
                gB = M * xb + (1 - M * xb) * (1 - aB_)
                rho = (lS * gS * cal['a_S'] + lB * gB * cal['a_B']) / M
                if rho > rho_max:
                    continue
                sS = cal['a_S'] / (1 - rho)
                sB = cal['a_B'] / (1 - rho)
                if gS > 0 and cal['T_S'] + kappa_dl * sS > R['D_S']:
                    continue
                if gB > 0 and cal['T_B'] + kappa_dl * sB > R['D_B']:
                    continue
                J = (lS * ((1 - gS) * locS + gS * (cal['T_S'] + sS))
                     + lB * ((1 - gB) * locB + gB * (cal['T_B'] + sB)))
                if best is None or J < best[0]:
                    best = (J, xs, xb, rho)
                continue
            else:
                rS, locS, mS = local_queue_stats(M * xs, R['L_S'], R['D_S'])
                rB, locB, mB = local_queue_stats(M * xb, R['L_B'], R['D_B'])
                if rS > rho_loc_max or rB > rho_loc_max or mS > eps_loc or mB > eps_loc:
                    continue
            J = (lS * ((1 - M * xs) * locS + M * xs * (cal['T_S'] + sS))
                 + lB * ((1 - M * xb) * locB + M * xb * (cal['T_B'] + sB)))
            if best is None or J < best[0]:
                best = (J, xs, xb, rho)
    if best is None:
        return None
    J, xs, xb, rho = best
    out = dict(x_S=float(xs), x_B=float(xb), rho=float(rho), J=float(J / (lS + lB)),
               rho_max=rho_max, kappa_dl=kappa_dl, local=local)
    if local == 'fifo_own':
        aS_, lS_ = local_admit_stats(M * xs, R['L_S'], R['D_S'])
        aB_, lB_ = local_admit_stats(M * xb, R['L_B'], R['D_B'])
        out.update(admit_S=aS_, admit_B=aB_, lat_loc_S=lS_, lat_loc_B=lB_,
                   off_eff_S=M * xs + (1 - M * xs) * (1 - aS_), off_eff_B=M * xb + (1 - M * xb) * (1 - aB_))
    elif local != 'none':
        rS, sojS, mS = local_queue_stats(M * xs, R['L_S'], R['D_S'])
        rB, sojB, mB = local_queue_stats(M * xb, R['L_B'], R['D_B'])
        out.update(eps_loc=eps_loc, rho_loc_max=rho_loc_max, rho_loc_S=rS, rho_loc_B=rB,
                   soj_loc_S=sojS, soj_loc_B=sojB, miss_loc_S=mS, miss_loc_B=mB)
    return out


# ----------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------
def soj_prior(a, rho):
    """Extra PS sojourn over the isolated service time a: a / (1 - rho) - a."""
    return a * rho / (1 - rho) if rho < 1 else 1e6


def static_pmfs(pol, N, R, cal, plan=None):
    """(2, M + 1) PMF for [safety, background] for a static policy."""
    M = CFG['M']
    M1 = M + 1
    q = np.array([[-R['L_S']] + [-cal['e_S']] * M, [-R['L_B']] + [-cal['e_B']] * M])
    dl = np.array([R['D_S'], R['D_B']])
    lam = np.array([CFG['LAM_S'], CFG['LAM_B']])
    pi = CFG['PI_S']
    aS, aB = cal['a_S'], cal['a_B']
    if pol == 'local':
        return np.tile(np.eye(M1)[0], (2, 1))
    if pol == 'greedy':
        return np.eye(M1)[q.argmax(1)]
    if pol == 'plan':
        return np.array([[1 - M * plan['x_S']] + [plan['x_S']] * M, [1 - M * plan['x_B']] + [plan['x_B']] * M])
    if pol == 'indep':
        return msoft_probs(q, np.zeros_like(q), dl, use_prior=False)
    kind = pol.split('_', 2)[-1] if pol.startswith('msoft_prior') else pol.split('_', 1)[1]
    if pol.startswith('gipa_nodiff'):
        kind = 'slot'
    C = np.zeros((2, M1))
    if kind == 'slot':
        nbar = N / M1
        C[0, 1:] = aS * max(nbar - 1, 0)
        C[1, 1:] = aS * pi * nbar + aB * max((1 - pi) * nbar - 1, 0)
    elif kind in ('soj_heur', 'soj_plan'):
        if kind == 'soj_heur':
            rho = (N / SLOT_MS / M1) * (pi * aS + (1 - pi) * aB)
        else:
            rho = plan['rho'] if plan is not None else 1.0
        C[0, 1:] = soj_prior(aS, rho)
        C[1, 1:] = soj_prior(aB, rho)
    else:
        raise ValueError(pol)
    if pol.startswith('msoft_prior'):
        return msoft_probs(q, C, dl)
    return gipa_probs(q, C, dl, lam, lam_on=not pol.startswith('gipa_nodiff'))


def own_tables(base, N, R, cal, plan, res=0.05):
    """Look-up PMFs for the own-backlog-aware policies, per class.

    Returns, for each class c, (L_c, nbins, cumsum rows): row i is the PMF
    when the predicted local latency is L_c + i * res (rounded up); the last
    row is the PMF with local execution masked (prediction above D_c).
    'gipa_slot': GIPA with the slot prior, local score -prediction and local
    masked above the deadline. 'plan': the plan, redrawn among the RSUs when
    local execution is masked.
    """
    M = CFG['M']
    M1 = M + 1
    P0 = static_pmfs(base, N, R, cal, plan)
    out = []
    for c, (Lc, Dc, ec) in enumerate([(R['L_S'], R['D_S'], cal['e_S']), (R['L_B'], R['D_B'], cal['e_B'])]):
        nb = int(math.ceil((Dc - Lc) / res)) + 1 if Dc >= Lc else 0
        pred = np.concatenate([Lc + res * np.arange(nb), [Dc + 1.0]])
        if base == 'plan':
            rows = np.tile(P0[c], (len(pred), 1))
            rsu = P0[c, 1:]
            rows[-1] = np.concatenate([[0.0], rsu / rsu.sum() if rsu.sum() > 0 else np.ones(M) / M])
        else:
            nbar = N / M1
            pi = CFG['PI_S']
            Cc = np.zeros(M1)
            Cc[1:] = cal['a_S'] * max(nbar - 1, 0) if c == 0 else cal['a_S'] * pi * nbar + cal['a_B'] * max((1 - pi) * nbar - 1, 0)
            q = np.column_stack([-pred] + [np.full(len(pred), -ec)] * M)
            C = np.tile(Cc, (len(pred), 1))
            lam = np.full(len(pred), CFG['LAM_S'] if c == 0 else CFG['LAM_B'])
            rows = gipa_probs(q, C, np.full(len(pred), Dc), lam)
        out.append((Lc, nb, np.cumsum(rows, 1)))
    return out


def run_async(pol, N, seed, R, cal, horizon_ms=20000.0, ac=False, plan=None, la_cfg=None, sample_ms=20.0,
              local='none'):
    """Simulate one run; returns the metric dict (see the module docstring).

    ``local``: 'none' (E5: fresh local processor for every task), 'fifo' or
    'ps' (E5b: one persistent on-board processor per vehicle). Policies
    ending in '+own' need ``local`` other than 'none'.

    Metrics are computed over tasks arriving after the warm-up point chosen
    by MSER-5 on the total RSU backlog (sampled every ``sample_ms``),
    truncated to at most half the run. Queues are drained after the
    horizon so every counted task has a completion time.
    """
    M = CFG['M']
    M1 = M + 1
    env = Env(N, seed, R)
    isS_v = env.isS[:N]
    rg_arr = np.random.default_rng(RNG_ARR_BASE + seed)
    lam_tot = N / SLOT_MS
    n_est = int(lam_tot * horizon_ms * 1.2) + 1000
    gaps = rg_arr.exponential(1 / lam_tot, n_est)
    t_arr = np.cumsum(gaps)
    t_arr = t_arr[t_arr < horizon_ms]
    K = len(t_arr)
    veh = rg_arr.integers(0, N, K)
    cls = np.where(isS_v[veh], 0, 1)                         # 0 safety, 1 background
    u = np.random.default_rng(RNG_ACT_BASE + seed).random(K)
    L = np.array([R['L_S'], R['L_B']])[cls]
    D = np.array([R['D_S'], R['D_B']])[cls]
    T = np.array([cal['T_S'], cal['T_B']])[cls]
    W = np.array([cal['a_S'], cal['a_B']])[cls]

    admit = None
    if ac:
        def admit(n, extra):
            Tk, ak, Dk = extra
            return Tk + ak * (n + 1) <= Dk
    queues = [PSQueue(admit) for _ in range(M)]
    act = np.zeros(K, dtype=int)

    own = pol.endswith('+own')
    base = pol[:-4] if own else pol
    LQ = None
    if local != 'none':
        LQ = [FIFOQueue() if local == 'fifo' else PSQueue() for _ in range(N)]
        if ac and (own or base in ('jsq_live', 'la')):
            raise ValueError('admission control is not combined with policies that read the local queue')
    elif own:
        raise ValueError('+own policies need a local queue')
    loc_t = np.full(K, np.nan)            # time each task joins its vehicle's local processor

    def lpredict(v, t, w):
        q = LQ[v]
        return q.predict(t, w) if local == 'fifo' else ps_predict(q, t, w)

    def ladd(k, t):
        loc_t[k] = t
        if LQ is not None:
            LQ[veh[k]].add(t, L[k], int(k))

    def push(idx, with_local=True):
        for k in idx:
            j = act[k]
            if j > 0:
                queues[j - 1].add(t_arr[k] + T[k], W[k], int(k), (T[k], W[k], D[k]))
            elif with_local:
                ladd(k, t_arr[k])

    def sample_rows(P_rows, idx):
        c = np.cumsum(P_rows, 1)
        return np.minimum((u[idx, None] > c).sum(1), M1 - 1)

    # Grid of sample times for the backlog series.
    grid_t = np.arange(sample_ms, horizon_ms + 1e-9, sample_ms)
    backlog = np.zeros((len(grid_t), M))
    busyser = np.zeros((len(grid_t), M))

    if base in STATIC:
        if own:
            tabs = own_tables(base, N, R, cal, plan)
            for k in range(K):
                c = cls[k]
                Lc, nb, cum = tabs[c]
                pred = lpredict(veh[k], t_arr[k], L[k])
                i = int(math.ceil((pred - Lc) / 0.05 - 1e-9)) if pred <= D[k] else nb
                row = cum[min(max(i, 0), nb)]
                a = min(int((u[k] > row).sum()), M1 - 1)
                act[k] = a
                if a == 0:
                    ladd(k, t_arr[k])
            push(range(K), with_local=False)
        else:
            P = static_pmfs(pol, N, R, cal, plan)
            act[:] = sample_rows(P[cls], np.arange(K))
            push(range(K))
        for g, tg in enumerate(grid_t):
            for j in range(M):
                queues[j].advance(tg)
                backlog[g, j] = queues[j].n
                busyser[g, j] = queues[j].busy
    else:
        period = BROADCAST.get(pol, SLOT_MS)
        if pol == 'la':
            P_la = np.tile(np.ones(M1) / M1, (N, 1))
            rate, algo = la_cfg['rate'], la_cfg['algo']
            eps = 0.1
            local_done = []                      # heap of (time, k)
            pending_rsu = set()
            pending_loc = set()
        q_cls = np.array([[-R['L_S']] + [-cal['e_S']] * M, [-R['L_B']] + [-cal['e_B']] * M])
        dl_cls = np.array([R['D_S'], R['D_B']])
        lam_cls = np.array([CFG['LAM_S'], CFG['LAM_B']])
        a_mix = CFG['PI_S'] * cal['a_S'] + (1 - CFG['PI_S']) * cal['a_B']
        g = 0
        k0 = 0
        t0 = 0.0
        while t0 < horizon_ms:
            t1 = min(t0 + period, horizon_ms)
            for j in range(M):
                queues[j].advance(t0)
            nj = np.array([queues[j].n for j in range(M)], float)
            k1 = int(np.searchsorted(t_arr, t1))
            idx = np.arange(k0, k1)
            if len(idx):
                if pol.startswith('rbr'):
                    cg = np.concatenate([[0.0], a_mix * nj])
                    z = (q_cls - cg[None, :]) / CFG['TAU']
                    z = z - z.max(1, keepdims=True)
                    Pc = np.exp(z)
                    Pc /= Pc.sum(1, keepdims=True)
                    act[idx] = sample_rows(Pc[cls[idx]], idx)
                elif pol.startswith('gipa_b'):
                    C = np.zeros((2, M1))
                    C[0, 1:] = cal['a_S'] * nj
                    C[1, 1:] = cal['a_B'] * nj
                    Pc = gipa_probs(q_cls, C, dl_cls, lam_cls)
                    act[idx] = sample_rows(Pc[cls[idx]], idx)
                elif pol == 'jsq_live':
                    est = np.zeros((2, M1))
                    est[:, 0] = [R['L_S'], R['L_B']]
                    est[0, 1:] = cal['T_S'] + cal['a_S'] * (nj + 1)
                    est[1, 1:] = cal['T_B'] + cal['a_B'] * (nj + 1)
                    if LQ is None:
                        act[idx] = est.argmin(1)[cls[idx]]
                    else:
                        # The vehicle also sees its own local queue (no communication needed).
                        for k in idx:
                            e = est[cls[k]].copy()
                            e[0] = lpredict(veh[k], t_arr[k], L[k])
                            act[k] = int(e.argmin())
                            if act[k] == 0:
                                ladd(k, t_arr[k])
                elif pol == 'la':
                    act[idx] = sample_rows(P_la[veh[idx]], idx)
                else:
                    raise ValueError(pol)
                push(idx, with_local=not (pol == 'jsq_live' and LQ is not None))
            if pol == 'la':
                # Feedback for tasks completed by t1 (local completions are deterministic).
                for j in range(M):
                    queues[j].advance(t1)
                for k in idx:
                    if act[k] == 0:
                        if LQ is None:
                            heapq.heappush(local_done, (t_arr[k] + L[k], int(k)))
                        else:
                            pending_loc.add(int(k))
                    else:
                        pending_rsu.add(int(k))
                fin = []
                while local_done and local_done[0][0] <= t1:
                    fin.append(heapq.heappop(local_done)[1])
                if LQ is not None:
                    for v in {int(veh[k]) for k in pending_loc}:
                        LQ[v].advance(t1)
                    for k in list(pending_loc):
                        dk = LQ[veh[k]].done.get(k)
                        if dk is not None and dk <= t1:
                            fin.append(k)
                            pending_loc.discard(k)
                for k in list(pending_rsu):
                    j = act[k] - 1
                    if k in queues[j].done or k in queues[j].rejected:
                        fin.append(k)
                        pending_rsu.discard(k)
                for k in sorted(fin):
                    j = act[k]
                    if j == 0:
                        lat_k = L[k] if LQ is None else LQ[veh[k]].done[k] - t_arr[k]
                    elif k in queues[j - 1].rejected:
                        lat_k = T[k] + HANDSHAKE_MS + L[k]
                    else:
                        lat_k = queues[j - 1].done[k] - t_arr[k]
                    met = lat_k <= D[k]
                    beta = float(met and lat_k <= L[k] + 1e-12)
                    v = veh[k]
                    p = P_la[v]
                    e = np.zeros(M1)
                    e[j] = 1.0
                    p += rate * beta * (e - p)
                    if algo == 'lrep' and beta == 0.0:
                        others = (e == 0)
                        p += eps * rate * (others / others.sum() - p)
                    p /= p.sum()
            while g < len(grid_t) and grid_t[g] <= t1 + 1e-9:
                for j in range(M):
                    queues[j].advance(grid_t[g])
                    backlog[g, j] = queues[j].n
                    busyser[g, j] = queues[j].busy
                g += 1
            k0 = k1
            t0 = t1

    # Drain: no new arrivals after the horizon.
    for j in range(M):
        queues[j].advance(math.inf)
    if LQ is not None:
        # Tasks rejected by admission control run on their own processor after the handshake.
        for j in range(M):
            for k in queues[j].rejected:
                ladd(k, t_arr[k] + T[k] + HANDSHAKE_MS)
        for q in LQ:
            q.advance(math.inf)

    lat = np.empty(K)
    rej = np.zeros(K, bool)
    rsu_soj = np.full(K, np.nan)
    for k in range(K):
        j = act[k]
        if j == 0:
            lat[k] = L[k] if LQ is None else LQ[veh[k]].done[k] - t_arr[k]
        else:
            qj = queues[j - 1]
            if k in qj.rejected:
                rej[k] = True
                lat[k] = T[k] + HANDSHAKE_MS + L[k] if LQ is None else LQ[veh[k]].done[k] - t_arr[k]
            else:
                lat[k] = qj.done[k] - t_arr[k]
                rsu_soj[k] = qj.done[k] - (t_arr[k] + T[k])

    # Local backlog per vehicle class (tasks waiting or in service on their own processor).
    if LQ is not None:
        isloc = ~np.isnan(loc_t)
        loc_dep = t_arr + lat
        bl_cls = np.zeros((len(grid_t), 2))
        for c in (0, 1):
            sel_c = isloc & (cls == c)
            a_s = np.sort(loc_t[sel_c])
            d_s = np.sort(loc_dep[sel_c])
            bl_cls[:, c] = np.searchsorted(a_s, grid_t, 'right') - np.searchsorted(d_s, grid_t, 'right')
        nveh = np.array([max(int((~isS_v).sum() if c else isS_v.sum()), 1) for c in (0, 1)])

    # Warm-up: MSER-5 on the total backlog, truncation at most half the run.
    tot = backlog.sum(1) + (bl_cls.sum(1) if LQ is not None else 0.0)
    nb = len(tot) // 5
    bm = tot[:nb * 5].reshape(nb, 5).mean(1)
    best_d, best_v = 0, math.inf
    for d in range(0, nb // 2 + 1):
        tail = bm[d:]
        v = tail.var() / len(tail)
        if v < best_v:
            best_d, best_v = d, v
    t_warm = best_d * 5 * sample_ms
    keep = t_arr >= t_warm
    S = cls == 0
    miss = lat > D

    # Backlog drift over the second half (tasks per second, per RSU) and stability.
    half = len(grid_t) // 2
    x = grid_t[half:] / 1000.0
    drift = [float(np.polyfit(x, backlog[half:, j], 1)[0]) for j in range(M)]
    grow = [float(backlog[-len(x) // 5:, j].mean() - backlog[half:half + len(x) // 5, j].mean()) for j in range(M)]
    unstable = bool(max(drift) > 5.0 and max(grow) > 20.0)
    # Utilisation over the measured window: busy time / elapsed time between the warm-up point and the horizon.
    gw = int(np.searchsorted(grid_t, t_warm))
    b0 = busyser[gw - 1] if gw > 0 else np.zeros(M)
    busy_frac = [float((busyser[-1, j] - b0[j]) / (grid_t[-1] - (grid_t[gw - 1] if gw > 0 else 0.0))) for j in range(M)]
    off = act > 0
    sel = np.bincount(act[keep], minlength=M1)[1:]
    adm = np.bincount(act[keep & ~rej], minlength=M1)[1:]
    span_s = (horizon_ms - t_warm) / 1000.0
    out = dict(
        tasks=int(keep.sum()), t_warm_ms=float(t_warm),
        lat=float(lat[keep].mean()), p95=float(np.percentile(lat[keep], 95)),
        mS=float(miss[keep & S].mean()) if (keep & S).any() else 0.0,
        mB=float(miss[keep & ~S].mean()) if (keep & ~S).any() else 0.0,
        miss=float(miss[keep].mean()),
        off=float(off[keep].mean()), offS=float(off[keep & S].mean()), offB=float(off[keep & ~S].mean()),
        rej=float(rej[keep].sum() / max(off[keep].sum(), 1)),
        soj_mean=float(np.nanmean(rsu_soj[keep])) if np.isfinite(rsu_soj[keep]).any() else None,
        soj_p95=float(np.nanpercentile(rsu_soj[keep], 95)) if np.isfinite(rsu_soj[keep]).any() else None,
        backlog_mean=float(backlog[grid_t >= t_warm].mean()),
        backlog_max=float(backlog.max()),
        backlog_cv=float(np.nanmean([np.std(r) / np.mean(r) for r in backlog[grid_t >= t_warm] if r.sum() > 0]))
        if (backlog[grid_t >= t_warm].sum(1) > 0).any() else None,
        drift_max=float(max(drift)), unstable=unstable,
        util_mean=float(np.mean(busy_frac)), util_max=float(max(busy_frac)), util=busy_frac,
        sel_rate=[float(v / span_s) for v in sel], adm_rate=[float(v / span_s) for v in adm],
    )
    if LQ is not None:
        # Local-processor metrics (E5b). Per-vehicle backlog drift over the second half, by class.
        drift_l = [float(np.polyfit(x, bl_cls[half:, c] / nveh[c], 1)[0]) for c in (0, 1)]
        grow_l = [float((bl_cls[-len(x) // 5:, c].mean() - bl_cls[half:half + len(x) // 5, c].mean()) / nveh[c])
                  for c in (0, 1)]
        unstable_local = bool(any(d > 0.5 and g > 2.0 for d, g in zip(drift_l, grow_l)))
        locm = keep & isloc
        ex = np.where(rej, T + HANDSHAKE_MS + L, L)
        busy_loc = [float(L[keep & isloc & (cls == c)].sum() / (nveh[c] * (horizon_ms - t_warm))) for c in (0, 1)]
        out.update(
            unstable_rsu=unstable, unstable_local=unstable_local, unstable=bool(unstable or unstable_local),
            share_local=float(locm.sum() / max(keep.sum(), 1)),
            lat_local=float(lat[locm].mean()) if locm.any() else None,
            wait_local=float((lat[locm] - ex[locm]).mean()) if locm.any() else None,
            miss_local=float(miss[locm].mean()) if locm.any() else None,
            backlog_local_S=float(bl_cls[grid_t >= t_warm, 0].mean() / nveh[0]),
            backlog_local_B=float(bl_cls[grid_t >= t_warm, 1].mean() / nveh[1]),
            drift_local_S=drift_l[0], drift_local_B=drift_l[1],
            util_local_S=busy_loc[0], util_local_B=busy_loc[1],
            lat_S=float(lat[keep & S].mean()) if (keep & S).any() else None,
            lat_B=float(lat[keep & ~S].mean()) if (keep & ~S).any() else None,
        )
    return out
