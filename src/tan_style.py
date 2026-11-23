"""
tan_style.py  (revision experiment E15, 16 Nov 2026)
====================================================
A Tan-style re-solved probabilistic plan as a baseline for GIPA, on the E12
heterogeneous environment (hetero.HetEnv: Eq. (5) processor sharing,
Regime A deadlines, lognormal task and link heterogeneity, optional partial
coverage).

Tan et al. (IEEE TVT 71(12), 2022) relax RSU selection to one probability
distribution per coverage edge (all vehicles of the edge share it), solve it
at the RSUs before every offloading period from the predicted aggregate
workload, enforce RSU load limits in expectation only, and let vehicles act
on the broadcast probabilities with no per-task feasibility check. The
adaptation used here keeps those properties and gives the baseline the
information it assumes:

* one action PMF per (class k, coverage group g), where g is the set of RSUs
  a vehicle reaches; local execution is an action (Tan has no local option);
* re-solved before EVERY slot from that slot's predicted aggregates:
  counts n_kg, mean predicted local latency (including the mean own backlog
  when local work carries over), mean predicted transfer time per reachable
  RSU, and the class-mean compute time per task a_k = mean(c_hat) / F;
* expected-latency model of the E2 planner, per RSU j with expected counts
  s_j (safety) and b_j (background):
      safety     T + a_S (s_j + b_j)
      background T + a_S s_j + a_B b_j
  so the expected total latency is a convex quadratic in the probabilities
  (a_B >= a_S);
* RSU deadline constraints on the expected load (z = 0, Tan) or on the load
  plus z binomial standard deviations (z = 2, GIPA's margin, as a variant);
* a group whose mean local latency exceeds its deadline must offload in full.

Two configurations are solved (safety tasks allowed at every RSU, or no
safety offloading) and the cheaper feasible one is kept, as the symmetric
planner does through its conditional constraints.
"""

import numpy as np
from scipy.optimize import minimize

from gipa_core import CFG, LOCAL_CARRY, samp, sig
from hetero import F_GHZ, GUARD, RATE, RNG_STAGE0, C_MEAN, D_MEAN, lognorm1, restrict

M = CFG['M']


# ----------------------------------------------------------------------
# Coverage groups
# ----------------------------------------------------------------------
def coverage_groups(reach):
    """Return (list of group masks over RSUs, per-vehicle group index)."""
    keys, gid = np.unique(reach, axis=0, return_inverse=True)
    return [k.astype(bool) for k in keys], gid.ravel()


# ----------------------------------------------------------------------
# Tan-style planner
# ----------------------------------------------------------------------
class TanPlanner:
    """Re-solved per-slot plan; see the module docstring."""

    def __init__(self, z=0.0, relax_fallback=False):
        self.z = float(z)
        self.relax_fallback = relax_fallback
        self.warm = {}
        self.fail = 0
        self.solves = 0

    def aggregates(self, env, t):
        """Slot aggregates from predicted features of the vehicles in ``env``."""
        e = env.cost(t, predicted=True)                 # (N, M+1); col 0 includes own backlog in own mode
        c_hat, d_hat = env.c_hat[t], env.d_hat[t]
        groups, gid = coverage_groups(env.reach)
        k = env.k
        a = np.array([c_hat[k == kk].mean() / F_GHZ if (k == kk).any() else C_MEAN[kk] / F_GHZ for kk in (0, 1)])
        trans = d_hat[:, None] / (RATE * env.g)         # (N, M)
        agg = []
        for kk in (0, 1):
            for gi, gm in enumerate(groups):
                sel = (k == kk) & (gid == gi)
                n = int(sel.sum())
                if n == 0:
                    continue
                agg.append(dict(k=kk, g=gi, n=n, L=float(e[sel, 0].mean()),
                                T=trans[sel].mean(0), rsus=np.flatnonzero(gm)))
        return agg, groups, gid, a

    def solve(self, env, t):
        agg, groups, gid, a = self.aggregates(env, t)
        aS, aB = float(a[0]), float(max(a[1], a[0]))
        R = env.R
        D = (R['D_S'], R['D_B'])
        best = None
        for cfg in ('A', 'B'):
            res = self._solve_cfg(agg, aS, aB, D, cfg)
            if res is not None and (best is None or res[0] < best[0]):
                best = res
        self.solves += 1
        if best is None:
            self.fail += 1
            if self.relax_fallback:
                # Generous fallback: drop the RSU deadline constraints and minimise expected latency.
                for cfg in ('A', 'B'):
                    res = self._solve_cfg(agg, aS, aB, D, cfg, relax=True)
                    if res is not None and (best is None or res[0] < best[0]):
                        best = res
            if best is None:
                best = self._fallback(agg)
        J, var, x = best
        # Per-vehicle PMF and planned congestion per RSU.
        X = np.zeros((2, len(groups), M))
        for (ai, j), xv in zip(var, x):
            X[agg[ai]['k'], agg[ai]['g'], j] = xv
        s = np.zeros(M)
        b = np.zeros(M)
        for (ai, j), xv in zip(var, x):
            if agg[ai]['k'] == 0:
                s[j] += agg[ai]['n'] * xv
            else:
                b[j] += agg[ai]['n'] * xv
        P = np.zeros((env.N, M + 1))
        P[:, 1:] = np.clip(X[env.k, gid], 0, 1)
        P[:, 1:] = np.where(env.reach, P[:, 1:], 0.0)
        tot = P[:, 1:].sum(1)
        over = tot > 1
        P[over, 1:] /= tot[over, None]
        P[:, 0] = 1 - P[:, 1:].sum(1)
        C = np.zeros((2, M + 1))
        C[0, 1:] = aS * np.maximum(s + b - 1, 0)
        C[1, 1:] = aS * s + aB * np.maximum(b - 1, 0)
        frac = np.zeros((2, len(groups)))
        split = np.zeros((2, len(groups), M))
        for kk in (0, 1):
            for gi in range(len(groups)):
                f = X[kk, gi].sum()
                frac[kk, gi] = min(f, 1.0)
                split[kk, gi] = X[kk, gi] / f if f > 1e-12 else groups[gi] / groups[gi].sum()
        return dict(P=P, C=C[env.k], s=s, b=b, frac=frac, split=split, gid=gid, J=J)

    def _fallback(self, agg):
        """No feasible plan: offload groups whose mean local latency misses, split evenly; else stay local."""
        var, x = [], []
        for ai, g in enumerate(agg):
            for j in g['rsus']:
                var.append((ai, j))
                x.append(1.0 / len(g['rsus']) if g['L'] > (12.0 if g['k'] == 0 else 40.0) else 0.0)
        return (np.inf, var, np.array(x))

    def _solve_cfg(self, agg, aS, aB, D, cfg, relax=False):
        var = [(ai, j) for ai, g in enumerate(agg) for j in g['rsus'] if not (cfg == 'B' and g['k'] == 0)]
        if not var:
            return None
        V = len(var)
        n = np.array([agg[ai]['n'] for ai, _ in var], float)
        isS = np.array([agg[ai]['k'] == 0 for ai, _ in var])
        jj = np.array([j for _, j in var])
        lin = np.array([agg[ai]['T'][j] - agg[ai]['L'] for ai, j in var]) * n
        A_s = np.zeros((M, V))
        A_b = np.zeros((M, V))
        A_s[jj[isS], np.flatnonzero(isS)] = n[isS]
        A_b[jj[~isS], np.flatnonzero(~isS)] = n[~isS]
        # Worst group transfer time per RSU and class (constraint must hold for every group that uses the RSU).
        Tmax = np.zeros((2, M))
        for ai, j in var:
            g = agg[ai]
            Tmax[g['k'], j] = max(Tmax[g['k'], j], g['T'][j])
        hasS = np.array([(A_s[j] > 0).any() for j in range(M)])
        hasB = np.array([(A_b[j] > 0).any() for j in range(M)])

        def loads(x):
            return A_s @ x, A_b @ x

        def obj(x):
            s, b = loads(x)
            q = aS * s ** 2 + 2 * aS * s * b + aB * b ** 2
            ds = 2 * aS * s + 2 * aS * b
            db = 2 * aS * s + 2 * aB * b
            return lin @ x + q.sum(), lin + A_s.T @ ds + A_b.T @ db

        z = self.z

        def margin(x):
            s, b = loads(x)
            nj = s + b
            var_j = (A_s + A_b) @ (x * (1 - x)) if z > 0 else np.zeros(M)
            nhi = nj + z * np.sqrt(np.maximum(var_j, 0))
            kf = np.where(nj > 1e-9, nhi / np.maximum(nj, 1e-9), 1.0)
            return s * kf, b * kf, nhi

        def dl_cons(x):
            s_hi, b_hi, nhi = margin(x)
            out = []
            for j in range(M):
                if hasS[j]:
                    out.append(D[0] - (Tmax[0, j] + aS * nhi[j]))
                if hasB[j]:
                    out.append(D[1] - (Tmax[1, j] + aS * s_hi[j] + aB * b_hi[j]))
            return np.array(out) if out else np.zeros(1)

        cons = [] if relax else [dict(type='ineq', fun=dl_cons)]
        # Per-group probability budget; equality when the group's mean local latency misses its deadline.
        for ai, g in enumerate(agg):
            idx = [v for v, (aj, _) in enumerate(var) if aj == ai]
            if not idx:
                if g['L'] > D[g['k']]:
                    return None              # this configuration cannot serve a group that must offload
                continue
            row = np.zeros(V)
            row[idx] = 1.0
            if g['L'] > D[g['k']]:
                cons.append(dict(type='eq', fun=lambda x, r=row: r @ x - 1.0, jac=lambda x, r=row: r))
            else:
                cons.append(dict(type='ineq', fun=lambda x, r=row: 1.0 - r @ x, jac=lambda x, r=row: -r))
        key = (cfg, tuple(var))
        x0 = self.warm.get(key)
        if x0 is None or len(x0) != V:
            x0 = np.full(V, 0.1)
        r = minimize(obj, x0, jac=True, method='SLSQP', bounds=[(0.0, 1.0)] * V, constraints=cons,
                     options=dict(maxiter=200, ftol=1e-9))
        if not r.success:
            r2 = minimize(obj, np.full(V, 0.05), jac=True, method='SLSQP', bounds=[(0.0, 1.0)] * V,
                          constraints=cons, options=dict(maxiter=400, ftol=1e-9))
            if not r2.success:
                return None
            r = r2
        x = np.clip(r.x, 0, 1)
        if not relax and dl_cons(x).min() < -1e-4:
            return None
        self.warm[key] = x
        const = sum(g['n'] * g['L'] for g in agg)
        return (float(r.fun + const), var, x)


# ----------------------------------------------------------------------
# Per-task layers on a per-vehicle plan
# ----------------------------------------------------------------------
def gipa_probs_general(env, t, PI, C, qbar_v, lam, beta):
    """GIPA operator anchored to a per-vehicle plan PI (same steps as hetero.gipa_plan_probs)."""
    q = -env.cost(t, predicted=True)
    D = env.dl[:, None]
    pred = -q + C
    m = np.isfinite(pred) & (pred * (1 + GUARD) <= D)
    m[~m.any(1), 0] = True
    pi = restrict(PI, m)
    p = pi.copy()
    qt = np.where(np.isfinite(q), q - qbar_v, -np.inf)
    rm = np.where(m, qt, -np.inf).max(1, keepdims=True)
    for r in range(1, CFG['K'] + 1):
        th = CFG['TH0'] * CFG['GA'] ** r
        arg = np.where(m, (qt - rm + th) / CFG['NU'], 0.0)
        g = np.where(m, np.exp(beta * sig(arg)), 0.0)
        p = p * g
        p = p / p.sum(1, keepdims=True)
        p = (1 - lam) * p + lam * pi
    return p


def benefit_population(sigma_h, sigma_r, coverage, n=20000):
    """Stage 0 synthetic benefit samples per class, drawn exactly as hetero.rank_thresholds draws them."""
    rg = np.random.default_rng(RNG_STAGE0)
    pops = []
    for k in (0, 1):
        c = C_MEAN[k] * lognorm1(rg, sigma_h, n)
        d = D_MEAN[k] * lognorm1(rg, sigma_h, n)
        g = lognorm1(rg, sigma_r, (n, M))
        e = d[:, None] / (RATE * g) + (c / F_GHZ)[:, None]
        if coverage < M:
            for i in range(n):
                e[i, rg.choice(M, M - coverage, replace=False)] = np.inf
        pops.append(np.sort(c - e.min(1)))
    return pops


def threshold(pop, frac, tie_tol=1e-9):
    """(tau, q_tie) such that a share ``frac`` of the population offloads (as hetero.rank_thresholds)."""
    if frac <= 0:
        return np.inf, 0.0
    t_k = float(np.quantile(pop, 1 - frac))
    above = np.mean(pop > t_k + tie_tol)
    tie = np.mean(np.abs(pop - t_k) <= tie_tol)
    return t_k, (float(np.clip((frac - above) / tie, 0, 1)) if tie > 0 else 0.0)


def rank_general(env, t, plan, pops, rg):
    """Per-task ranking within the plan's (class, group) offload fraction; RSU drawn from the plan's split.

    Consumes one uniform per vehicle, like hetero.plan_rank_actions.
    """
    e = env.cost(t, predicted=True)
    u = rg.random(env.N)
    ben = e[:, 0] - e[:, 1:].min(1)
    local_bad = e[:, 0] > env.dl
    gid = plan['gid']
    tk = np.empty(env.N)
    qk = np.empty(env.N)
    cache = {}
    for i in range(env.N):
        key = (env.k[i], gid[i])
        if key not in cache:
            cache[key] = threshold(pops[env.k[i]], plan['frac'][key])
        tk[i], qk[i] = cache[key]
    above = ben > tk + 1e-9
    tie = np.abs(ben - tk) <= 1e-9
    tie_go = tie & (u < qk)
    go = above | tie_go | local_bad
    v = np.where(tie_go & ~above & ~local_bad, u / np.maximum(qk, 1e-12), u)
    split = plan['split'][env.k, gid]                     # (N, M)
    split = np.where(env.reach, split, 0.0)
    tot = split.sum(1, keepdims=True)
    unif = env.reach / env.reach.sum(1, keepdims=True)
    split = np.where(tot > 1e-12, split / np.maximum(tot, 1e-300), unif)
    j = np.minimum((v[:, None] > np.cumsum(split, 1)).sum(1), M - 1) + 1
    return np.where(go, j, 0)
