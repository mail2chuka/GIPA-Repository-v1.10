"""
exp_tan.py  (revision experiment E15, 16 Nov 2026)
==================================================
GIPA's per-task layer against a Tan-style re-solved plan (see tan_style.py).

Conditions
----------
* The six E12 scenarios (homog, cov2, het25, het50, het50cov2, het50cov2p20),
  N = 80, 160, 240, 320, 30 seeds (0..29), with and without carried-over
  local work (set GIPA_LOCAL_CARRY=own before starting Python).
* Fluctuating fleet (batch only): HetEnv with 1.4 N_bar vehicles; every 10
  slots the active fleet is redrawn as round(U(0.6, 1.4) N_bar) (generator
  14000 + seed); the first N_t vehicles are active. Built on homog and het50.

Arms
----
local        all-local
plan         E12 static class plan (z = 2), sampled per class (with the own-queue
             redirect in own mode, exactly as hetero.run_het)
plan_rank    E12 per-task ranking within the static plan's fraction
gipa_plan    E12 GIPA operator on the static plan (lambda, beta tuned on 4600..4609)
tan          Tan-style plan, re-solved every slot, z = 0, sampled blindly
tan_z2       the same with GIPA's z = 2 margin, sampled blindly
tan_gipa     GIPA operator (same lambda, beta) on the Tan-style z = 0 plan
tan_rank     per-task ranking within the Tan-style z = 0 plan's fractions
Fluctuating fleet only:
plan_refresh   static plan re-planned every 20 slots for the current fleet (Stage 2), blind
gipa_refresh   GIPA operator on that refreshed plan

Usage
-----
python -I exp_tan.py --src /path/to/src --part batch|carry|fluct|tune --out file.json [--workers 2]
"""
import argparse
import json
import os
import sys
import time
from functools import lru_cache
from multiprocessing import Pool

import numpy as np

SCEN = {
    'homog': dict(sigma_h=0.0, sigma_r=0.0, coverage=3),
    'cov2': dict(sigma_h=0.0, sigma_r=0.0, coverage=2),
    'het25': dict(sigma_h=0.25, sigma_r=0.3, coverage=3),
    'het50': dict(sigma_h=0.5, sigma_r=0.3, coverage=3),
    'het50cov2': dict(sigma_h=0.5, sigma_r=0.3, coverage=2),
    'het50cov2p20': dict(sigma_h=0.5, sigma_r=0.3, coverage=2, sigma_p=0.2),
}
FLUCT = {'fluct_homog': 'homog', 'fluct_het50': 'het50'}
NS = (80, 160, 240, 320)
SEEDS = range(30)
TUNE_SEEDS = range(4600, 4610)
LAMS = (0.0, 0.1, 0.25, 0.5)
BETAS = (0.4, 1.0, 2.0, 4.0)
ARMS = ['local', 'plan', 'plan_rank', 'gipa_plan', 'tan', 'tan_z2', 'tan_gipa', 'tan_rank']
ARMS_FLUCT = ARMS + ['plan_refresh', 'gipa_refresh']
G = {}


def init(src):
    sys.path.insert(0, src)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import gipa_core
    import hetero
    import tan_style
    G.update(core=gipa_core, het=hetero, tan=tan_style)


@lru_cache(maxsize=None)
def class_plan_cached(N):
    return G['het'].class_plan(int(N), G['core'].REG['A'])


@lru_cache(maxsize=None)
def tau_cached(N, scen_key):
    sc = dict(scen_key)
    return G['het'].rank_thresholds(class_plan_cached(N), sc['sigma_h'], sc['sigma_r'], sc['coverage'], G['core'].REG['A'])


@lru_cache(maxsize=None)
def pops_cached(scen_key):
    sc = dict(scen_key)
    return G['tan'].benefit_population(sc['sigma_h'], sc['sigma_r'], sc['coverage'])


def sub_env(env, idx):
    het = G['het']
    s = object.__new__(het.HetEnv)
    s.N, s.R, s.M, s.n = len(idx), env.R, env.M, env.n
    s.isS, s.dl, s.k = env.isS[idx], env.dl[idx], env.k[idx]
    s.c, s.d, s.c_hat, s.d_hat = env.c[:, idx], env.d[:, idx], env.c_hat[:, idx], env.d_hat[:, idx]
    s.g, s.reach = env.g[idx], env.reach[idx]
    return s


def act(arm, env, t, rg, ctx):
    core, het, tan = G['core'], G['het'], G['tan']
    N = env.N
    own = core.LOCAL_CARRY == 'own'
    if arm == 'local':
        return np.zeros(N, int)
    if arm in ('plan', 'plan_refresh'):
        st = ctx['st_refresh'] if arm == 'plan_refresh' else ctx['st']
        P = het.plan_pmf(env, st)
        a = core.samp(P, rg)
        if own:
            u2 = rg.random(N)
            bad = (a == 0) & (env.cost(t, predicted=True)[:, 0] > env.dl)
            if bad.any():
                rs = P[:, 1:]
                tot = rs.sum(1, keepdims=True)
                unif = env.reach / env.reach.sum(1, keepdims=True)
                rs = np.where(tot > 0, rs / np.maximum(tot, 1e-300), unif)
                j = np.minimum((u2[:, None] > np.cumsum(rs, 1)).sum(1), rs.shape[1] - 1) + 1
                a = np.where(bad, j, a)
        return a
    if arm == 'plan_rank':
        return het.plan_rank_actions(env, t, ctx['st'], ctx['tau'], rg)
    if arm in ('gipa_plan', 'gipa_refresh'):
        st = ctx['st_refresh'] if arm == 'gipa_refresh' else ctx['st']
        return core.samp(het.gipa_plan_probs(env, t, st, ctx['lam'], True, ctx['beta']), rg)
    if arm in ('tan', 'tan_z2', 'tan_gipa', 'tan_rank'):
        plan = ctx['tan_plan'](arm, env, t)
        if arm in ('tan', 'tan_z2'):
            a = core.samp(plan['P'], rg)
            if own:
                rg.random(N)            # keep the uniform stream aligned with 'plan'
            return a
        if arm == 'tan_gipa':
            qbar_v = ctx['st']['qbar'][env.k]
            return core.samp(tan.gipa_probs_general(env, t, plan['P'], plan['C'], qbar_v, ctx['lam'], ctx['beta']), rg)
        return tan.rank_general(env, t, plan, ctx['pops'], rg)
    raise ValueError(arm)


def metrics_slot(out, env, a, cost, cnt):
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


def run_one(arm, scen_name, N, seed, lam, beta):
    """One episode of one arm on a fixed-fleet scenario."""
    core, het, tan = G['core'], G['het'], G['tan']
    sc = SCEN[scen_name]
    key = tuple(sorted(sc.items()))
    T = core.CFG['SLOTS']
    warm = core.CFG['WARM']
    env = het.HetEnv(N, seed, core.REG['A'], sc['sigma_h'], sc['sigma_r'], sc['coverage'], sc.get('sigma_p', 0.0), slots=T)
    rg = np.random.default_rng(core.RNG_ACT_BASE + seed)
    ctx = dict(lam=lam, beta=beta)
    if arm != 'local':
        ctx['st'] = class_plan_cached(N)
    if arm == 'plan_rank':
        ctx['tau'] = tau_cached(N, key)
    if arm == 'tan_rank':
        ctx['pops'] = pops_cached(key)
    planner = None
    if arm.startswith('tan'):
        planner = tan.TanPlanner(z=2.0 if arm == 'tan_z2' else 0.0, relax_fallback=os.environ.get('TAN_RELAX') == '1')
        ctx['tan_plan'] = lambda arm_, e, t: planner.solve(e, t)
    out = {k: [] for k in ('mS', 'mB', 'miss', 'lat', 'p95', 'off', 'offS', 'offB', 'maxl')}
    for t in range(T):
        a = act(arm, env, t, rg, ctx)
        cost, cnt = env.lat(a, t)
        if t >= warm:
            metrics_slot(out, env, a, cost, cnt)
    r = {k: core.safe_mean(v) for k, v in out.items()}
    if planner is not None:
        r['solve_fail'] = planner.fail / max(planner.solves, 1)
    return r


def schedule(Nbar, seed, T):
    rg = np.random.default_rng(14000 + seed)
    f = rg.uniform(0.6, 1.4, size=(T + 9) // 10)
    return np.array([int(round(f[t // 10] * Nbar)) for t in range(T)])


def run_fluct(arm, scen_name, Nbar, seed, lam, beta):
    """One episode on a fluctuating fleet (batch mode only)."""
    core, het, tan = G['core'], G['het'], G['tan']
    base = FLUCT[scen_name]
    sc = SCEN[base]
    key = tuple(sorted(sc.items()))
    T = core.CFG['SLOTS']
    warm = core.CFG['WARM']
    Nmax = int(round(1.4 * Nbar))
    env = het.HetEnv(Nmax, seed, core.REG['A'], sc['sigma_h'], sc['sigma_r'], sc['coverage'], sc.get('sigma_p', 0.0), slots=T)
    rg = np.random.default_rng(core.RNG_ACT_BASE + seed)
    Nt = schedule(Nbar, seed, T)
    ctx = dict(lam=lam, beta=beta)
    if arm != 'local':
        ctx['st'] = class_plan_cached(Nbar)
        ctx['tau'] = tau_cached(Nbar, key)
    if arm == 'tan_rank':
        ctx['pops'] = pops_cached(key)
    planner = None
    if arm.startswith('tan'):
        planner = tan.TanPlanner(z=2.0 if arm == 'tan_z2' else 0.0, relax_fallback=os.environ.get('TAN_RELAX') == '1')
        ctx['tan_plan'] = lambda arm_, e, t: planner.solve(e, t)
    out = {k: [] for k in ('mS', 'mB', 'miss', 'lat', 'p95', 'off', 'offS', 'offB', 'maxl')}
    for t in range(T):
        if t % 20 == 0:
            ctx['st_refresh'] = class_plan_cached(int(Nt[t]))       # Stage 2: re-plan for the fleet seen now
        e = sub_env(env, np.arange(Nt[t]))
        a = act(arm, e, t, rg, ctx)
        cost, cnt = e.lat(a, t)
        if t >= warm:
            metrics_slot(out, e, a, cost, cnt)
    r = {k: core.safe_mean(v) for k, v in out.items()}
    if planner is not None:
        r['solve_fail'] = planner.fail / max(planner.solves, 1)
    return r


def job(args):
    kind, arm, scen, N, seed, lam, beta = args
    t0 = time.time()
    r = (run_fluct if kind == 'fluct' else run_one)(arm, scen, N, seed, lam, beta)
    r['sec'] = time.time() - t0
    return (kind, arm, scen, N, seed, lam, beta), r


def boot_ci(d, rg, B=10000):
    d = np.asarray(d, float)
    if np.allclose(d, d[0]):
        return [float(d[0]), float(d[0])]
    idx = rg.integers(0, len(d), size=(B, len(d)))
    m = d[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--part', required=True, choices=['tune', 'batch', 'carry', 'fluct'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--chosen', default=None, help='JSON with tuned lam/beta per scen|N (batch tuning output or carry_own E12 file)')
    ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    init(a.src)
    core = G['core']
    if a.part == 'carry' and core.LOCAL_CARRY != 'own':
        raise SystemExit('set GIPA_LOCAL_CARRY=own for --part carry')
    if a.part != 'carry' and core.LOCAL_CARRY:
        raise SystemExit('unset GIPA_LOCAL_CARRY for this part')

    t0 = time.time()
    if a.part == 'tune':
        jobs = [('fixed', 'gipa_plan', sc, N, s, lam, beta) for sc in SCEN for N in NS
                for lam in LAMS for beta in BETAS for s in TUNE_SEEDS]
        with Pool(a.workers, initializer=init, initargs=(a.src,)) as pool:
            res = pool.map(job, jobs, chunksize=8)
        table = {}
        for (kind, arm, sc, N, s, lam, beta), r in res:
            table.setdefault((sc, N, lam, beta), []).append(r)
        tuning, chosen = [], {}
        for (sc, N, lam, beta), rs in sorted(table.items()):
            lat = float(np.mean([r['lat'] for r in rs]))
            miss = float(np.mean([r['miss'] for r in rs]))
            tuning.append(dict(scen=sc, N=N, lam=lam, beta=beta, lat=lat, miss=miss))
        for sc in SCEN:
            for N in NS:
                rows = [r for r in tuning if r['scen'] == sc and r['N'] == N]
                # Selection rule of Section VI-E: lowest mean latency among settings whose miss rate is
                # within 0.5 percentage points of the lowest.
                mmin = min(r['miss'] for r in rows)
                best = min([r for r in rows if r['miss'] <= mmin + 0.005], key=lambda r: r['lat'])
                chosen[f'{sc}|{N}|gipa_plan'] = dict(lam=best['lam'], beta=best['beta'])
        json.dump(dict(rule='lowest mean latency among settings whose miss rate is within 0.5 pp of the lowest, '
                            'tuning seeds 4600-4609', tuning=tuning, chosen=chosen,
                       seconds=time.time() - t0), open(a.out, 'w'), indent=1)
        print('tuning done', round(time.time() - t0), 's')
        return

    chosen = json.load(open(a.chosen))['chosen']
    jobs = []
    if a.part in ('batch', 'carry'):
        for sc in SCEN:
            for N in NS:
                c = chosen[f'{sc}|{N}|gipa_plan']
                for arm in ARMS:
                    for s in SEEDS:
                        jobs.append(('fixed', arm, sc, N, s, c['lam'], c['beta']))
    else:
        for fl, base in FLUCT.items():
            for N in NS:
                c = chosen[f'{base}|{N}|gipa_plan']
                for arm in ARMS_FLUCT:
                    for s in SEEDS:
                        jobs.append(('fluct', arm, fl, N, s, c['lam'], c['beta']))
    # Longest jobs first for better load balance.
    jobs.sort(key=lambda j: (not j[1].startswith('tan'), -j[3]))
    with Pool(a.workers, initializer=init, initargs=(a.src,)) as pool:
        res = pool.map(job, jobs, chunksize=4)
    per_seed = {}
    for (kind, arm, sc, N, s, lam, beta), r in res:
        per_seed.setdefault(f'{sc}|{N}|{arm}', {})[s] = r
    per_seed = {k: [v[s] for s in sorted(v)] for k, v in per_seed.items()}
    main_t, paired = {}, {}
    rgb = np.random.default_rng(20260804)
    arms = ARMS_FLUCT if a.part == 'fluct' else ARMS
    scens = FLUCT if a.part == 'fluct' else SCEN
    for sc in scens:
        for N in NS:
            row = {}
            for arm in arms:
                rs = per_seed[f'{sc}|{N}|{arm}']
                row[arm] = {k: core.safe_mean([r[k] for r in rs]) for k in rs[0]}
            main_t[f'{sc}|{N}'] = row
            pr = {}
            for arm in arms:
                if arm == 'tan':
                    continue
                for metric in ('lat', 'miss'):
                    d = [x[metric] - y[metric] for x, y in zip(per_seed[f'{sc}|{N}|{arm}'], per_seed[f'{sc}|{N}|tan'])]
                    pr[f'{arm}-tan|{metric}'] = dict(mean=float(np.mean(d)), ci=boot_ci(d, rgb))
            for x, y in (('tan_gipa', 'gipa_plan'), ('tan_rank', 'plan_rank'), ('plan_rank', 'gipa_plan')):
                for metric in ('lat', 'miss'):
                    d = [p[metric] - q[metric] for p, q in zip(per_seed[f'{sc}|{N}|{x}'], per_seed[f'{sc}|{N}|{y}'])]
                    pr[f'{x}-{y}|{metric}'] = dict(mean=float(np.mean(d)), ci=boot_ci(d, rgb))
            paired[f'{sc}|{N}'] = pr
    json.dump(dict(part=a.part, carry=core.LOCAL_CARRY or 'none', chosen=chosen, main=main_t, paired=paired,
                   per_seed=per_seed, seconds=time.time() - t0,
                   seeds=dict(evaluation='0..29', actions='5000+s', fleet='14000+s')),
              open(a.out, 'w'), indent=1)
    print(a.part, 'done', round(time.time() - t0), 's')


if __name__ == '__main__':
    main()
