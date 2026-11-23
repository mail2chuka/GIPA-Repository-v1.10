"""
exp_async.py
============
Workplan experiment E5: asynchronous task arrivals with unfinished work
carried forward (Reviewer 1 comment 6). See src/async_sim.py for the model,
the policies and the information each one uses.

STATUS: written for the first revision of manuscript ITS-2026-08-0579. Not
part of the submitted v1.0.0 results.

What it runs
------------
0. Checks: the rebuilt Table III processor-sharing check (5 ms tasks over
   the first quarter of a 20 ms slot, 4,000 replications, seed 0); exact
   agreement with the E2 batch formula for simultaneous arrivals; the
   M/M/1-PS mean sojourn; and the per-class GIPA PMF against
   calibration.gipa_pv.
1. Regimes A and B under the E2 (Eq. 5) calibration, 20 s simulated per run,
   30 seeds, fleets N = 55, 80, 87, 104, 160 and 240. N = 80, 160, 240 match
   the synchronous densities (N tasks per 20 ms on average); N = 55, 87 and
   104 give an all-offload RSU utilisation of 0.5, 0.8 and 0.95
   (rho = N / 20 * a_mix / M).
   Policies: local, greedy, indep, GIPA with the slot prior, GIPA with the
   backlog-aware priors (Stage 0 heuristic and planned utilisation), GIPA
   without diffusion, prior-adjusted masked softmax with the planned prior,
   the asynchronous planner, RBR and GIPA on a stale backlog broadcast
   (20 and 100 ms), a near-live shortest-expected-delay reference (1 ms),
   and learning automata (E6 tuned L_R-I). Admission-control (-AC) variants
   for greedy, indep, GIPA (slot prior), the planner and RBR (20 ms).
2. Summaries over seeds: mean of each metric, share of unstable runs, and
   the median latency (robust when some runs are unstable). Paired
   differences (seed-level bootstrap, 10,000 resamples, seed 20260804) for
   GIPA (slot prior) - planner, learning automata - planner,
   near-live reference - planner and GIPA (slot prior) - all-local.

Output
------
results/res_async.json (path configurable with --out): "checks", "plans",
"main" (regime -> N -> policy -> summary), "paired", "per_seed".

Runtime
-------
About 20-40 minutes on a laptop CPU with two worker processes.

Run (from the repository root)
------------------------------
    PYTHONPATH=src python experiments/exp_async.py
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from async_sim import (BROADCAST, STATIC, check_batch, check_gipa_probs, check_mm1, check_table3,
                       plan_async, run_async)
from calibration import make_calibration
from gipa_core import CFG, REG

ROOT = Path(__file__).resolve().parents[1]
FLEETS = [55, 80, 87, 104, 160, 240]
SEEDS = list(range(CFG['NSEED']))
AC_POLICIES = ['greedy', 'indep', 'gipa_slot', 'plan', 'rbr_b20']
BOOT_REPS, BOOT_SEED = 10000, 20260804
TABLE3_PAPER = {5: 21.64, 10: 45.90, 20: 95.48, 40: 195.25}


def policies():
    return STATIC + list(BROADCAST) + ['la'] + [p + '-AC' for p in AC_POLICIES]


def job(spec):
    reg, N, pol, seed, plan, la_cfg = spec
    base = pol[:-3] if pol.endswith('-AC') else pol
    m = run_async(base, N, seed, REG[reg], make_calibration(), ac=pol.endswith('-AC'), plan=plan, la_cfg=la_cfg)
    return (reg, N, pol, seed), m


def paired_ci(x, y):
    d = np.asarray(x, float) - np.asarray(y, float)
    rg = np.random.default_rng(BOOT_SEED)
    bs = d[rg.integers(0, len(d), (BOOT_REPS, len(d)))].mean(1)
    return dict(mean=float(d.mean()), ci95=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))])


def summarise(rs):
    out = {}
    for k in rs[0]:
        vals = [r[k] for r in rs]
        if isinstance(vals[0], bool):
            out[k + '_share'] = float(np.mean(vals))
        elif isinstance(vals[0], list):
            out[k] = np.mean(np.array(vals, float), 0).tolist()
        elif all(v is None for v in vals):
            out[k] = None
        else:
            out[k] = float(np.mean([v for v in vals if v is not None]))
    out['lat_median'] = float(np.median([r['lat'] for r in rs]))
    return out


def main():
    ap = argparse.ArgumentParser(description='E5 asynchronous arrivals')
    ap.add_argument('--out', type=Path, default=ROOT / 'results' / 'res_async.json')
    ap.add_argument('--workers', type=int, default=2)
    args = ap.parse_args()

    t3 = check_table3()
    bt = check_batch()
    mm = check_mm1()
    gp = check_gipa_probs()
    checks = dict(table3={c: dict(closed_form=v[0], des=v[1], paper=TABLE3_PAPER[c]) for c, v in t3.items()},
                  batch_formula=dict(passed=bool(bt[0]), max_abs_diff=float(bt[1])),
                  mm1_ps=dict(theory=mm[0], simulated=mm[1]), gipa_probs=gp)
    print('0. Table III: ' + ', '.join(f'{c}: {v[1]:.2f} ms (paper {TABLE3_PAPER[c]:.2f})' for c, v in t3.items()), flush=True)
    print(f'   batch formula {bt[0]} (max diff {bt[1]:.1e}); M/M/1-PS {mm[1]:.3f} vs {mm[0]:.3f}; GIPA PMF {gp}', flush=True)

    cal = make_calibration()
    plans = {reg: {N: plan_async(N, REG[reg], cal) for N in FLEETS} for reg in ['A', 'B']}
    for reg in ['A', 'B']:
        for N in FLEETS:
            p = plans[reg][N]
            print(f'   plan {reg} N={N}: ' + ('infeasible' if p is None else
                  f"x_S {p['x_S']:.3f}, x_B {p['x_B']:.3f}, rho {p['rho']:.3f}, expected latency {p['J']:.2f} ms"), flush=True)
    la_chosen = json.load(open(ROOT / 'results' / 'res_la.json'))['chosen']
    specs = []
    for reg in ['A', 'B']:
        c = la_chosen[f'{reg}|ps|500']['lri']
        la_cfg = dict(algo='lri', resp='binary', rate=c['rate'])
        for N in FLEETS:
            for pol in policies():
                base = pol[:-3] if pol.endswith('-AC') else pol
                if base == 'plan' and plans[reg][N] is None:
                    continue
                for s in SEEDS:
                    specs.append((reg, N, pol, s, plans[reg][N], la_cfg))
    with Pool(args.workers) as pool:
        res = dict(pool.map(job, specs, chunksize=4))

    main_tab, paired, per_seed = {}, {}, {}
    for reg in ['A', 'B']:
        main_tab[reg], paired[reg] = {}, {}
        for N in FLEETS:
            main_tab[reg][N] = {}
            for pol in policies():
                rs = [res[(reg, N, pol, s)] for s in SEEDS if (reg, N, pol, s) in res]
                if not rs:
                    continue
                main_tab[reg][N][pol] = summarise(rs)
                per_seed[f'{reg}|{N}|{pol}'] = [dict(lat=r['lat'], miss=r['miss'], unstable=r['unstable']) for r in rs]

            def lat(p):
                return [res[(reg, N, p, s)]['lat'] for s in SEEDS] if (reg, N, p, 0) in res else None
            pr = {}
            for a, b in [('gipa_slot', 'plan'), ('la', 'plan'), ('jsq_live', 'plan'), ('gipa_slot', 'local')]:
                if lat(a) and lat(b):
                    pr[f'{a} - {b}'] = paired_ci(lat(a), lat(b))
            paired[reg][N] = pr
            t = main_tab[reg][N]

            def f(p):
                v = t.get(p)
                if v is None:
                    return '   n/a'
                s = f"{v['lat_median']:7.2f}"
                if v['unstable_share'] > 0:
                    s += f"(U{v['unstable_share']:.0%})"
                return s
            print(f"1. {reg} N={N:3d} median lat: local {f('local')} greedy {f('greedy')} GIPA {f('gipa_slot')} "
                  f"GIPA-soj {f('gipa_soj_heur')}/{f('gipa_soj_plan')} plan {f('plan')} LA {f('la')} "
                  f"RBR20/100 {f('rbr_b20')}/{f('rbr_b100')} GIPA-b20/100 {f('gipa_b20')}/{f('gipa_b100')} "
                  f"JSQ {f('jsq_live')} | miss% GIPA {100 * t['gipa_slot']['miss']:.1f} plan "
                  f"{100 * t['plan']['miss'] if 'plan' in t else float('nan'):.1f} LA {100 * t['la']['miss']:.1f} "
                  f"JSQ {100 * t['jsq_live']['miss']:.1f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(dict(checks=checks, fleets=FLEETS, plans=plans, main=main_tab, paired=paired,
                       per_seed=per_seed, horizon_ms=20000.0, seeds=SEEDS), fh)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
