"""
exp_qlearning.py
================
Workplan experiment E4: Q-learning baseline reliability (Reviewer 2,
comment 4). See src/qlearning.py for the learner and the fixes.

STATUS: written for the first revision of manuscript ITS-2026-08-0579. Not
part of the submitted v1.0.0 results.

What it runs
------------
1. Isolating the fixes (Regime A, affine model, N = 160):
     archived   training seed 5999, training environments 0..6 (overlapping
                evaluation), no episode reset: must reproduce res_main.json
     fixed      the same training seed with disjoint training environments
                and the episode reset
2. Main grid, for regimes A and B and latency models 'affine' and 'ps'
   (Eq. 5), with 10 independent training runs (exploration seeds
   6000-6009, training environments 3000-3006):
     gamma0_N160        gamma = 0, trained at N = 160, evaluated at
                        N = 80, 160, 240 and 320 (the archived protocol with
                        the fixes)
     gamma0_perdensity  gamma = 0, trained separately at N = 80, 240 and 320
                        and evaluated at its own density (N = 160 is the
                        row above)
     gamma09_N160       gamma = 0.9 (bootstrapped target), trained at
                        N = 160, evaluated at every density
   Every evaluation uses the 30 archived seeds (environment 1000-1029,
   action sampling 5000-5029) with the episode reset.

Statistics
----------
For each configuration and evaluation density, a 10 x 30 matrix (training
runs x evaluation seeds) per metric. Reported: grand mean, two-level
bootstrap 95% CI (10,000 replicates, seed 20260804), and the standard
deviation of the 10 per-run means (between-run variability). Convergence is
summarised by the largest Q-value change over the final 500 training steps
and by the range of slot latency over the final 20% of training.

Output
------
results/res_qlearning.json (path configurable with --out). Contents:
"fixes" (part 1), "grid" (configuration -> evaluation density -> metric ->
{mean, ci95, sd_between_runs}), "runs" (per training run: Q-table, greedy
policy, learning curve, per-density evaluation means and per-seed records),
and "archived_reference" (the archived Q-learning values from
results/res_main.json).

Runtime
-------
About 5-15 minutes on a laptop CPU (two worker processes).

Run (from the repository root)
------------------------------
    PYTHONPATH=src python experiments/exp_qlearning.py
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from gipa_core import CFG, REG
from qlearning import QLearner, evaluate, two_level_bootstrap

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SEEDS = list(range(6000, 6010))
DENSITIES = [80, 160, 240, 320]
METRICS = ['lat', 'p95', 'mS', 'mB', 'off', 'offS', 'offB', 'maxl']


def _clean(d):
    """Replace NaN with None so the output is strict JSON."""
    return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in d.items()}


def job(spec):
    """Train one learner and evaluate it; runs in a worker process."""
    reg, model, gamma, N_train, seed, eval_Ns = spec
    R = REG[reg]
    ql = QLearner(CFG['M'] + 1, train_seed=seed, gamma=gamma, model=model)
    ql.train(N_train, R)
    ev = {}
    for N in eval_Ns:
        m, rs = evaluate(ql, N, R, reset=True, model=model)
        ev[N] = dict(mean=_clean(m), per_seed=[_clean(r) for r in rs])
    tail = ql.curve[-max(1, len(ql.curve) // 5):]
    lat_tail = [c['lat'] for c in tail]
    return dict(regime=reg, model=model, gamma=gamma, N_train=N_train, train_seed=seed,
                Q=ql.Q.tolist(), greedy_policy=ql.greedy_policy(), curve=ql.curve,
                final_max_dQ=ql.curve[-1]['max_dQ'],
                tail_lat_range=[float(min(lat_tail)), float(max(lat_tail))], eval=ev)


def fixes_part():
    """Part 1: reproduce the archived value, then apply the two fixes to the same seed."""
    R = REG['A']
    out = {}
    archived = QLearner(CFG['M'] + 1, train_seed=5999, env_offset=0, gamma=0.0, model='affine')
    archived.train(160, R, log_every=0)
    # The archived protocol evaluated the learner at N = 80 before N = 160, with the
    # load state carried over; replicate that order to reproduce the archived value.
    evaluate(archived, 80, R, reset=False)
    m, _ = evaluate(archived, 160, R, reset=False)
    out['archived'] = _clean(m)
    fixed = QLearner(CFG['M'] + 1, train_seed=5999, gamma=0.0, model='affine')
    fixed.train(160, R, log_every=0)
    m, _ = evaluate(fixed, 160, R, reset=True)
    out['fixed_same_seed'] = _clean(m)
    return out


def main():
    """Run parts 1 and 2 and write the JSON output."""
    ap = argparse.ArgumentParser(description='E4 Q-learning reliability')
    ap.add_argument('--out', type=Path, default=ROOT / 'results' / 'res_qlearning.json')
    ap.add_argument('--workers', type=int, default=2)
    args = ap.parse_args()

    fixes = fixes_part()
    ref = json.load(open(ROOT / 'results' / 'res_main.json'))
    print(f"1. archived protocol at N = 160: lat {fixes['archived']['lat']:.4f} ms "
          f"(res_main.json: {ref['A']['160']['ql']['lat']:.4f}); with fixes, same seed: "
          f"lat {fixes['fixed_same_seed']['lat']:.4f} ms", flush=True)

    specs = []
    for reg in ['A', 'B']:
        for model in ['affine', 'ps']:
            for s in TRAIN_SEEDS:
                specs.append((reg, model, 0.0, 160, s, DENSITIES))
                specs.append((reg, model, 0.9, 160, s, DENSITIES))
                for N in [80, 240, 320]:
                    specs.append((reg, model, 0.0, N, s, [N]))
    with Pool(args.workers) as pool:
        runs = pool.map(job, specs, chunksize=1)

    grid = {}
    for reg in ['A', 'B']:
        for model in ['affine', 'ps']:
            for name, sel in [('gamma0_N160', lambda r: r['gamma'] == 0.0 and r['N_train'] == 160),
                              ('gamma09_N160', lambda r: r['gamma'] == 0.9),
                              ('gamma0_perdensity', lambda r: r['gamma'] == 0.0)]:
                key = f'{reg}|{model}|{name}'
                grid[key] = {}
                rs = [r for r in runs if r['regime'] == reg and r['model'] == model and sel(r)]
                for N in DENSITIES:
                    if name == 'gamma0_perdensity':
                        cand = [r for r in rs if r['N_train'] == N]
                    else:
                        cand = rs
                    cand = [r for r in cand if N in r['eval'] or str(N) in r['eval']]
                    if not cand:
                        continue
                    stats = {}
                    for met in METRICS:
                        Mx = np.array([[np.nan if x[met] is None else x[met] for x in r['eval'][N]['per_seed']]
                                       for r in cand])
                        mean, lo, hi = two_level_bootstrap(Mx)
                        sd = float(np.nanstd(np.nanmean(Mx, axis=1), ddof=1)) if len(cand) > 1 else 0.0
                        stats[met] = dict(mean=mean, ci95=[lo, hi], sd_between_runs=sd)
                    stats['n_runs'] = len(cand)
                    stats['final_max_dQ_median'] = float(np.median([r['final_max_dQ'] for r in cand
                                                                    if r['N_train'] == (160 if name != 'gamma0_perdensity' else N)]))
                    grid[key][N] = stats
                    l = stats['lat']
                    print(f"2. {key:28s} N={N:3d}: lat {l['mean']:6.2f} [{l['ci95'][0]:.2f}, {l['ci95'][1]:.2f}] "
                          f"sd {l['sd_between_runs']:.3f} | miss S/B {100 * stats['mS']['mean']:5.1f}/{100 * stats['mB']['mean']:4.1f}% "
                          f"| offB {stats['offB']['mean']:.3f} | median final max dQ {stats['final_max_dQ_median']:.2f}", flush=True)

    archived_ref = {reg: {N: ref[reg][str(N)]['ql'] for N in DENSITIES} for reg in ['A', 'B']}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(dict(fixes=fixes, grid=grid, runs=runs, archived_reference=archived_ref,
                       train_seeds=TRAIN_SEEDS), fh)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
