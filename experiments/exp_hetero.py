"""
exp_hetero.py
=============
Workplan experiment E12: heterogeneous tasks, uneven links and partial
coverage. Tests decision D5's "Framing A" (GIPA as the per-task executor of a
planned allocation) against "Framing B" (the planner, sampled per class, or
with a deterministic per-task ranking). See src/hetero.py for the model and
policies.

STATUS: written for the first revision of manuscript ITS-2026-08-0579. Not
part of the submitted v1.0.0 results.

What it runs
------------
0. Check: with no heterogeneity and full coverage, 'plan' and 'gipa'
   reproduce E2's per-seed latencies exactly.
1. Scenarios (Regime A, E2 physical calibration):
     homog        identical tasks, full coverage (sanity: all plan-based
                  policies coincide)
     cov2         identical tasks, each vehicle reaches 2 of 3 RSUs
     het25        task compute and data size lognormal (sigma 0.25), link
                  rates lognormal (sigma 0.3), full coverage
     het50        sigma 0.5 for tasks, 0.3 for links
     het50cov2    het50 plus 2-of-3 coverage
     het50cov2p20 het50cov2 with predicted features (multiplicative
                  lognormal error, sigma 0.2)
   N = 80, 160, 240, 320.
2. Tuning on seed indices 4600-4609 (never used for evaluation): GIPA on the
   plan, relative and absolute variants, lambda in {0, 0.1, 0.25, 0.5} x beta
   in {0.4, 1, 2, 4}, per scenario and density. Rule: overall miss within 0.5
   points of the lowest, then lowest latency.
3. Evaluation on the 30 archived seeds: local, GIPA as submitted, plan,
   plan_mask, plan_rank, gipa_plan (tuned), gipa_plan (Table II defaults
   lambda 0.5, beta 0.4), gipa_plan_abs (tuned), learning automata (500
   warm-up slots). Paired bootstrap CIs (10,000 resamples, seed 20260804):
   gipa_plan - plan_rank, gipa_plan - plan_mask (what amplification adds),
   plan_rank - plan (what per-task ranking adds), gipa_plan - gipa.

Output
------
results/res_hetero.json: "check", "scenarios", "tuning", "chosen", "main",
"paired", "per_seed".

Runtime
-------
About 10-20 minutes on a laptop CPU with two worker processes.

Run (from the repository root)
------------------------------
    PYTHONPATH=src python experiments/exp_hetero.py
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from gipa_core import CFG, REG
from hetero import check_homogeneous, run_het
from stats import Bootstrap

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = {
    'homog': dict(sigma_h=0.0, sigma_r=0.0, coverage=3),
    'cov2': dict(sigma_h=0.0, sigma_r=0.0, coverage=2),
    'het25': dict(sigma_h=0.25, sigma_r=0.3, coverage=3),
    'het50': dict(sigma_h=0.5, sigma_r=0.3, coverage=3),
    'het50cov2': dict(sigma_h=0.5, sigma_r=0.3, coverage=2),
    'het50cov2p20': dict(sigma_h=0.5, sigma_r=0.3, coverage=2, sigma_p=0.2),
}
DENSITIES = [80, 160, 240, 320]
LAMS = [0.0, 0.1, 0.25, 0.5]
BETAS = [0.4, 1.0, 2.0, 4.0]
TUNE_SEEDS = list(range(4600, 4610))
MISS_TOL = 0.005


def _clean(d):
    return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in d.items()}


def job_tune(spec):
    scen, N, pol, lam, beta = spec
    rs = [run_het(pol, N, s, REG['A'], SCENARIOS[scen], lam=lam, beta=beta) for s in TUNE_SEEDS]
    return dict(scen=scen, N=N, pol=pol, lam=lam, beta=beta,
                lat=float(np.mean([r['lat'] for r in rs])), miss=float(np.mean([r['miss'] for r in rs])))


def job_eval(spec):
    scen, N, name, pol, kw = spec
    rs = [_clean(run_het(pol, N, s, REG['A'], SCENARIOS[scen], **kw)) for s in range(CFG['NSEED'])]
    return (scen, N, name), rs


def choose(rows):
    best = min(r['miss'] for r in rows)
    return min([r for r in rows if r['miss'] <= best + MISS_TOL], key=lambda r: r['lat'])


def main():
    ap = argparse.ArgumentParser(description='E12 heterogeneous tasks')
    ap.add_argument('--out', type=Path, default=ROOT / 'results' / 'res_hetero.json')
    ap.add_argument('--workers', type=int, default=2)
    args = ap.parse_args()

    chk = check_homogeneous()
    print(f'0. homogeneous check against E2: max |diff| {chk:.3g} ms', flush=True)

    specs = [(sc, N, pol, lam, beta) for sc in SCENARIOS for N in DENSITIES
             for pol in ['gipa_plan', 'gipa_plan_abs'] for lam in LAMS for beta in BETAS]
    with Pool(args.workers) as pool:
        tuning = pool.map(job_tune, specs, chunksize=8)
    chosen = {}
    for sc in SCENARIOS:
        for N in DENSITIES:
            for pol in ['gipa_plan', 'gipa_plan_abs']:
                c = choose([r for r in tuning if r['scen'] == sc and r['N'] == N and r['pol'] == pol])
                chosen[f'{sc}|{N}|{pol}'] = dict(lam=c['lam'], beta=c['beta'])
    print('2. tuned (lambda, beta) for gipa_plan: ' + '; '.join(
        f"{sc} " + ' '.join(f"N{N}:{chosen[f'{sc}|{N}|gipa_plan']['lam']}/{chosen[f'{sc}|{N}|gipa_plan']['beta']}" for N in DENSITIES)
        for sc in SCENARIOS), flush=True)

    la_rate = json.load(open(ROOT / 'results' / 'res_la.json'))['chosen']['A|ps|500']['lri']['rate']
    la_cfg = dict(algo='lri', resp='binary', rate=la_rate)
    ev = []
    for sc in SCENARIOS:
        for N in DENSITIES:
            ev += [(sc, N, 'local', 'local', {}), (sc, N, 'gipa', 'gipa', {}), (sc, N, 'plan', 'plan', {}),
                   (sc, N, 'plan_mask', 'plan_mask', {}), (sc, N, 'plan_rank', 'plan_rank', {}),
                   (sc, N, 'gipa_plan', 'gipa_plan', chosen[f'{sc}|{N}|gipa_plan']),
                   (sc, N, 'gipa_plan_default', 'gipa_plan', dict(lam=0.5, beta=0.4)),
                   (sc, N, 'gipa_plan_abs', 'gipa_plan_abs', chosen[f'{sc}|{N}|gipa_plan_abs']),
                   (sc, N, 'la', 'la', dict(la_cfg=la_cfg, warm=500))]
    with Pool(args.workers) as pool:
        res = dict(pool.map(job_eval, ev, chunksize=2))

    main_tab, paired, per_seed = {}, {}, {}
    for sc in SCENARIOS:
        main_tab[sc], paired[sc] = {}, {}
        for N in DENSITIES:
            bs = Bootstrap()
            row = {}
            for (s2, n2, name), rs in res.items():
                if s2 != sc or n2 != N:
                    continue
                row[name] = {k: (float(np.nanmean([r[k] for r in rs if r[k] is not None]))
                                 if any(r[k] is not None for r in rs) else None) for k in rs[0]}
                row[name]['lat_ci95'] = bs.ci([r['lat'] for r in rs])[1]
                per_seed[f'{sc}|{N}|{name}'] = [dict(lat=r['lat'], miss=r['miss']) for r in rs]
            main_tab[sc][N] = row
            pr = {}
            for a, b in [('gipa_plan', 'plan_rank'), ('gipa_plan', 'plan_mask'), ('plan_rank', 'plan'),
                         ('gipa_plan', 'gipa'), ('plan_rank', 'la')]:
                xa = [r['lat'] for r in res[(sc, N, a)]]
                xb = [r['lat'] for r in res[(sc, N, b)]]
                ma = [r['miss'] for r in res[(sc, N, a)]]
                mb = [r['miss'] for r in res[(sc, N, b)]]
                pr[f'{a} - {b}'] = dict(lat=bs.paired(xa, xb), miss=bs.paired(ma, mb))
            paired[sc][N] = pr
            f = lambda n: f"{row[n]['lat']:6.2f}/{100 * row[n]['miss']:4.1f}%"
            print(f"3. {sc:13s} N={N:3d} local {f('local')} gipa {f('gipa')} plan {f('plan')} plan_mask {f('plan_mask')} "
                  f"plan_rank {f('plan_rank')} gipa_plan {f('gipa_plan')} (default {f('gipa_plan_default')}, abs {f('gipa_plan_abs')}) "
                  f"LA {f('la')} | gipa_plan-plan_rank {pr['gipa_plan - plan_rank']['lat']['mean_difference']:+.2f} "
                  f"[{pr['gipa_plan - plan_rank']['lat']['ci95'][0]:+.2f}, {pr['gipa_plan - plan_rank']['lat']['ci95'][1]:+.2f}]",
                  flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(dict(check=dict(homogeneous_max_abs_diff_ms=chk), scenarios=SCENARIOS, tuning=tuning,
                       chosen=chosen, main=main_tab, paired=paired, per_seed=per_seed,
                       seeds=dict(evaluation='0..29', tuning=f'{TUNE_SEEDS[0]}..{TUNE_SEEDS[-1]}',
                                  features='12000+s', links='12500+s', prediction='13000+s', stage0=13500)), fh)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
