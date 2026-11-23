"""
exp_asym.py
===========
Workplan experiment E9: anti-herding under asymmetric RSUs (the core claim).
See src/asymmetric.py for the environments, priors, planner and herding
metrics.

STATUS: written for the first revision of manuscript ITS-2026-08-0579. Not
part of the submitted v1.0.0 results.

What it runs
------------
0. Checks: with sp = (1, 1, 1) and the 'uniform' prior, every policy
   reproduces the archived (affine) and E2 (Eq. 5) latencies exactly; the
   asymmetric planner with sp = (1, 1, 1) reproduces the grid planner's
   objective (equal or lower).
1. For scenarios 'sym' (1, 1, 1) and 'fast1' (0.7, 1, 1), regimes A and B,
   latency models 'affine' and 'ps', N = 80, 160, 240, 320, on the 30
   archived seeds:
     local, greedy, indep (no prior), RBR (per-slot load feedback),
     masked softmax, prior-adjusted masked softmax, entropy-regularised
     softmax, GIPA and GIPA without diffusion, each with the 'uniform'
     (archived Eq. 13) and 'capacity' (recalibrated Eq. 12) priors,
     the asymmetric planner (z = 2), and learning automata (the E6 'best'
     configuration for that regime and model, 500 warm-up slots).
2. Paired differences (seed-level bootstrap, 10,000 resamples, seed
   20260804): GIPA - planner, GIPA - GIPA without diffusion, and
   LA - planner, for latency; GIPA - GIPA without diffusion for the
   fast-RSU herding ratio.

Output
------
results/res_asym.json (path configurable with --out): "checks", "plans",
"main" (scenario|regime|model -> N -> policy -> metric means),
"paired", "per_seed" (latency and herding ratio per seed).

Runtime
-------
About 5-10 minutes on a laptop CPU with two worker processes.

Run (from the repository root)
------------------------------
    PYTHONPATH=src python experiments/exp_asym.py
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from asymmetric import PRIOR_POLICIES, SCENARIOS, agg_asym, check_planner, plan_asym
from gipa_core import CFG, REG

ROOT = Path(__file__).resolve().parents[1]
DENSITIES = [80, 160, 240, 320]
MODELS = ['affine', 'ps']
PRIORS = ['uniform', 'capacity']
BOOT_REPS, BOOT_SEED = 10000, 20260804


def _clean(d):
    return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in d.items()}


def paired_ci(x, y):
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return None
    rg = np.random.default_rng(BOOT_SEED)
    bs = d[rg.integers(0, len(d), (BOOT_REPS, len(d)))].mean(1)
    return dict(mean=float(d.mean()), ci95=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
                n=int(len(d)))


def variants():
    out = [('local', 'capacity'), ('greedy', 'capacity'), ('indep', 'capacity'), ('rbr', 'capacity'),
           ('plan', 'capacity'), ('la', 'capacity')]
    out += [(p, pr) for p in PRIOR_POLICIES for pr in PRIORS]
    return out


def name(pol, prior):
    return f'{pol}|{prior}' if pol in PRIOR_POLICIES else pol


def job_plan(spec):
    scen, reg, model, N = spec
    return spec, plan_asym(N, REG[reg], SCENARIOS[scen], model, z=2.0)


def job(spec):
    scen, reg, model, N, pol, prior, plan, la_cfg = spec
    kw = dict(model=model, prior=prior, plan=plan, la_cfg=la_cfg, warm=500 if pol == 'la' else CFG['WARM'])
    if pol == 'plan' and plan is None:
        return dict(key=(scen, reg, model, N, name(pol, prior)), mean=None, per_seed=None)
    m, rs = agg_asym(pol, N, REG[reg], SCENARIOS[scen], **kw)
    return dict(key=(scen, reg, model, N, name(pol, prior)), mean=_clean(m),
                per_seed=dict(lat=[r['lat'] for r in rs], herd_fast=[r['herd_fast'] for r in rs]))


def checks():
    """Symmetric reproduction of archived/E2 values and the planner check."""
    ref_main = json.load(open(ROOT / 'results' / 'res_main.json'))
    ref_e2 = json.load(open(ROOT / 'results' / 'res_calibration.json'))['main']
    worst = 0.0
    for pol in ['gipa', 'msoft', 'msoft_prior', 'entreg', 'greedy', 'indep', 'rbr', 'local']:
        for reg in ['A', 'B']:
            for N in DENSITIES:
                m, _ = agg_asym(pol, N, REG[reg], SCENARIOS['sym'], model='affine', prior='uniform')
                worst = max(worst, abs(m['lat'] - ref_main[reg][str(N)][pol]['lat']))
                key = pol + '|class' if pol in ['gipa', 'msoft', 'msoft_prior', 'entreg'] else pol
                m, _ = agg_asym(pol, N, REG[reg], SCENARIOS['sym'], model='ps', prior='uniform')
                worst = max(worst, abs(m['lat'] - ref_e2[reg][str(N)][key]['lat']))
    gap, gaps = check_planner()
    return dict(symmetric_max_abs_diff_ms=worst, symmetric_passed=worst == 0.0,
                planner_max_rel_gap=gap, planner_gaps=gaps, planner_passed=gap <= 1e-6)


def main():
    ap = argparse.ArgumentParser(description='E9 asymmetric RSUs')
    ap.add_argument('--out', type=Path, default=ROOT / 'results' / 'res_asym.json')
    ap.add_argument('--workers', type=int, default=2)
    args = ap.parse_args()

    chk = checks()
    print(f"0. symmetric reproduction: max |diff| {chk['symmetric_max_abs_diff_ms']:.3g} ms "
          f"({'passed' if chk['symmetric_passed'] else 'FAILED'}); planner vs grid: max relative gap "
          f"{chk['planner_max_rel_gap']:.2g} ({'passed' if chk['planner_passed'] else 'FAILED'})", flush=True)

    la_chosen = json.load(open(ROOT / 'results' / 'res_la.json'))['chosen']
    grid = [(scen, reg, model, N) for scen in SCENARIOS for reg in ['A', 'B'] for model in MODELS for N in DENSITIES]
    with Pool(args.workers) as pool:
        plans = dict(pool.map(job_plan, grid))
    specs = []
    for g in grid:
        scen, reg, model, N = g
        b = la_chosen[f'{reg}|{model}|500']['best']
        la_cfg = dict(algo=b['algo'], resp=b['resp'], rate=b['rate'])
        for pol, prior in variants():
            specs.append((scen, reg, model, N, pol, prior, plans[g], la_cfg))
    with Pool(args.workers) as pool:
        res = pool.map(job, specs, chunksize=2)
    R = {tuple(r['key']): r for r in res}

    main_tab, paired, per_seed = {}, {}, {}
    for scen, reg, model, N in grid:
        key = f'{scen}|{reg}|{model}'
        main_tab.setdefault(key, {})[N] = {k[4]: v['mean'] for k, v in R.items() if k[:4] == (scen, reg, model, N)}
        for k, v in R.items():
            if k[:4] == (scen, reg, model, N) and v['per_seed'] is not None:
                per_seed[f'{key}|{N}|{k[4]}'] = v['per_seed']

        def ps(pol, met='lat'):
            v = R[(scen, reg, model, N, pol)]['per_seed']
            return None if v is None else v[met]
        pr = {}
        for prior in PRIORS:
            g = f'gipa|{prior}'
            pr[f'{g} - plan'] = paired_ci(ps(g), ps('plan')) if ps('plan') else None
            pr[f'{g} - gipa_nodiff|{prior}'] = paired_ci(ps(g), ps(f'gipa_nodiff|{prior}'))
            pr[f'herd: {g} - gipa_nodiff|{prior}'] = paired_ci(ps(g, 'herd_fast'), ps(f'gipa_nodiff|{prior}', 'herd_fast'))
        pr['la - plan'] = paired_ci(ps('la'), ps('plan')) if ps('plan') else None
        paired.setdefault(key, {})[N] = pr

        t = main_tab[key][N]

        def f(p, met='lat', scale=1, fmt='{:6.2f}'):
            v = t.get(p)
            return '   n/a' if v is None or v.get(met) is None else fmt.format(scale * v[met])
        print(f"1. {key:18s} N={N:3d} lat: GIPA u/c {f('gipa|uniform')}/{f('gipa|capacity')} "
              f"noDiff c {f('gipa_nodiff|capacity')} plan {f('plan')} LA {f('la')} local {f('local')} "
              f"greedy {f('greedy')} RBR {f('rbr')} | herd GIPA u/c {f('gipa|uniform', 'herd_fast', fmt='{:4.2f}')}/"
              f"{f('gipa|capacity', 'herd_fast', fmt='{:4.2f}')} noDiff {f('gipa_nodiff|capacity', 'herd_fast', fmt='{:4.2f}')} "
              f"plan {f('plan', 'herd_fast', fmt='{:4.2f}')} | miss S GIPA u/c {f('gipa|uniform', 'mS', 100, '{:5.1f}')}/"
              f"{f('gipa|capacity', 'mS', 100, '{:5.1f}')}%", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(dict(checks=chk, scenarios=SCENARIOS,
                       plans={'|'.join(map(str, k)): v for k, v in plans.items()},
                       main=main_tab, paired=paired, per_seed=per_seed), fh)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
