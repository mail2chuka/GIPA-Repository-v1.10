"""
exp_planner.py
==============
Evaluate the system-optimal planned allocation (``src/planner.py``) against
GIPA and all-local execution.

STATUS: exploratory, written for the first revision of manuscript
ITS-2026-08-0579. Not part of the submitted v1.0.0 results.

What it runs
------------
1. Main comparison. For regimes A and B and densities N in {80, 160, 240,
   320}: GIPA (heuristic prior), all-local, and the planned allocation with
   margins z = 0 and z = 2, each over the same 30 matched seeds used by
   exp.py (environment 1000-1029, action sampling 5000-5029). Because the
   seeds match, per-seed results can be paired with the archived policies.
2. Misspecification sweep (Regime A, true N in {160, 240}): the plan is built
   for N_bar = f * N with f in {0.5, 0.75, 1.0, 1.25, 1.5} and evaluated at
   the true N, to show how sensitive the planner is to a wrong fleet-size
   estimate.

Output
------
results/res_planner.json (path configurable with --out):
    {
      "main":   {regime: {N: {"gipa": {...}, "local": {...},
                               "plan_z0": {...}, "plan_z2": {...}}}},
      "misspec": {"A": {N: {f: {...}}}},
      "per_seed": same keys as "main", lists of 30 per-seed dicts
    }
Each result dict holds the metrics of ``gipa_core.run`` (mS, mB, lat, p95,
fair, off, offS, offB, en, maxl). Planned entries also carry "plan": the
output of ``planner.plan_allocation``; an infeasible plan is stored as
{"plan": null}.

Runtime
-------
A few minutes on a laptop CPU.

Run (from the repository root)
------------------------------
    PYTHONPATH=src python experiments/exp_planner.py
    PYTHONPATH=src python experiments/exp_planner.py --out /tmp/res_planner.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from gipa_core import REG, agg
from planner import plan_allocation, agg_planned

DENSITIES = [80, 160, 240, 320]
MARGINS = [0.0, 2.0]
MISSPEC_N = [160, 240]
MISSPEC_F = [0.5, 0.75, 1.0, 1.25, 1.5]


def _clean(d):
    """Replace NaN with None so the output is strict JSON."""
    return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in d.items()}


def _fmt(m):
    """One-line console summary of a result dict."""
    return (f"lat={m['lat']:6.2f} p95={m['p95']:5.1f} missS={100 * m['mS']:5.1f}% "
            f"missB={100 * m['mB']:5.2f}% offS={m['offS']:.2f} offB={m['offB']:.2f} "
            f"maxRSU={m['maxl']:5.1f}")


def main():
    """Run the main comparison and the misspecification sweep; write JSON."""
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    root = Path(__file__).resolve().parents[1]
    ap.add_argument('--out', type=Path, default=root / 'results' / 'res_planner.json')
    args = ap.parse_args()

    main_res, per_seed = {}, {}
    for reg in ['A', 'B']:
        R = REG[reg]
        main_res[reg], per_seed[reg] = {}, {}
        print(f'===== Regime {reg}')
        for N in DENSITIES:
            row, seeds = {}, {}
            for name in ['gipa', 'local']:
                m, rs = agg(name, N, R)
                row[name], seeds[name] = _clean(m), [_clean(r) for r in rs]
                print(f'N={N:3d} {name:8s} {_fmt(m)}')
            for z in MARGINS:
                key = f'plan_z{int(z)}'
                plan = plan_allocation(N, R, z=z)
                if plan is None:
                    row[key], seeds[key] = {'plan': None}, []
                    print(f'N={N:3d} {key:8s} infeasible')
                    continue
                m, rs = agg_planned(N, R, plan)
                row[key] = dict(_clean(m), plan=plan)
                seeds[key] = [_clean(r) for r in rs]
                print(f"N={N:3d} {key:8s} {_fmt(m)}  (x_S={plan['x_S']:.3f}, x_B={plan['x_B']:.3f})")
            main_res[reg][N], per_seed[reg][N] = row, seeds

    print('===== Misspecification sweep (Regime A)')
    mis = {'A': {}}
    for N in MISSPEC_N:
        mis['A'][N] = {}
        for f in MISSPEC_F:
            plan = plan_allocation(f * N, REG['A'])
            m, _ = agg_planned(N, REG['A'], plan)
            mis['A'][N][f] = dict(_clean(m), plan=plan)
            print(f'true N={N} plan for {f:.2f}N: {_fmt(m)}')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump({'main': main_res, 'misspec': mis, 'per_seed': per_seed}, fh, indent=1)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
