"""
exp_calibration.py
==================
Workplan experiment E2: validate GIPA under the computation-based latency
model of manuscript Eq. (5), with physically calibrated, class-dependent
compute and exact processor sharing (Reviewer 2, comment 2).

STATUS: written for the first revision of manuscript ITS-2026-08-0579. Not
part of the submitted v1.0.0 results.

What it runs (all over the 30 matched seeds)
--------------------------------------------
A. Special-case check. Eq. (5) with class-independent compute
   (a_S = a_B = 0.55 ms) must reproduce the affine model of the submitted
   experiments (calibration.check_special_case).
B. Principal comparison, density sweep and ablation under the default
   calibration (F = 33.45 GHz; safety 0.299 ms and background 0.658 ms of
   RSU compute per task), regimes A and B, N in {80, 160, 240, 320}:
     - policies that do not use the prior: greedy, independent softmax,
       reactive best response, congestion pricing, all-local;
     - prior-dependent policies, each with the submitted class-agnostic
       prior ('agnostic') and with the class-aware prior from the same
       calibration ('class'): masked softmax, prior-adjusted masked softmax,
       entropy-regularised selection, GIPA and its five ablations;
     - the planner, planned with the affine model ('agnostic') or with
       Eq. (5) ('class'), plus the Eq. (5) planner with margin z = 2.
   Q-learning is retrained under both latency models in experiment E4.
C. RSU capacity sweep at N = 160, regimes A and B: F in {10, 18.4, 33.45, 60}
   GHz (mean compute per task 1.84, 1.00, 0.55 and 0.31 ms), for all-local,
   reactive best response, prior-adjusted masked softmax (class prior), GIPA
   (both priors) and the Eq. (5) planner.

Output
------
results/res_calibration.json (path configurable with --out):
    {
      "calibration": {...default calibration constants...},
      "special_case": {"max_abs_diff": float, "passed": bool},
      "main":     {regime: {N: {key: {metric: mean} or null}}},
      "per_seed": {regime: {N: {key: [30 per-seed dicts] or null}}},
      "capacity_sweep": {regime: {F_ghz: {key: {...}}}}
    }
Keys are a policy name, optionally with '|agnostic' or '|class'; null marks
an infeasible plan. Metric keys are those of gipa_core.run.

Runtime
-------
A few minutes on a laptop CPU.

Run (from the repository root)
------------------------------
    PYTHONPATH=src python experiments/exp_calibration.py
"""

import argparse
import json
from pathlib import Path

import numpy as np

from gipa_core import REG
from planner import plan_allocation
from calibration import (make_calibration, check_special_case, agg_ps,
                         plan_allocation_ps)

DENSITIES = [80, 160, 240, 320]
PLAIN = ['greedy', 'indep', 'rbr', 'price', 'local']
PRIOR_POLS = [('msoft', {}), ('msoft_prior', {}), ('entreg', {}), ('gipa', {}),
              ('gipa_noamp', {'amp_on': False}), ('gipa_nodiff', {'lam_on': False}),
              ('gipa_fixed', {'anneal': False}), ('gipa_binary', {'graded': False}),
              ('gipa_noprior', {'prior_on': False})]
F_SWEEP = [10.0, 18.4, None, 60.0]          # None = default calibration (33.45 GHz)


def _clean(d):
    """Replace NaN with None so the output is strict JSON."""
    return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in d.items()}


def _fmt(m):
    """One-line summary of a result dict."""
    return (f"lat {m['lat']:6.2f} p95 {m['p95']:5.1f} miss S/B {100 * m['mS']:5.1f}/{100 * m['mB']:4.1f}% "
            f"off S/B {m['offS']:.2f}/{m['offB']:.2f} maxRSU {m['maxl']:5.1f}")


def main():
    """Run parts A-C and write the JSON output."""
    ap = argparse.ArgumentParser(description='E2 Eq. (5) calibration experiment')
    root = Path(__file__).resolve().parents[1]
    ap.add_argument('--out', type=Path, default=root / 'results' / 'res_calibration.json')
    args = ap.parse_args()
    cal = make_calibration()

    # ---------------- A. special case ----------------
    worst, ok = check_special_case()
    print(f'A. Eq. (5) with class-independent compute vs affine model: '
          f'max |difference| = {worst:.1e} -> {"PASS" if ok else "FAIL"}')

    # ---------------- B. main comparison ----------------
    main_res, per_seed = {}, {}
    for reg in ['A', 'B']:
        R = REG[reg]
        main_res[reg], per_seed[reg] = {}, {}
        for N in DENSITIES:
            row, seeds = {}, {}

            def put(key, m, rs):
                row[key] = _clean(m)
                seeds[key] = [_clean(r) for r in rs]

            for pol in PLAIN:
                put(pol, *agg_ps(pol, N, R, cal))
            for name, kw in PRIOR_POLS:
                base = 'gipa' if name.startswith('gipa') else name
                for prior in ['agnostic', 'class']:
                    put(f'{name}|{prior}', *agg_ps(base, N, R, cal, prior, **kw))
            plans = {'plan|agnostic': plan_allocation(N, R),
                     'plan|class': plan_allocation_ps(N, R, cal),
                     'plan_z2|class': plan_allocation_ps(N, R, cal, z=2.0)}
            for key, plan in plans.items():
                if plan is None:
                    row[key], seeds[key] = None, None
                else:
                    m, rs = agg_ps('plan', N, R, cal, plan=plan)
                    put(key, m, rs)
                    row[key]['plan'] = plan
            main_res[reg][N], per_seed[reg][N] = row, seeds
            print(f'B. Regime {reg}, N = {N}')
            for key in ['local', 'rbr', 'msoft|agnostic', 'msoft|class', 'gipa|agnostic',
                        'gipa|class', 'plan|agnostic', 'plan|class', 'plan_z2|class']:
                r = row[key]
                print(f'   {key:16s} ' + ('infeasible plan' if r is None else _fmt(r)), flush=True)

    # ---------------- C. RSU capacity sweep ----------------
    sweep = {}
    for reg in ['A', 'B']:
        R = REG[reg]
        sweep[reg] = {}
        for F in F_SWEEP:
            c = make_calibration(F_ghz=F)
            res = {}
            for key, pol, prior in [('local', 'local', 'class'), ('rbr', 'rbr', 'class'),
                                    ('msoft_prior|class', 'msoft_prior', 'class'),
                                    ('gipa|agnostic', 'gipa', 'agnostic'),
                                    ('gipa|class', 'gipa', 'class')]:
                m, _ = agg_ps(pol, 160, R, c, prior)
                res[key] = _clean(m)
            plan = plan_allocation_ps(160, R, c)
            if plan is None:
                res['plan|class'] = None
            else:
                m, _ = agg_ps('plan', 160, R, c, plan=plan)
                res['plan|class'] = dict(_clean(m), plan=plan)
            sweep[reg][round(c['F_ghz'], 2)] = dict(res, mean_compute_ms=c['a_mix'])
            print(f"C. Regime {reg}, F = {c['F_ghz']:5.2f} GHz (mean compute {c['a_mix']:.2f} ms/task)")
            for key, r in res.items():
                print(f'   {key:18s} ' + ('infeasible plan' if r is None else _fmt(r)), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump({'calibration': cal, 'special_case': {'max_abs_diff': worst, 'passed': bool(ok)},
                   'main': main_res, 'per_seed': per_seed, 'capacity_sweep': sweep}, fh, indent=1)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
