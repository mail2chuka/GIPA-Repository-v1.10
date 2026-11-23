"""
exp_capacity.py
===============
Workplan experiment E3: RSU capacity compliance, separating what vehicles
selected from what RSUs admitted (Reviewer 2, comment 3).

STATUS: written for the first revision of manuscript ITS-2026-08-0579. Not
part of the submitted v1.0.0 results.

What it runs
------------
For regimes A and B, densities N in {80, 160, 240, 320}, and 11 policies
(the 10 archived policies plus the planned allocation from planner.py),
each policy is run twice over the 30 matched seeds:

* soft : every selected task is admitted (the v1.0.0 behaviour);
* ac   : the same admission layer wraps every policy (capacity.admit):
         deadline-safe, random-order admission; rejected tasks run locally
         after a 1 ms handshake penalty.

The Q-learner is trained once per regime at N = 160, as in exp.py, and its
previous-slot load is reset at the start of every episode (``ql_reset``),
fixing a known issue in the archived evaluation. Its numbers can therefore
differ slightly from res_main.json.

The planned allocation uses planner.plan_allocation(N, R, z=0); where no
feasible plan exists (Regime B, N >= 240) it is stored as null.

Output
------
results/res_capacity.json (path configurable with --out):
    {
      "capacities": {regime: {"Q_S": int, "Q_B": int}},
      "main":     {regime: {N: {policy: {"soft": {...}, "ac": {...}}}}},
      "per_seed": same keys, lists of 30 per-seed dicts
    }
Metric keys are documented in src/capacity.py.

Seeds
-----
Environment 1000-1029, action sampling 5000-5029 (paired with exp.py),
admission order 9000-9029, Q-learning exploration 5999.

Runtime
-------
Several minutes on a laptop CPU.

Run (from the repository root)
------------------------------
    PYTHONPATH=src python experiments/exp_capacity.py
"""

import argparse
import json
from pathlib import Path

import numpy as np

from gipa_core import CFG, REG, QL
from capacity import agg_ac, capacities
from planner import plan_allocation

DENSITIES = [80, 160, 240, 320]
POLICIES = ['greedy', 'indep', 'rbr', 'price', 'msoft', 'msoft_prior', 'entreg',
            'ql', 'local', 'gipa', 'plan']


def _clean(d):
    """Replace NaN with None so the output is strict JSON."""
    return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in d.items()}


def _line(name, s, a):
    """Console summary: soft versus admission-control results for one policy."""
    def pct(x):
        return '  n/a' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{100 * x:5.1f}'
    return (f"  {name:12s} overload(sel) {pct(s['over_sel'])}%  maxSel {s['maxl_sel']:5.1f} | "
            f"soft lat {s['lat']:6.2f} missS {pct(s['mS'])}% missB {pct(s['mB'])}% | "
            f"AC lat {a['lat']:6.2f} missS {pct(a['mS'])}% missB {pct(a['mB'])}% "
            f"rej {pct(a['rej'])}% maxAdm {a['maxl_adm']:5.1f}")


def main():
    """Run soft and admission-control variants of every policy; write JSON."""
    ap = argparse.ArgumentParser(description='E3 capacity experiment')
    root = Path(__file__).resolve().parents[1]
    ap.add_argument('--out', type=Path, default=root / 'results' / 'res_capacity.json')
    args = ap.parse_args()

    caps, main_res, per_seed = {}, {}, {}
    for reg in ['A', 'B']:
        R = REG[reg]
        q_s, q_b = capacities(R)
        caps[reg] = {'Q_S': q_s, 'Q_B': q_b}
        ql = QL(CFG['M'] + 1)
        ql.train(160, R)
        main_res[reg], per_seed[reg] = {}, {}
        for N in DENSITIES:
            main_res[reg][N], per_seed[reg][N] = {}, {}
            for pol in POLICIES:
                kw = {}
                if pol == 'ql':
                    kw['ql'] = ql
                if pol == 'plan':
                    plan = plan_allocation(N, R, z=0.0)
                    if plan is None:
                        main_res[reg][N][pol] = None
                        per_seed[reg][N][pol] = None
                        continue
                    kw['plan'] = plan
                row, seeds = {}, {}
                for mode, ac in [('soft', False), ('ac', True)]:
                    m, rs = agg_ac(pol, N, R, ac=ac, ql_reset=True, **kw)
                    row[mode] = _clean(m)
                    seeds[mode] = [_clean(r) for r in rs]
                main_res[reg][N][pol] = row
                per_seed[reg][N][pol] = seeds
            print(f'Regime {reg}, N = {N}  (Q_S = {q_s}, Q_B = {q_b})')
            for pol in POLICIES:
                r = main_res[reg][N][pol]
                if r is None:
                    print(f'  {pol:12s} infeasible plan')
                else:
                    print(_line(pol, r['soft'], r['ac']))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump({'capacities': caps, 'main': main_res, 'per_seed': per_seed}, fh, indent=1)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
