"""
exp_prediction.py
=================
Workplan experiment E1: robustness to prediction error (Reviewer 2, comment
1: "evaluate how prediction errors affect offloading decisions, latency and
deadline-miss rates").

STATUS: written for the first revision of manuscript ITS-2026-08-0579. Not
part of the submitted v1.0.0 results.

Design (Regime A; 30 matched seeds per condition; see src/prediction.py)
------------------------------------------------------------------------
At N = 160:
  baseline      oracle scores and mask (reproduces the archived results)
  arm (a)       score error only,  sigma in {0.05, 0.10, 0.20, 0.30}
  arm (b)       mask error only,   same sigmas
  combined      one predictor feeds scores and mask, same sigmas
  bias          combined, sigma = 0.10, bias in {-0.10 (optimistic), +0.10}
  guard band    combined with guard in {0.10, 0.20}, at (sigma 0.30, bias 0)
                and at (sigma 0.10, bias -0.10)
  arm (c)       prior misspecification: N_bar = f * N, f in
                {0.5, 0.75, 1.25, 1.5}; congestion slope x {0.7, 1.3}
                (only policies that use the prior; the planner is rebuilt
                with the same wrong inputs)
  drift + EWMA  AR(1) drifting true latency, sigma_eta in {0.1, 0.2, 0.3},
                with an oracle predictor and with the EWMA predictor on the
                same drift; plus EWMA with guard 0.10 at sigma_eta = 0.2.
                The oracle-versus-EWMA difference isolates misses caused by
                prediction error from misses caused by the drift itself.
At N = 80, where the deadline margin is tight: baseline; combined at sigma
in {0.10, 0.20, 0.30}; arms (a) and (b) alone at sigma = 0.30; combined with
bias -0.10 at sigma = 0.10; guard band {0.10, 0.20} at (sigma 0.30, bias 0)
and at (sigma 0.10, bias -0.10); and drift sigma_eta = 0.2 with oracle and
EWMA.

Policies: greedy, independent softmax, reactive best response, congestion
pricing, masked softmax, prior-adjusted masked softmax, entropy-regularised
selection, all-local, GIPA and the planner (planner.py, built for the true N
except in arm c). Q-learning is excluded because it never uses latency
predictions.

Output
------
results/res_prediction.json (path configurable with --out):
    {"conditions": [
        {"name": str, "N": int, "params": {...},
         "results": {policy: {metric: mean}},
         "per_seed": {policy: [30 per-seed dicts]}}, ...]}
Metric keys are documented in src/prediction.py.

Seeds
-----
Environment 1000-1029 and action sampling 5000-5029 (paired with exp.py);
prediction noise and drift 7000-7029.

Runtime
-------
A few minutes on a laptop CPU.

Run (from the repository root)
------------------------------
    PYTHONPATH=src python experiments/exp_prediction.py
"""

import argparse
import json
from pathlib import Path

import numpy as np

from gipa_core import CFG, REG
from planner import plan_allocation
from prediction import agg_pred, prior_misspecified, POLICIES, PRIOR_POLICIES

SIGMAS = [0.05, 0.10, 0.20, 0.30]


def conditions():
    """Enumerate (name, N, kwargs for run_pred, prior spec, policy list)."""
    out = []
    full = POLICIES

    def add(name, N, kw=None, prior=(1.0, 1.0), pols=full):
        out.append((name, N, kw or {}, prior, pols))

    N = 160
    add('baseline', N, dict(arm='none'))
    for arm in ['score', 'mask', 'both']:
        for s in SIGMAS:
            add(f'{arm}_sigma{s:.2f}', N, dict(arm=arm, sigma=s))
    for b in [-0.10, 0.10]:
        add(f'both_sigma0.10_bias{b:+.2f}', N, dict(arm='both', sigma=0.10, bias=b))
    for g in [0.10, 0.20]:
        add(f'both_sigma0.30_guard{g:.2f}', N, dict(arm='both', sigma=0.30, guard=g))
        add(f'both_sigma0.10_bias-0.10_guard{g:.2f}', N,
            dict(arm='both', sigma=0.10, bias=-0.10, guard=g))
    for f in [0.5, 0.75, 1.25, 1.5]:
        add(f'prior_f{f:.2f}', N, dict(arm='none'), prior=(f, 1.0), pols=PRIOR_POLICIES)
    for x in [0.7, 1.3]:
        add(f'prior_xi{x:.1f}', N, dict(arm='none'), prior=(1.0, x), pols=PRIOR_POLICIES)
    for se in [0.1, 0.2, 0.3]:
        for pred in ['oracle', 'ewma']:
            add(f'drift{se:.1f}_{pred}', N,
                dict(arm='both', truth='ar1', sigma_eta=se, predictor=pred))
    add('drift0.2_ewma_guard0.10', N,
        dict(arm='both', truth='ar1', sigma_eta=0.2, predictor='ewma', guard=0.10))

    N = 80
    add('baseline', N, dict(arm='none'))
    for s in [0.10, 0.20, 0.30]:
        add(f'both_sigma{s:.2f}', N, dict(arm='both', sigma=s))
    # At N = 80 the deadline margin is tight, so this is where the arms and
    # the guard band separate.
    for arm in ['score', 'mask']:
        add(f'{arm}_sigma0.30', N, dict(arm=arm, sigma=0.30))
    add('both_sigma0.10_bias-0.10', N, dict(arm='both', sigma=0.10, bias=-0.10))
    for g in [0.10, 0.20]:
        add(f'both_sigma0.30_guard{g:.2f}', N, dict(arm='both', sigma=0.30, guard=g))
        add(f'both_sigma0.10_bias-0.10_guard{g:.2f}', N,
            dict(arm='both', sigma=0.10, bias=-0.10, guard=g))
    for pred in ['oracle', 'ewma']:
        add(f'drift0.2_{pred}', N, dict(arm='both', truth='ar1', sigma_eta=0.2, predictor=pred))
    return out


def _clean(d):
    """Replace NaN with None so the output is strict JSON."""
    return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in d.items()}


def main():
    """Run every condition; print a summary line per condition; write JSON."""
    ap = argparse.ArgumentParser(description='E1 prediction-error experiment')
    root = Path(__file__).resolve().parents[1]
    ap.add_argument('--out', type=Path, default=root / 'results' / 'res_prediction.json')
    args = ap.parse_args()
    R = REG['A']

    results = []
    for name, N, kw, (f, xm), pols in conditions():
        cb = prior_misspecified(N, f, xm)
        plan = plan_allocation(f * N, R, xi=xm * CFG['CONG'] if xm != 1.0 else None)
        res, seeds = {}, {}
        for pol in pols:
            m, rs = agg_pred(pol, N, R, cb, plan=plan, **kw)
            res[pol] = _clean(m)
            seeds[pol] = [_clean(r) for r in rs]
        results.append(dict(name=name, N=N,
                            params=dict(kw, prior_f=f, prior_xi_mult=xm),
                            results=res, per_seed=seeds))
        g, ms, pl = res['gipa'], res['msoft'], res['plan']
        print(f"N={N:3d} {name:34s} err {g['pred_err']:.3f} | "
              f"GIPA lat {g['lat']:6.2f} miss S/B {100*g['mS']:5.2f}/{100*g['mB']:5.2f}% offB {g['offB']:.2f} | "
              f"msoft lat {ms['lat']:6.2f} miss S/B {100*ms['mS']:5.2f}/{100*ms['mB']:5.2f}% | "
              f"plan lat {pl['lat']:6.2f} miss S/B {100*pl['mS']:5.2f}/{100*pl['mB']:5.2f}%", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump({'conditions': results}, fh, indent=1)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
