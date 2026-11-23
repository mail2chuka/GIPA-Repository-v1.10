"""
exp_prediction_obs.py
=====================
Workplan experiment E1b: what each vehicle can observe, and the planned
policy when it does use per-task predictions (second revision round, Nov 2026).

STATUS: written for the first revision of manuscript ITS-2026-08-0579. Not
part of the submitted v1.0.0 results. E1 (experiments/exp_prediction.py,
results/res_prediction.json) is unchanged.

Why
---
Two simplifications of E1:
1. Its EWMA predictor learns from the *uncongested* latency of the action a
   vehicle took, which assumes the RSU returns a timing breakdown with each
   result. A vehicle may see only its realised latency, congestion included.
2. Its planned policy samples the plan and so never reads a per-task
   prediction; its robustness to prediction error was by construction.
   GIPA with planned prior (Algorithm 2) does read them: through the
   predicted-feasibility mask and through the advantage (16) that drives
   amplification.

Design (Regime A, affine model, 30 matched seeds, N = 80, 160, 240)
------------------------------------------------------------------
A. Per-task prediction noise, static truth, one predictor feeding scores and
   mask: sigma in {0.1, 0.2, 0.3}, and sigma 0.1 with a 10% optimistic bias;
   guard band 0 and 0.1.
B. Observability under drift: AR(1) drifting latency (sigma_eta 0.1, 0.2,
   0.3) with the oracle predictor and the EWMA predictor observing
   'breakdown' (E1), 'total' (realised latency, used as is) or
   'total_debiased' (realised latency minus the congestion the broadcast
   prior anticipates); guard 0 for all, and guard 0.1 at sigma_eta 0.2.
C. Stage 0 latency-model error: the plan (and the class averages q_bar of
   'gipa_plan') computed with local latencies L_k and idle-RSU latencies E_k
   scaled by 0.8 or 1.2 (four combinations); oracle per-task predictions.

Policies: all-local, the plan sampled directly ('plan', as in E1), the plan
restricted to the predicted-feasible set ('plan_mask'), GIPA with planned
prior ('gipa_plan'), the same two with the RSU part of the mask evaluated at
the margin load of P1, planned mean plus z = 2 standard deviations
('plan_mask_z2', 'gipa_plan_z2'), GIPA with the heuristic prior ('gipa'),
prior-adjusted masked softmax, and in B reactive best response. The plan is
built for the true fleet size with z = 0, as in E1 (in Regime A the margin
does not change the plan).

Output
------
results/res_prediction_obs.json: {"conditions": [{"block", "name", "N",
"params", "results": {policy: means}, "per_seed": {policy: [...]}}]}.

Runtime
-------
About 5-10 minutes on a laptop CPU.

Run (from the repository root)
------------------------------
    PYTHONPATH=src python experiments/exp_prediction_obs.py
"""

import argparse
import json
from pathlib import Path

import numpy as np

from gipa_core import REG
from planner import plan_allocation
from prediction import agg_pred

FLEETS = [80, 160, 240]
# label -> (policy, extra keyword arguments)
SPEC = {'local': ('local', {}), 'plan': ('plan', {}), 'plan_mask': ('plan_mask', {}), 'gipa_plan': ('gipa_plan', {}),
        'plan_mask_z2': ('plan_mask', dict(mask_z=2.0)), 'gipa_plan_z2': ('gipa_plan', dict(mask_z=2.0)),
        'gipa': ('gipa', {}), 'msoft_prior': ('msoft_prior', {}), 'rbr': ('rbr', {})}
POL_A = ['local', 'plan', 'plan_mask', 'gipa_plan', 'plan_mask_z2', 'gipa_plan_z2', 'gipa', 'msoft_prior']
POL_B = POL_A + ['rbr']
POL_C = ['plan', 'gipa_plan', 'gipa_plan_z2']


def conditions():
    out = []
    for N in FLEETS:
        out.append(('ref', 'oracle', N, dict(arm='none'), None, POL_B))
        for g in [0.0, 0.1]:
            for s in [0.1, 0.2, 0.3]:
                out.append(('A', f'noise{s:.1f}_guard{g:.1f}', N, dict(arm='both', sigma=s, guard=g), None, POL_A))
            out.append(('A', f'noise0.1_bias-0.1_guard{g:.1f}', N,
                        dict(arm='both', sigma=0.1, bias=-0.1, guard=g), None, POL_A))
        for se in [0.1, 0.2, 0.3]:
            base = dict(arm='both', truth='ar1', sigma_eta=se)
            out.append(('B', f'drift{se:.1f}_oracle', N, dict(base, predictor='oracle'), None, POL_B))
            for obs in ['breakdown', 'total', 'total_debiased']:
                out.append(('B', f'drift{se:.1f}_ewma_{obs}', N, dict(base, predictor='ewma', obs=obs), None, POL_B))
                if se == 0.2:
                    out.append(('B', f'drift{se:.1f}_ewma_{obs}_guard0.1', N,
                                dict(base, predictor='ewma', obs=obs, guard=0.1), None, POL_B))
        for fl in [0.8, 1.2]:
            for fe in [0.8, 1.2]:
                out.append(('C', f'stage0_L{fl:.1f}_E{fe:.1f}', N, dict(arm='none'), (fl, fe), POL_C))
    return out


def _clean(d):
    return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser(description='E1b observability and prediction-using planned policies')
    root = Path(__file__).resolve().parents[1]
    ap.add_argument('--out', type=Path, default=root / 'results' / 'res_prediction_obs.json')
    args = ap.parse_args()
    R = REG['A']
    results = []
    for block, name, N, kw, scale, pols in conditions():
        if scale is None:
            plan = plan_allocation(N, R)
            extra = {}
        else:
            fl, fe = scale
            R0 = dict(R, L_S=R['L_S'] * fl, L_B=R['L_B'] * fl, E_S=R['E_S'] * fe, E_B=R['E_B'] * fe)
            plan = plan_allocation(N, R0)
            extra = dict(R0=R0)
        res, seeds = {}, {}
        for pol in pols:
            name_, kz = SPEC[pol]
            m, rs = agg_pred(name_, N, R, None, plan=plan, **kw, **extra, **kz)
            res[pol] = _clean(m)
            seeds[pol] = [_clean(r) for r in rs]
        results.append(dict(block=block, name=name, N=N, params=dict(kw, stage0_scale=scale),
                            results=res, per_seed=seeds))
        print(f"{block} N={N:3d} {name:32s} " + ' | '.join(
            f"{p} {res[p]['lat']:6.2f} {100 * res[p]['mS']:5.2f}/{100 * res[p]['mB']:5.2f}% off {res[p]['off']:.2f}"
            for p in pols), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump({'conditions': results}, fh, indent=1)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
