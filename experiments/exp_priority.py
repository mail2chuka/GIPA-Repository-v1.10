"""
exp_priority.py
===============
Experiment E18: queue priority at the original targets (identical tasks,
Regime A, affine latency, carried-over local work; run with
GIPA_LOCAL_CARRY=own, see run_e18.sh).

E17 (exp_backlog.py) changes two things at once: it re-optimises the offload
target with the screened local latency (backlog.backlog_plan) and it screens
background tasks by their own backlog. This experiment separates the two. The
offload targets are those of the carry-aware planned prior of E13
(planner.plan_allocation, z = 2): the safety share is unchanged and the
background vehicles keep the plan's effective offload share f_B (sampling plus
redirects). Only the order in which background vehicles offload changes: the
screen (theta, q) of backlog.best_screen gives the same stationary share f_B
with the lowest mean local latency, so vehicles with longer queues offload
first.

Arms (30 archived evaluation seeds, N = 80, 160, 240, 320)
  priority      screen with q_min = 0.1 (the value tuned in E17)
  priority_det  screen with q_min = 0 (deterministic threshold where preferred)
Comparator: carry-aware planned prior, results/carry_own/res_planner.json (plan_z2).
Paired differences: 10,000 bootstrap resamples of the 30 seeds (generator 20260804).
Writes results/res_priority.json and results/priority_summary.md.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import backlog as B  # noqa: E402
from gipa_core import CFG, REG, LOCAL_CARRY  # noqa: E402
from planner import plan_allocation  # noqa: E402
from exp_backlog import overall_miss, boot  # noqa: E402

NS = (80, 160, 240, 320)
R = REG['A']
Q_MIN = 0.1


def main():
    assert LOCAL_CARRY == 'own', 'set GIPA_LOCAL_CARRY=own'
    M = CFG['M']
    old = json.load(open(ROOT / 'results' / 'carry_own' / 'res_planner.json'))['per_seed']['A']
    out = dict(q_min=Q_MIN, cells={})
    md = ['# E18: queue priority at the original targets (identical tasks, Regime A, carried-over work)', '',
          'Targets: carry-aware planned prior (E13, z = 2). Screen chosen for the same background offload share. '
          'Mean latency (ms), overall misses (%), measured background offload share, paired difference '
          'screened minus planned prior with 95% bootstrap CI.', '',
          '| N | Arm | f_B target | theta, q | Latency | Misses | Measured f_B | vs planned prior (ms) |',
          '|---|---|---|---|---|---|---|---|']
    for N in NS:
        p0 = plan_allocation(N, R, z=2.0)
        fB = p0['carry']['B']['target_offload']
        base = old[str(N)]['plan_z2']
        base_lat = np.array([r['lat'] for r in base])
        base_miss = np.array([overall_miss(N, s, r) for s, r in enumerate(base)])
        base_offB = float(np.mean([r['offB'] for r in base]))
        cell = dict(plan_z2=dict(lat=float(base_lat.mean()), miss=float(base_miss.mean()), offB=base_offB,
                                 x_S=p0['x_S'], f_B=fB))
        for name, qm in (('priority', Q_MIN), ('priority_det', 0.0)):
            scr = B.best_screen(fB, R['L_B'], R['D_B'], q_min=qm)
            plan = dict(x_S=p0['x_S'], x_B=fB / M, theta=scr['theta'], q=scr['q'])
            m, rs = B.agg_backlog(N, R, plan)
            lat = np.array([r['lat'] for r in rs])
            miss = np.array([overall_miss(N, s, r) for s, r in enumerate(rs)])
            d0, c0 = boot(lat - base_lat)
            cell[name] = dict(screen=scr, lat=float(lat.mean()), miss=float(miss.mean()), offB=m['offB'],
                              mS=m['mS'], mB=m['mB'], p95=m['p95'], maxl=m['maxl'],
                              per_seed=[dict(lat=float(a), miss=float(b)) for a, b in zip(lat, miss)],
                              vs_plan=[d0, c0])
            th = 'none' if scr['theta'] is None else scr['theta']
            md.append(f"| {N} | {name} | {fB:.4f} | {th}, {scr['q']:.3f} | {lat.mean():.2f} | {100 * miss.mean():.3f} | "
                      f"{m['offB']:.4f} | {d0:+.2f} [{c0[0]:+.2f}, {c0[1]:+.2f}] |")
            print(md[-1], flush=True)
        md.append(f"| {N} | planned prior (E13) | {fB:.4f} | | {base_lat.mean():.2f} | {100 * base_miss.mean():.3f} | "
                  f"{base_offB:.4f} | |")
        out['cells'][str(N)] = cell
    json.dump(out, open(ROOT / 'results' / 'res_priority.json', 'w'), indent=1)
    open(ROOT / 'results' / 'priority_summary.md', 'w').write('\n'.join(md) + '\n')


if __name__ == '__main__':
    main()
