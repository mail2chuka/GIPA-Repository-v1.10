"""
exp_backlog.py
==============
Experiment E17: backlog-aware screening of the planned prior with
carried-over local work (identical tasks, Regime A, affine latency).

Run with GIPA_LOCAL_CARRY=own (see run_e17.sh).

Arms (30 archived evaluation seeds, N = 80, 160, 240, 320)
  screen_qX   backlog-screened plan (src/backlog.py) with minimum random
              offload probability q_min = X for vehicles below the threshold
  screen_det  the same with q_min = 0, i.e. a deterministic threshold where
              the planner prefers one (shows the synchronisation it causes)
Comparators (archived, unchanged)
  plan_z2     carry-aware planned prior, results/carry_own/res_planner.json
  SLA-MA-PPO  results/res_sla.json (homog|own)

q_min is chosen at N = 160 on the tuning seeds 2000-2009 from {0.05, 0.1,
0.2} by the rule of Section VI-E (lowest latency with a miss rate within
0.5 points of the lowest) and used at every N. Paired differences use
10,000 bootstrap resamples of the 30 seeds (generator 20260804).
Writes results/res_backlog.json and results/backlog_summary.md.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import backlog as B  # noqa: E402
from gipa_core import CFG, REG, Env, LOCAL_CARRY  # noqa: E402

NS = (80, 160, 240, 320)
QMIN_GRID = (0.05, 0.1, 0.2)
TUNE = range(2000, 2010)
R = REG['A']


def overall_miss(N, seed, r):
    S = Env(N, seed, R).isS[:N]
    return (S.sum() * r['mS'] + (~S).sum() * r['mB']) / N


def boot(d, B_=10000, seed=20260804):
    d = np.asarray(d, float)
    idx = np.random.default_rng(seed).integers(0, len(d), (B_, len(d)))
    m = d[idx].mean(1)
    return float(d.mean()), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    assert LOCAL_CARRY == 'own', 'set GIPA_LOCAL_CARRY=own'
    tune = []
    for qm in QMIN_GRID:
        p = B.backlog_plan(160, R, z=2.0, q_min=qm)
        rs = [B.run_backlog(160, s, R, p) for s in TUNE]
        tune.append(dict(q_min=qm, lat=float(np.mean([r['lat'] for r in rs])),
                         miss=float(np.mean([overall_miss(160, s, r) for s, r in zip(TUNE, rs)])),
                         theta=p['theta'], q=p['q']))
        print('tune', tune[-1], flush=True)
    lo = min(t['miss'] for t in tune)
    qsel = min([t for t in tune if t['miss'] <= lo + 0.005], key=lambda t: t['lat'])['q_min']
    print('chosen q_min', qsel)

    old = json.load(open(ROOT / 'results' / 'carry_own' / 'res_planner.json'))['per_seed']['A']
    sla = json.load(open(ROOT / 'results' / 'res_sla.json'))['groups']['homog|own']
    out = dict(tune=tune, q_min=qsel, cells={})
    md = ['# E17: backlog-aware screening with carried-over local work (identical tasks, Regime A)', '',
          f'q_min = {qsel} (tuned at N = 160 on seeds 2000-2009). Mean latency (ms), overall misses (%), '
          'paired difference screened minus comparator with 95% bootstrap CI.', '',
          '| N | Arm | Plan (theta, q, x_B, x_S) | Latency | Misses | vs planned prior (ms) | vs SLA-MA-PPO (ms) | vs SLA-MA-PPO misses (pp) |',
          '|---|---|---|---|---|---|---|---|']
    for N in NS:
        cell = {}
        base_lat = np.array([r['lat'] for r in old[str(N)]['plan_z2']])
        base_miss = np.array([overall_miss(N, s, r) for s, r in enumerate(old[str(N)]['plan_z2'])])
        runs = sla[str(N)]
        sla_lat = np.mean([[x['lat'] for x in r['per_seed']] for r in runs], 0)
        sla_miss = np.mean([[x['miss'] for x in r['per_seed']] for r in runs], 0)
        cell['plan_z2'] = dict(lat=float(base_lat.mean()), miss=float(base_miss.mean()))
        cell['sla'] = dict(lat=float(sla_lat.mean()), miss=float(sla_miss.mean()))
        for name, qm in (('screen', qsel), ('screen_det', 0.0)):
            p = B.backlog_plan(N, R, z=2.0, q_min=qm)
            m, rs = B.agg_backlog(N, R, p)
            lat = np.array([r['lat'] for r in rs])
            miss = np.array([overall_miss(N, s, r) for s, r in enumerate(rs)])
            d0, c0 = boot(lat - base_lat)
            d1, c1 = boot(lat - sla_lat)
            d2, c2 = boot(100 * (miss - sla_miss))
            cell[name] = dict(plan={k: v for k, v in p.items()}, lat=float(lat.mean()), miss=float(miss.mean()),
                              mS=m['mS'], mB=m['mB'], offB=m['offB'], maxl=m['maxl'], p95=m['p95'],
                              per_seed=[dict(lat=float(a), miss=float(b)) for a, b in zip(lat, miss)],
                              vs_plan=[d0, c0], vs_sla=[d1, c1], vs_sla_miss_pp=[d2, c2])
            th = 'none' if p['theta'] is None else p['theta']
            md.append(f"| {N} | {name} | {th}, {p['q']:.3f}, {p['x_B']:.4f}, {p['x_S']:.4f} | {lat.mean():.2f} | "
                      f"{100 * miss.mean():.3f} | {d0:+.2f} [{c0[0]:+.2f}, {c0[1]:+.2f}] | {d1:+.2f} [{c1[0]:+.2f}, {c1[1]:+.2f}] | "
                      f"{d2:+.3f} [{c2[0]:+.3f}, {c2[1]:+.3f}] |")
            print(md[-1], flush=True)
        md.append(f"| {N} | planned prior (E13) | | {base_lat.mean():.2f} | {100 * base_miss.mean():.3f} | | | |")
        md.append(f"| {N} | SLA-MA-PPO (E16) | | {sla_lat.mean():.2f} | {100 * sla_miss.mean():.3f} | | | |")
        out['cells'][str(N)] = cell
    json.dump(out, open(ROOT / 'results' / 'res_backlog.json', 'w'), indent=1)
    open(ROOT / 'results' / 'backlog_summary.md', 'w').write('\n'.join(md) + '\n')


if __name__ == '__main__':
    main()
