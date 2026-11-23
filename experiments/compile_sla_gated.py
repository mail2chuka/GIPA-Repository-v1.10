"""
compile_sla_gated.py
====================
Experiments E16/E19 summary: SLA-MA-PPO with idealised feedback (E16,
results/sla/), retrained with completion-gated feedback (E19,
results/sla_gated/<setting>_<carry>_N<N>_r<run>.json) and the archived E16
actors evaluated with gated feedback (results/sla_gated/reeval_*), each against
fixed GIPA variants on the same 30 evaluation seeds.

GIPA variants (archived per-seed results)
  identical tasks, idle    planned prior          results/res_planner.json (plan_z2)
  identical tasks, own     planned prior          results/carry_own/res_planner.json (plan_z2)
                           backlog screening      results/res_backlog.json (screen, E17)
  heterogeneous, idle/own  per-task ranking       results[/carry_own]/res_hetero.json (plan_rank)
                           amplification          the same (gipa_plan)

Intervals for the paired difference GIPA minus SLA-MA-PPO (negative favours GIPA):
  seeds    10,000 percentile-bootstrap resamples of the 30 evaluation seeds, SLA-MA-PPO averaged over its runs
           (the intervals of E16);
  two-level  10,000 resamples that draw the training runs (with replacement) and, independently, the 30 seeds
           (with replacement); the difference is the mean over drawn seeds of GIPA minus the mean of the drawn
           runs, so it includes variation across training runs.
Generator 20260804. Writes results/sla_gated_summary.json and results/sla_gated_summary.md.
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))
from compile_sla import overall_miss  # noqa: E402

RES = ROOT / 'results'
NS = ('80', '160', '240', '320')
B = 10000


def runs(pattern):
    fs = sorted(glob.glob(str(pattern)))
    lat = np.array([[x['lat'] for x in json.load(open(f))['per_seed']] for f in fs])
    miss = np.array([[x['miss'] for x in json.load(open(f))['per_seed']] for f in fs])
    return lat, miss, len(fs)


def gipa_arms(setting, carry, N):
    arms = {}
    if setting == 'homog':
        f = RES / ('carry_own/res_planner.json' if carry == 'own' else 'res_planner.json')
        ps = json.load(open(f))['per_seed']['A'][N]['plan_z2']
        arms['planned prior'] = (np.array([r['lat'] for r in ps]),
                                 np.array([overall_miss(N, s, r['mS'], r['mB']) for s, r in enumerate(ps)]))
        if carry == 'own':
            ps = json.load(open(RES / 'res_backlog.json'))['cells'][N]['screen']['per_seed']
            arms['backlog screening'] = (np.array([r['lat'] for r in ps]), np.array([r['miss'] for r in ps]))
    else:
        f = RES / ('carry_own/res_hetero.json' if carry == 'own' else 'res_hetero.json')
        ps = json.load(open(f))['per_seed']
        for k, name in (('plan_rank', 'per-task ranking'), ('gipa_plan', 'amplification')):
            rs = ps[f'het50|{N}|{k}']
            arms[name] = (np.array([r['lat'] for r in rs]), np.array([r['miss'] for r in rs]))
    return arms


def ci_seeds(g, S, rg):
    d = g - S.mean(0)
    idx = rg.integers(0, len(d), (B, len(d)))
    m = d[idx].mean(1)
    return float(d.mean()), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def ci_two_level(g, S, rg):
    R, n = S.shape
    out = np.empty(B)
    for b in range(B):
        r = rg.integers(0, R, R)
        s = rg.integers(0, n, n)
        out[b] = (g[s] - S[np.ix_(r, s)].mean(0)).mean()
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def main():
    out, md = {}, ['# SLA-MA-PPO with idealised and completion-gated feedback against fixed GIPA variants', '',
                   'Mean latency (ms) and overall misses (%) over the 30 evaluation seeds; SLA-MA-PPO: mean of its '
                   'runs. Differences GIPA minus SLA-MA-PPO with 95% intervals: seeds only / two-level (runs and '
                   'seeds). Negative favours GIPA.', '']
    for setting in ('homog', 'het50'):
        for carry in ('idle', 'own'):
            key = f'{setting}|{carry}'
            out[key] = {}
            md += [f'## {setting}, {"carried-over local work" if carry == "own" else "idle processor"}', '',
                   '| N | SLA-MA-PPO feedback | Runs | SLA lat. | SLA miss | GIPA variant | GIPA lat. | GIPA miss | '
                   'Δ latency [seeds] [two-level] | Δ misses pp [seeds] [two-level] |',
                   '|---|---|---|---|---|---|---|---|---|---|']
            for N in NS:
                cell = {}
                arms = gipa_arms(setting, carry, N)
                for fb, pat in (('ideal', RES / 'sla' / f'{setting}_{carry}_N{N}_r*.json'),
                                ('gated', RES / 'sla_gated' / f'{setting}_{carry}_N{N}_r*.json'),
                                ('gated, archived actors', RES / 'sla_gated' / f'reeval_{setting}_{carry}_N{N}_r*.json')):
                    SL, SM, n = runs(pat)
                    if not n:
                        continue
                    c = dict(runs=n, lat=float(SL.mean()), miss=float(SM.mean()),
                             run_lat=[float(x) for x in SL.mean(1)], arms={})
                    for name, (gl, gm) in arms.items():
                        rg = np.random.default_rng(20260804)
                        dl, cl = ci_seeds(gl, SL, rg)
                        cl2 = ci_two_level(gl, SL, rg)
                        dm, cm = ci_seeds(100 * gm, 100 * SM, rg)
                        cm2 = ci_two_level(100 * gm, 100 * SM, rg)
                        c['arms'][name] = dict(lat=float(gl.mean()), miss=float(gm.mean()), d_lat=dl, ci_lat=cl,
                                               ci_lat_2l=cl2, d_miss_pp=dm, ci_miss=cm, ci_miss_2l=cm2)
                        md.append(f'| {N} | {fb} | {n} | {SL.mean():.2f} | {100 * SM.mean():.2f} | {name} | '
                                  f'{gl.mean():.2f} | {100 * gm.mean():.2f} | {dl:+.2f} [{cl[0]:+.2f}, {cl[1]:+.2f}] '
                                  f'[{cl2[0]:+.2f}, {cl2[1]:+.2f}] | {dm:+.2f} [{cm[0]:+.2f}, {cm[1]:+.2f}] '
                                  f'[{cm2[0]:+.2f}, {cm2[1]:+.2f}] |')
                    cell[fb] = c
                out[key][N] = cell
            md.append('')
    json.dump(out, open(RES / 'sla_gated_summary.json', 'w'), indent=1)
    open(RES / 'sla_gated_summary.md', 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
