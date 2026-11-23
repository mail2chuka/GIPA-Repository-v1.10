"""
compile_sla.py
==============
Experiment E16 summary: SLA-MA-PPO (results/res_sla.json) against the
archived planned policies on the same 30 evaluation seeds.

Comparators (all archived, unchanged):
  homog, idle   results/res_planner.json           plan_z2 (planned prior), gipa (heuristic), local
                results/res_mappo.json             MAPPO (prior mask at 80/160, deadline mask at 240/320)
  homog, own    results/carry_own/res_planner.json plan_z2, gipa, local
  het50, idle   results/res_hetero.json            gipa_plan (GIPA-A), plan_rank (GIPA-B), plan, gipa, local
  het50, own    results/carry_own/res_hetero.json  the same

SLA-MA-PPO's per-seed value is the mean over its training runs. Paired
differences (GIPA minus SLA-MA-PPO) use 10,000 percentile-bootstrap
resamples of the 30 seeds (generator 20260804). Writes
results/sla_summary.json and results/sla_summary.md.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gipa_core import Env, REG  # noqa: E402

RES = ROOT / 'results'
NS = ('80', '160', '240', '320')


def overall_miss(N, seed, mS, mB):
    S = Env(int(N), seed, REG['A']).isS[:int(N)]
    return (S.sum() * mS + (~S).sum() * mB) / int(N)


def boot(d, B=10000, seed=20260804):
    d = np.asarray(d, float)
    rg = np.random.default_rng(seed)
    idx = rg.integers(0, len(d), (B, len(d)))
    m = d[idx].mean(1)
    return float(d.mean()), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def homog_arms(carry, N):
    f = RES / ('carry_own/res_planner.json' if carry == 'own' else 'res_planner.json')
    ps = json.load(open(f))['per_seed']['A'][N]
    arms = {}
    for k, name in (('plan_z2', 'Planned prior'), ('gipa', 'GIPA, heuristic prior'), ('local', 'All-local')):
        arms[name] = dict(lat=[r['lat'] for r in ps[k]],
                          miss=[overall_miss(N, s, r['mS'], r['mB']) for s, r in enumerate(ps[k])])
    if carry == 'idle':
        m = json.load(open(RES / 'res_mappo.json'))
        g = 'affine|base|central|A' if N in ('80', '160') else 'affine|base|central|A|deadline'
        runs = [r for r in m['groups'][g] if str(r['N_train']) == N]
        per = [r['eval']['per_seed'] for r in runs]
        arms['MAPPO'] = dict(lat=list(np.mean([[x['lat'] for x in p] for p in per], 0)),
                             miss=list(np.mean([[overall_miss(N, s, x['mS'], x['mB']) for s, x in enumerate(p)]
                                                for p in per], 0)))
    return arms, 'Planned prior'


def het_arms(carry, N):
    f = RES / ('carry_own/res_hetero.json' if carry == 'own' else 'res_hetero.json')
    ps = json.load(open(f))['per_seed']
    arms = {}
    for k, name in (('gipa_plan', 'GIPA-A (operator on plan)'), ('plan_rank', 'GIPA-B (ranking on plan)'),
                    ('plan', 'Plan sampled per class'), ('gipa', 'GIPA, heuristic prior'), ('local', 'All-local')):
        rs = ps[f'het50|{N}|{k}']
        arms[name] = dict(lat=[r['lat'] for r in rs], miss=[r['miss'] for r in rs])
    return arms, None


def main():
    sla = json.load(open(RES / 'res_sla.json'))
    out, md = {}, ['# E16: SLA-MA-PPO against the planned policies', '',
                   'Mean latency (ms) and overall deadline misses over the 30 archived seeds. SLA-MA-PPO: mean of '
                   'its training runs (range of run means in brackets). Paired difference = GIPA arm minus '
                   'SLA-MA-PPO, 95% bootstrap CI; negative favours GIPA.', '']
    for key in ('homog|idle', 'homog|own', 'het50|idle', 'het50|own'):
        if key not in sla['groups']:
            continue
        setting, carry = key.split('|')
        md += [f'## {setting}, {"carried-over local work (own)" if carry == "own" else "idle on-board processor"}', '',
               '| N | Method | Mean latency (ms) | Misses (%) | Paired diff. latency (ms) | Paired diff. misses (pp) |',
               '|---|---|---|---|---|---|']
        out[key] = {}
        for N in NS:
            runs = sla['groups'][key].get(N)
            if not runs:
                continue
            lat = np.mean([[x['lat'] for x in r['per_seed']] for r in runs], 0)
            miss = np.mean([[x['miss'] for x in r['per_seed']] for r in runs], 0)
            arms, ref = homog_arms(carry, N) if setting == 'homog' else het_arms(carry, N)
            if ref is None:   # heterogeneous: the better GIPA arm (lower mean latency) in this cell
                a, b = 'GIPA-A (operator on plan)', 'GIPA-B (ranking on plan)'
                ref = a if np.mean(arms[a]['lat']) <= np.mean(arms[b]['lat']) else b
            rng_ = [min(r['mean']['lat'] for r in runs), max(r['mean']['lat'] for r in runs)]
            cell = dict(sla=dict(lat=float(lat.mean()), miss=float(miss.mean()), lat_range=rng_, runs=len(runs),
                                 plateau=sum(r['plateau'] for r in runs),
                                 off=float(np.mean([r['mean']['off'] for r in runs])),
                                 fallback=float(np.mean([r['mean']['fallback'] for r in runs]))),
                        ref=ref, arms={})
            md.append(f'| {N} | SLA-MA-PPO [24] | {lat.mean():.2f} ({rng_[0]:.2f}–{rng_[1]:.2f}) | '
                      f'{100 * miss.mean():.2f} | — | — |')
            for name, v in arms.items():
                dl, ci = boot(np.array(v['lat']) - lat)
                dm, cim = boot(100 * (np.array(v['miss']) - miss))
                cell['arms'][name] = dict(lat=float(np.mean(v['lat'])), miss=float(np.mean(v['miss'])),
                                          diff_lat=dl, ci_lat=ci, diff_miss_pp=dm, ci_miss_pp=cim)
                star = ' **' if name == ref else ''
                md.append(f'| {N} | {name}{star} | {np.mean(v["lat"]):.2f} | {100 * np.mean(v["miss"]):.2f} | '
                          f'{dl:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] | {dm:+.2f} [{cim[0]:+.2f}, {cim[1]:+.2f}] |')
            out[key][N] = cell
        md += ['', '** reference GIPA arm for this cell.', '']
    json.dump(out, open(RES / 'sla_summary.json', 'w'), indent=1)
    open(RES / 'sla_summary.md', 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
