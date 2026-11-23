"""
exp_sla.py
==========
Experiment E16: SLA-MA-PPO (Georgiades et al., IEEE TNSM 2026), a recent
task-specific vehicular DRL method, against GIPA's planned policies.

Stages
------
  tune   learning rate x beta_tau grid at N = 160, one training run each, on the
         tuning seeds 2000-2009, for one setting ('homog' or 'het50', idle
         processor); writes results/sla/tune_<setting>.json. Rule: lowest mean
         latency among configurations whose miss rate is within 0.5
         percentage points of the lowest (the rule of Section VI-E).
  train  trains and evaluates runs for one setting, densities and run seeds;
         writes results/sla/<setting>_<carry>_N<N>_r<seed>.json (per-seed
         evaluation, learning curve, trained actor).
  merge  combines every run file into results/res_sla.json.
  reeval evaluates the archived actors of results/sla (one setting, densities,
         run seeds) under the feedback timing set by GIPA_SLA_FEEDBACK and
         writes per-seed results to --outdir (E19: archived actors with
         completion-gated feedback).

Feedback timing: GIPA_SLA_FEEDBACK=gated trains and evaluates with
completion-gated feedback (E19, see src/sla_mappo.py); use --outdir
results/sla_gated so the idealised-feedback runs of E16 are kept.

The carried-over-work runs are made by setting GIPA_LOCAL_CARRY=own (see
run_e16.sh). Run from the repository root with src on PYTHONPATH.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import sla_mappo as S  # noqa: E402
from gipa_core import LOCAL_CARRY  # noqa: E402

OUT = ROOT / 'results' / 'sla'
DENSITIES = (80, 160, 240, 320)
GRID = [dict(lr=lr, beta_tau=bt, beta_risk=0.5) for lr in (3e-3, 3e-4) for bt in (0.5, 2.0)]


def carry_tag():
    return LOCAL_CARRY or 'idle'


def stage_tune(setting, iters):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for setting in (setting,):
        for g in GRID:
            t0 = time.time()
            r = S.train(setting, 160, 0, iters=iters, **g)
            m, _ = S.evaluate(r['actor'], setting, 160, seeds=S.TUNE_SEEDS, cfg=g)
            rows.append(dict(setting=setting, **g, lat=m['lat'], miss=m['miss'], plateau=r['plateau'],
                             chosen_iter=r['chosen_iter'], seconds=time.time() - t0))
            print(f"tune {setting} {g}  lat {m['lat']:.3f}  miss {100 * m['miss']:.2f}%  "
                  f"plateau {r['plateau']}  {time.time() - t0:.0f}s", flush=True)
    lo = min(r['miss'] for r in rows)
    ok = [r for r in rows if r['miss'] <= lo + 0.005]
    best = min(ok, key=lambda r: r['lat'])
    chosen = dict(lr=best['lr'], beta_tau=best['beta_tau'], beta_risk=best['beta_risk'])
    json.dump(dict(setting=setting, grid=rows, chosen=chosen,
                   rule='lowest latency with miss rate within 0.5 pp of the lowest',
                   density=160, seeds=list(S.TUNE_SEEDS), iterations=iters),
              open(OUT / f'tune_{setting}.json', 'w'), indent=1)
    print('chosen', chosen)


def chosen_cfg(setting):
    return json.load(open(ROOT / 'results' / 'sla' / f'tune_{setting}.json'))['chosen']


def stage_train(setting, densities, seeds, iters):
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = chosen_cfg(setting)
    for N in densities:
        for s in seeds:
            f = OUT / f'{setting}_{carry_tag()}_N{N}_r{s}.json'
            if f.exists():
                print('exists', f.name)
                continue
            t0 = time.time()
            r = S.train(setting, N, s, iters=iters, **cfg)
            m, per = S.evaluate(r['actor'], setting, N, cfg=cfg)
            json.dump(dict(setting=setting, carry=carry_tag(), feedback=S.FEEDBACK, N=N, run_seed=s, cfg=cfg,
                           iterations=iters,
                           mean=m, per_seed=per, plateau=r['plateau'], plateau_rel_change=r['plateau_rel_change'],
                           chosen_iter=r['chosen_iter'], checkpoints=r['checkpoints'],
                           train_seconds=r['seconds'], curve=r['curve'], actor=r['actor'].state()),
                      open(f, 'w'))
            print(f"{setting} {carry_tag()} N={N} run {s}: lat {m['lat']:.3f}  miss {100 * m['miss']:.3f}%  "
                  f"off {m['off']:.3f}  plateau {r['plateau']}  train {r['seconds']:.0f}s  total {time.time() - t0:.0f}s",
                  flush=True)


def stage_reeval(setting, densities, seeds):
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = chosen_cfg(setting)
    for N in densities:
        for s in seeds:
            src = ROOT / 'results' / 'sla' / f'{setting}_{carry_tag()}_N{N}_r{s}.json'
            f = OUT / f'reeval_{setting}_{carry_tag()}_N{N}_r{s}.json'
            if f.exists():
                print('exists', f.name)
                continue
            d = json.load(open(src))
            actor = S.MLP.from_state(d['actor'])
            m, per = S.evaluate(actor, setting, N, cfg=cfg)
            json.dump(dict(setting=setting, carry=carry_tag(), feedback=S.FEEDBACK, N=N, run_seed=s, cfg=cfg,
                           source=src.name, mean=m, per_seed=per), open(f, 'w'))
            print(f"reeval {setting} {carry_tag()} {S.FEEDBACK} N={N} run {s}: lat {m['lat']:.3f} "
                  f"(archived {d['mean']['lat']:.3f})  miss {100 * m['miss']:.3f}% (archived {100 * d['mean']['miss']:.3f}%)",
                  flush=True)


def stage_merge():
    groups = {}
    for f in sorted(OUT.glob('*_N*_r*.json')):
        d = json.load(open(f))
        groups.setdefault(f"{d['setting']}|{d['carry']}", {}).setdefault(str(d['N']), []).append(
            dict(run_seed=d['run_seed'], mean=d['mean'], per_seed=d['per_seed'], plateau=d['plateau'],
                 train_seconds=d['train_seconds']))
    summary = {}
    for g, byN in groups.items():
        summary[g] = {}
        for N, runs in byN.items():
            keys = runs[0]['mean'].keys()
            summary[g][N] = dict(runs=len(runs), plateau=sum(r['plateau'] for r in runs),
                                 **{k: float(np.mean([r['mean'][k] for r in runs])) for k in keys},
                                 lat_range=[min(r['mean']['lat'] for r in runs), max(r['mean']['lat'] for r in runs)])
    json.dump(dict(summary=summary, groups=groups, tune={st: json.load(open(OUT / f'tune_{st}.json')) for st in ('homog', 'het50')}),
              open(ROOT / 'results' / 'res_sla.json', 'w'))
    for g in sorted(summary):
        for N in sorted(summary[g], key=int):
            s = summary[g][N]
            print(f"{g:14s} N={N:>3s}  lat {s['lat']:.2f} ({s['lat_range'][0]:.2f}-{s['lat_range'][1]:.2f})  "
                  f"miss {100 * s['miss']:.2f}%  S {100 * s['mS']:.2f}%  B {100 * s['mB']:.2f}%  off {s['off']:.3f}  "
                  f"runs {s['runs']} plateau {s['plateau']}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['tune', 'train', 'merge', 'reeval'])
    ap.add_argument('--setting', choices=['homog', 'het50'])
    ap.add_argument('--densities', type=int, nargs='+', default=list(DENSITIES))
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--iters', type=int, default=400)
    ap.add_argument('--outdir', default=None, help='write run files here instead of results/sla (sensitivity runs)')
    a = ap.parse_args()
    if a.outdir:
        OUT = ROOT / a.outdir
    if a.stage == 'tune':
        stage_tune(a.setting, a.iters)
    elif a.stage == 'train':
        stage_train(a.setting, a.densities, a.seeds, a.iters)
    elif a.stage == 'reeval':
        stage_reeval(a.setting, a.densities, a.seeds)
    else:
        stage_merge()
