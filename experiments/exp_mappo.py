"""
exp_mappo.py
============
Workplan experiment E7: the multi-agent DRL baseline (MAPPO), answering
Reviewer 1 comment 4. See src/mappo.py for the algorithm.

STATUS: written for the first revision of manuscript ITS-2026-08-0579. Not
part of the submitted v1.0.0 results.

Stages
------
bench   Time a short training run and extrapolate the full plan's runtime.
tune    Fixed grid of 6 configurations (reward in {team, mixed, individual}
        x actor learning rate in {3e-4, 1e-3}), one training run each at
        N = 80, evaluated on the tuning seeds (indices 2000-2009, disjoint
        from the 30 archived evaluation seeds). Selection: the lowest mean
        latency among configurations with a total miss rate <= 0.1%.
train   Train the chosen configuration for each run seed and training
        density, then evaluate on the 30 archived seeds at the training
        density. Models trained at N = 160 are also evaluated at every
        density (generalisation). Learning curves, plateau verdicts and the
        trained networks are saved.
merge   Combine the part files written by parallel ``train`` processes.

Typical use on a 2-core laptop (see train_mappo.sh, which does all of this)
---------------------------------------------------------------------------
    PYTHONPATH=src python experiments/exp_mappo.py --stage bench
    PYTHONPATH=src python experiments/exp_mappo.py --stage tune
    PYTHONPATH=src python experiments/exp_mappo.py --stage train --seeds 0 1 2 --part a &
    PYTHONPATH=src python experiments/exp_mappo.py --stage train --seeds 3 4   --part b &
    wait
    PYTHONPATH=src python experiments/exp_mappo.py --stage merge

Options
-------
--model affine|ps       latency model (submitted affine model, or Eq. 5 from E2)
--variant base|load     'load' adds previous-slot RSU loads to the actor input
                        (uses gamma = 0.9 with GAE lambda = 0.95)
--critic central|local  'local' gives the IPPO fallback
--mask prior|deadline   'prior' (default) is GIPA's prior-inclusive deadline
                        mask; 'deadline' uses uncongested latency only, for a
                        fair comparison at N >= 240 where GIPA's mask forbids
                        every RSU
--out                   merge output path (default results/res_mappo.json)
--densities ...         training densities (default 80 160 240 320)
--iters                 training iterations (episodes) per run (default 400)
--regime A|B            workload regime (default A)

Outputs (results/mappo/)
------------------------
tune.json                          tuning grid, scores and the chosen config
train_<model>_<variant>_<critic>_<regime>[_dmask]_<part>.json   one per train
                                   process ('_dmask' marks the deadline mask)
results/res_mappo.json             merged: config, per-run curves and plateau
                                   verdicts, evaluation means and per-seed
                                   records, trained network parameters

Seeds
-----
Training environments: seed indices 1000-1049 (environment RNG 2000-2049).
Training sampling and initialisation: 8000 + run seed. Tuning evaluation:
seed indices 2000-2009. Final evaluation: the 30 archived seeds (environment
1000-1029, action sampling 5000-5029), paired with every other policy.
"""

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np

from gipa_core import REG
import mappo

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / 'results' / 'mappo'
GRID = [dict(reward=r, lr=lr) for r in ['team', 'mixed', 'individual'] for lr in [3e-4, 1e-3]]
ALL_DENSITIES = [80, 160, 240, 320]


def _clean(d):
    """Replace NaN with None so the output is strict JSON."""
    return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in d.items()}


def _variant_cfg(args):
    """Configuration keys implied by the command-line variant options."""
    cfg = dict(model=args.model, variant=args.variant, critic=args.critic, mask=args.mask)
    if args.variant == 'load':
        cfg.update(gamma=0.9, gae_lambda=0.95)
    return cfg


def stage_bench(args):
    """Time 30 iterations at N = 160 and extrapolate the default plan."""
    R = REG[args.regime]
    res = mappo.train(160, R, 0, iters=30, **_variant_cfg(args))
    spi = res['seconds_per_iter']
    runs = len(args.seeds) * len(args.densities)
    total = runs * args.iters * spi
    tune = len(GRID) * 300 * spi * 0.5          # N = 80 iterations are about half as long
    print(f'{spi:.3f} s per iteration at N = 160 on this machine')
    print(f'tuning (6 configs x 300 iterations at N = 80): about {tune / 60:.0f} min')
    print(f'training ({runs} runs x {args.iters} iterations, one process): about {total / 60:.0f} min; '
          f'about {total / 120:.0f} min with two processes in parallel')


def stage_tune(args):
    """Run the 6-configuration grid at N = 80 and write tune.json."""
    R = REG[args.regime]
    tune_seeds = range(mappo.TUNE_SEED_INDEX, mappo.TUNE_SEED_INDEX + 10)
    rows = []
    for g in GRID:
        t0 = time.time()
        res = mappo.train(80, R, 100, iters=300, **dict(_variant_cfg(args), **g))
        m, _ = mappo.evaluate(res['actor'], 80, R, tune_seeds, args.variant, args.model, args.mask)
        miss = m['mS'] * 0.3 + m['mB'] * 0.7
        rows.append(dict(config=g, lat=m['lat'], miss=miss, plateau=res['plateau'],
                         minutes=(time.time() - t0) / 60))
        print(f"  reward {g['reward']:10s} lr {g['lr']:.0e}: lat {m['lat']:6.2f} ms, "
              f"miss {100 * miss:5.2f}%, plateau {res['plateau']}", flush=True)
    ok = [r for r in rows if r['miss'] <= 0.001]
    best = min(ok, key=lambda r: r['lat']) if ok else min(rows, key=lambda r: r['miss'])
    OUTDIR.mkdir(parents=True, exist_ok=True)
    with open(OUTDIR / 'tune.json', 'w') as fh:
        json.dump(dict(grid=rows, chosen=best['config'], rule='lowest latency with miss <= 0.1%',
                       tuning_density=80, tuning_seed_indices=list(tune_seeds), iterations=300), fh, indent=1)
    print('chosen:', best['config'])


def _chosen():
    """The tuned configuration, or the module defaults if tuning has not run."""
    p = OUTDIR / 'tune.json'
    if p.exists():
        return json.load(open(p))['chosen']
    print('note: results/mappo/tune.json not found; using module defaults (team reward, lr 1e-3)')
    return {}


def stage_train(args):
    """Train and evaluate every (seed, density) pair given; write one part file."""
    R = REG[args.regime]
    cfg = dict(_chosen(), **_variant_cfg(args))
    runs = []
    for N in args.densities:
        for s in args.seeds:
            res = mappo.train(N, R, s, iters=args.iters, log_every=args.log_every, **cfg)
            m, rs = mappo.evaluate(res['actor'], N, R, None, args.variant, args.model, args.mask)
            gen = {}
            if N == 160:
                for N2 in ALL_DENSITIES:
                    if N2 != 160:
                        g, grs = mappo.evaluate(res['actor'], N2, R, None, args.variant, args.model, args.mask)
                        gen[N2] = dict(mean=_clean(g), per_seed=[_clean(x) for x in grs])
            runs.append(dict(N_train=N, run_seed=s, config=res['config'], plateau=res['plateau'],
                             plateau_rel_change=res['plateau_rel_change'], seconds=res['seconds'],
                             curve=res['curve'], eval=dict(mean=_clean(m), per_seed=[_clean(x) for x in rs]),
                             generalisation=gen, actor=res['actor'].state()))
            print(f"N={N:3d} seed {s}: lat {m['lat']:6.2f} p95 {m['p95']:5.1f} "
                  f"miss S/B {100 * m['mS']:5.2f}/{100 * m['mB']:5.2f}% off S/B {m['offS']:.2f}/{m['offB']:.2f} "
                  f"plateau {res['plateau']} ({res['seconds'] / 60:.1f} min)", flush=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    dm = '_dmask' if args.mask == 'deadline' else ''
    name = f'train_{args.model}_{args.variant}_{args.critic}_{args.regime}{dm}_{args.part}.json'
    with open(OUTDIR / name, 'w') as fh:
        json.dump(dict(regime=args.regime, model=args.model, variant=args.variant,
                       critic=args.critic, mask=args.mask, runs=runs), fh)
    print('wrote', OUTDIR / name)


def stage_merge(args):
    """Merge every train_*.json part file into one results file (default
    results/res_mappo.json), with a summary. Group keys are
    model|variant|critic|regime, with '|deadline' appended for runs that used
    the deadline-only mask."""
    parts = sorted(glob.glob(str(OUTDIR / 'train_*.json')))
    groups = {}
    for p in parts:
        d = json.load(open(p))
        key = f"{d['model']}|{d['variant']}|{d['critic']}|{d['regime']}"
        if d.get('mask', 'prior') == 'deadline':
            key += '|deadline'
        groups.setdefault(key, []).extend(d['runs'])
    summary = {}
    for key, runs in groups.items():
        summary[key] = {}
        for N in sorted({r['N_train'] for r in runs}):
            rs = [r for r in runs if r['N_train'] == N]
            lat = [r['eval']['mean']['lat'] for r in rs]
            summary[key][N] = dict(n_runs=len(rs), lat_mean=float(np.mean(lat)),
                                   lat_min=float(np.min(lat)), lat_max=float(np.max(lat)),
                                   mS_mean=float(np.mean([r['eval']['mean']['mS'] for r in rs])),
                                   mB_mean=float(np.mean([r['eval']['mean']['mB'] for r in rs])),
                                   plateau_all=bool(all(r['plateau'] for r in rs)))
            print(f"{key}  N={N:3d}: {len(rs)} runs, mean latency {np.mean(lat):.2f} ms "
                  f"(range {np.min(lat):.2f}-{np.max(lat):.2f}), all plateaued: {summary[key][N]['plateau_all']}")
    tune = json.load(open(OUTDIR / 'tune.json')) if (OUTDIR / 'tune.json').exists() else None
    out = Path(args.out) if args.out else ROOT / 'results' / 'res_mappo.json'
    with open(out, 'w') as fh:
        json.dump(dict(tune=tune, summary=summary, groups=groups, part_files=[os.path.basename(p) for p in parts]), fh)
    print('wrote', out)


def main():
    """Parse arguments and run one stage."""
    ap = argparse.ArgumentParser(description='E7 MAPPO baseline')
    ap.add_argument('--stage', choices=['bench', 'tune', 'train', 'merge'], required=True)
    ap.add_argument('--model', choices=['affine', 'ps'], default='affine')
    ap.add_argument('--variant', choices=['base', 'load'], default='base')
    ap.add_argument('--critic', choices=['central', 'local'], default='central')
    ap.add_argument('--mask', choices=['prior', 'deadline'], default='prior')
    ap.add_argument('--out', default=None, help='merge output path')
    ap.add_argument('--regime', choices=['A', 'B'], default='A')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--densities', type=int, nargs='+', default=ALL_DENSITIES)
    ap.add_argument('--iters', type=int, default=400)
    ap.add_argument('--part', default='a')
    ap.add_argument('--log-every', type=int, default=0)
    args = ap.parse_args()
    {'bench': stage_bench, 'tune': stage_tune, 'train': stage_train, 'merge': stage_merge}[args.stage](args)


if __name__ == '__main__':
    main()
