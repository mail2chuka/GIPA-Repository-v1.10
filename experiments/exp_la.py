"""
exp_la.py
=========
Workplan experiment E6: learning-automata baselines (Reviewer 1 comment 3).
See src/learning_automata.py for the automata, responses and seeds.

STATUS: written for the first revision of manuscript ITS-2026-08-0579. Not
part of the submitted v1.0.0 results.

What it runs
------------
1. Self-check of the update rules (``check_updates``).
2. Tuning grid on tuning seed indices 4500-4509 (never used for
   evaluation): 3 algorithms (L_R-I, L_R-eP, pursuit) x 2 responses
   (binary, S-model) x 3 rates (0.01, 0.05, 0.10), for regimes A and B,
   latency models 'affine' and 'ps' (Eq. 5), warm-up 25 and 500 slots, at
   N = 80, 160, 240 and 320.
   Selection rule (per regime, model, warm-up and algorithm, at the tuning
   density N = 160): keep the settings whose overall miss rate is within
   0.5 percentage points of the lowest, then take the lowest mean latency.
   The same rule picks the best LA overall ('best'), and, as a sensitivity
   check, the best LA separately at each density ('best_perdensity').
3. Main evaluation on the 30 archived seeds for every tuned configuration at
   every density, with references computed on the same seeds: GIPA
   (affine: gipa_core.run with the heuristic prior; ps: calibration.run_ps
   with the class prior), the planner with margin z = 2 (where feasible),
   and all-local. Paired differences LA - GIPA and LA - planner with a
   seed-level percentile bootstrap (10,000 resamples, seed 20260804).
   Learning curves (mean latency per 25-slot block) for warm-up 500.
   One reference row with GIPA's prior-inclusive mask (affine, Regime A).
4. Density-change test (Regime A, affine): N = 80 for slots 0-499, 240 for
   500-999, 80 for 1000-1499 (vehicles 81-240 arrive at slot 500 and leave
   at slot 1000). Policies: each LA algorithm with its tuned warm-up-500
   settings, GIPA with the prior refreshed to the current N, GIPA with the
   stale N = 80 prior, the planner re-planned or stale, and all-local.
   Per-slot traces averaged over the 30 seeds, and per phase: transient
   (first 50 slots), steady state (last 100 slots) and recovery time (first
   slot after which the 20-slot rolling mean latency stays within 0.2 ms of
   the steady value and the rolling miss rate within 0.5 points).

Output
------
results/res_la.json (path configurable with --out): "check", "tuning",
"chosen", "main", "per_seed", "density_change", "seeds".

Runtime
-------
About 5-10 minutes on a laptop CPU with two worker processes.

Run (from the repository root)
------------------------------
    PYTHONPATH=src python experiments/exp_la.py
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from gipa_core import CFG, REG, run as run_core
from learning_automata import (ALGOS, RATES, RESPONSES, TUNE_SEED_INDEX, N_TUNE_SEEDS,
                               agg_la, check_updates, run_schedule)
from planner import plan_allocation, run_planned

ROOT = Path(__file__).resolve().parents[1]
DENSITIES = [80, 160, 240, 320]
WARMS = [25, 500]
MODELS = ['affine', 'ps']
TUNE_N = 160
TUNE_SEEDS = list(range(TUNE_SEED_INDEX, TUNE_SEED_INDEX + N_TUNE_SEEDS))
MISS_TOL = 0.005
BOOT_REPS, BOOT_SEED = 10000, 20260804
SCHEDULE = [80] * 500 + [240] * 500 + [80] * 500
CHANGES = [0, 500, 1000]


def _clean(d):
    return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in d.items()}


def overall_miss(m):
    """Overall miss rate from the class rates (safety share PI_S)."""
    return CFG['PI_S'] * m['mS'] + (1 - CFG['PI_S']) * m['mB']


def choose(rows):
    """Selection rule: overall miss within MISS_TOL of the lowest, then lowest latency."""
    best_miss = min(overall_miss(r) for r in rows)
    ok = [r for r in rows if overall_miss(r) <= best_miss + MISS_TOL]
    return min(ok, key=lambda r: r['lat'])


def paired_ci(x, y):
    """Mean of x - y over seeds with a percentile bootstrap 95% CI."""
    d = np.asarray(x, float) - np.asarray(y, float)
    rg = np.random.default_rng(BOOT_SEED)
    bs = d[rg.integers(0, len(d), (BOOT_REPS, len(d)))].mean(1)
    return dict(mean=float(d.mean()), ci95=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))])


# ----------------------------------------------------------------------
# Worker jobs
# ----------------------------------------------------------------------
def job_tune(spec):
    reg, model, warm, algo, resp, rate, N = spec
    m, _ = agg_la(algo, resp, rate, N, REG[reg], seeds=TUNE_SEEDS, model=model, warm=warm)
    return dict(regime=reg, model=model, warm=warm, algo=algo, resp=resp, rate=rate, N=N, **_clean(m))


def job_eval(spec):
    reg, model, warm, algo, resp, rate, N, mask = spec
    m, rs = agg_la(algo, resp, rate, N, REG[reg], model=model, warm=warm, mask=mask,
                   curve_every=25 if warm == 500 else 0)
    return dict(spec=spec, mean=_clean(m), per_seed=[_clean({k: v for k, v in r.items() if k != 'curve'}) for r in rs])


def job_ref(spec):
    """Reference policies on the 30 archived seeds: GIPA, planner (z = 2), all-local."""
    reg, model, N = spec
    R = REG[reg]
    out = {}
    if model == 'affine':
        out['gipa'] = [_clean(run_core('gipa', N, s, R)) for s in range(CFG['NSEED'])]
        out['local'] = [_clean(run_core('local', N, s, R)) for s in range(CFG['NSEED'])]
        plan = plan_allocation(N, R, z=2.0)
        if plan is not None:
            out['plan'] = [_clean(run_planned(N, s, R, plan)) for s in range(CFG['NSEED'])]
    else:
        from calibration import make_calibration, plan_allocation_ps, run_ps
        cal = make_calibration()
        out['gipa'] = [_clean(run_ps('gipa', N, s, R, cal, prior='class')) for s in range(CFG['NSEED'])]
        out['local'] = [_clean(run_ps('local', N, s, R, cal)) for s in range(CFG['NSEED'])]
        plan = plan_allocation_ps(N, R, cal, z=2.0)
        if plan is not None:
            out['plan'] = [_clean(run_ps('plan', N, s, R, cal, plan=plan)) for s in range(CFG['NSEED'])]
    return dict(spec=spec, refs=out)


def job_schedule(spec):
    pol, la_cfg, seed = spec
    R = REG['A']
    return dict(spec=[pol, la_cfg, seed],
                trace=run_schedule(pol, SCHEDULE, seed, R, 'affine', la_cfg=la_cfg,
                                   plan_fn=lambda N: plan_allocation(N, R, z=2.0)))


# ----------------------------------------------------------------------
# Density-change summaries
# ----------------------------------------------------------------------
def phase_summary(lat, miss):
    """Transient, steady state and recovery time for each phase of SCHEDULE."""
    out = []
    bounds = CHANGES + [len(SCHEDULE)]
    for p in range(len(CHANGES)):
        lo, hi = bounds[p], bounds[p + 1]
        L, Mi = np.asarray(lat[lo:hi]), np.asarray(miss[lo:hi])
        ss_l, ss_m = float(L[-100:].mean()), float(Mi[-100:].mean())
        rl = np.convolve(L, np.ones(20) / 20, 'valid')
        rm = np.convolve(Mi, np.ones(20) / 20, 'valid')
        good = (np.abs(rl - ss_l) <= 0.2) & (rm <= ss_m + 0.005)
        rec = None
        for s in range(len(good)):
            if good[s:].all():
                rec = s
                break
        out.append(dict(N=SCHEDULE[lo], transient_lat=float(L[:50].mean()), transient_miss=float(Mi[:50].mean()),
                        steady_lat=ss_l, steady_miss=ss_m, recovery_slots=rec))
    return out


def main():
    ap = argparse.ArgumentParser(description='E6 learning automata')
    ap.add_argument('--out', type=Path, default=ROOT / 'results' / 'res_la.json')
    ap.add_argument('--workers', type=int, default=2)
    args = ap.parse_args()

    ok = check_updates()
    print('1. update-rule self-check:', 'passed' if ok else 'FAILED', flush=True)

    # 2. Tuning grid
    specs = [(reg, model, warm, algo, resp, rate, N)
             for reg in ['A', 'B'] for model in MODELS for warm in WARMS
             for algo in ALGOS for resp in RESPONSES for rate in RATES for N in DENSITIES]
    with Pool(args.workers) as pool:
        tuning = pool.map(job_tune, specs, chunksize=8)
    print(f'2. tuning grid: {len(tuning)} settings x {N_TUNE_SEEDS} tuning seeds', flush=True)

    chosen = {}
    for reg in ['A', 'B']:
        for model in MODELS:
            for warm in WARMS:
                key = f'{reg}|{model}|{warm}'
                rows = [r for r in tuning if r['regime'] == reg and r['model'] == model and r['warm'] == warm]
                at = [r for r in rows if r['N'] == TUNE_N]
                c = {algo: choose([r for r in at if r['algo'] == algo]) for algo in ALGOS}
                c['best'] = choose(at)
                c['best_perdensity'] = {N: choose([r for r in rows if r['N'] == N]) for N in DENSITIES}
                chosen[key] = c
                print(f'   {key:16s} ' + ' | '.join(f"{a}: {c[a]['resp']} a={c[a]['rate']}" for a in ALGOS)
                      + f" | best: {c['best']['algo']} {c['best']['resp']} a={c['best']['rate']}", flush=True)

    # 3. Main evaluation
    ev_specs = set()
    for key, c in chosen.items():
        reg, model, warm = key.split('|')
        warm = int(warm)
        for algo in ALGOS:
            for N in DENSITIES:
                ev_specs.add((reg, model, warm, algo, c[algo]['resp'], c[algo]['rate'], N, 'deadline'))
        for N in DENSITIES:
            b = c['best_perdensity'][N]
            ev_specs.add((reg, model, warm, b['algo'], b['resp'], b['rate'], N, 'deadline'))
    cA = chosen['A|affine|500']['best']
    for N in DENSITIES:
        ev_specs.add(('A', 'affine', 500, cA['algo'], cA['resp'], cA['rate'], N, 'prior'))
    ev_specs = sorted(ev_specs, key=str)
    ref_specs = [(reg, model, N) for reg in ['A', 'B'] for model in MODELS for N in DENSITIES]
    with Pool(args.workers) as pool:
        evals = pool.map(job_eval, ev_specs, chunksize=2)
        refs = pool.map(job_ref, ref_specs, chunksize=1)
    ev = {tuple(e['spec']): e for e in evals}
    rf = {tuple(r['spec']): r['refs'] for r in refs}

    main_tab, per_seed = {}, {}
    for key, c in chosen.items():
        reg, model, warm = key.split('|')
        warm = int(warm)
        main_tab[key] = {}
        for N in DENSITIES:
            refsN = rf[(reg, model, N)]
            row = {}
            for name in ALGOS + ['best', 'best_perdensity']:
                cc = c['best_perdensity'][N] if name == 'best_perdensity' else c[name]
                e = ev[(reg, model, warm, cc['algo'], cc['resp'], cc['rate'], N, 'deadline')]
                lat = [r['lat'] for r in e['per_seed']]
                row[name] = dict(config=dict(algo=cc['algo'], resp=cc['resp'], rate=cc['rate']), **e['mean'],
                                 minus_gipa=paired_ci(lat, [r['lat'] for r in refsN['gipa']]),
                                 minus_plan=(paired_ci(lat, [r['lat'] for r in refsN['plan']]) if 'plan' in refsN else None))
                per_seed[f'{key}|{N}|{name}'] = e['per_seed']
            for pol, rs in refsN.items():
                row[pol] = {k: (float(np.mean([r[k] for r in rs if r[k] is not None]))
                                if any(r[k] is not None for r in rs) else None) for k in rs[0]}
            main_tab[key][N] = row
            b = row['best']
            print(f"3. {key:16s} N={N:3d}  best LA {b['lat']:6.2f} (miss S/B {100 * b['mS']:5.1f}/{100 * b['mB']:4.1f}%)  "
                  f"GIPA {row['gipa']['lat']:6.2f} ({100 * row['gipa']['mS']:5.1f}%)  "
                  f"plan {row['plan']['lat'] if 'plan' in row else float('nan'):6.2f}  local {row['local']['lat']:6.2f}  "
                  f"LA-GIPA {b['minus_gipa']['mean']:+.2f} [{b['minus_gipa']['ci95'][0]:+.2f}, {b['minus_gipa']['ci95'][1]:+.2f}]",
                  flush=True)
    prior_mask = {N: ev[('A', 'affine', 500, cA['algo'], cA['resp'], cA['rate'], N, 'prior')]['mean'] for N in DENSITIES}
    for N in DENSITIES:
        print(f"   prior-inclusive mask, A affine warm 500, N={N}: lat {prior_mask[N]['lat']:.2f}, off {prior_mask[N]['off']:.3f}")

    # 4. Density-change test
    la_cfgs = {algo: dict(algo=algo, resp=chosen['A|affine|500'][algo]['resp'],
                          rate=chosen['A|affine|500'][algo]['rate']) for algo in ALGOS}
    pols = [(f'la_{algo}', cfg) for algo, cfg in la_cfgs.items()] + \
           [(p, None) for p in ['gipa_fresh', 'gipa_stale', 'plan_fresh', 'plan_stale', 'local']]
    sch_specs = [(('la' if name.startswith('la_') else name), cfg, s) for name, cfg in pols for s in range(CFG['NSEED'])]
    with Pool(args.workers) as pool:
        traces = pool.map(job_schedule, sch_specs, chunksize=4)
    dc = {}
    for name, cfg in pols:
        pol = 'la' if name.startswith('la_') else name
        ts = [t['trace'] for t in traces if t['spec'][0] == pol and t['spec'][1] == cfg]
        mean_tr = {k: np.mean([t[k] for t in ts], 0).tolist() for k in ts[0]}
        dc[name] = dict(config=cfg, trace=mean_tr, phases=phase_summary(mean_tr['lat'], mean_tr['miss']))
        ph = dc[name]['phases']
        print(f'4. {name:11s} ' + ' | '.join(
            f"N={p['N']}: trans {p['transient_lat']:.2f}/{100 * p['transient_miss']:.1f}% "
            f"steady {p['steady_lat']:.2f}/{100 * p['steady_miss']:.1f}% rec {p['recovery_slots']}" for p in ph), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(dict(check=ok, tuning=tuning, chosen=chosen, main=main_tab, prior_mask=prior_mask,
                       per_seed=per_seed, density_change=dict(schedule=SCHEDULE, policies=dc),
                       seeds=dict(evaluation='0..29 (env 1000+s, actions 5000+s)',
                                  tuning=f'{TUNE_SEEDS[0]}..{TUNE_SEEDS[-1]}', jitter='9800+s')), fh)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
