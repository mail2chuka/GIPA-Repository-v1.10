"""
exp_async_local.py
==================
Workplan experiment E5b: asynchronous arrivals with a persistent on-board
processor on every vehicle, in addition to the persistent RSU queues of E5.

STATUS: written for the first revision of manuscript ITS-2026-08-0579 (second
round, November 2026). Not part of the submitted v1.0.0 results. E5
(experiments/exp_async.py, results/res_async.json) is unchanged.

Why
---
In E5 a task kept local takes L_k from arrival, as if the vehicle's processor
were always idle. With one task per 20 ms on average per vehicle and
L_B = 22 ms (Regime A), a background vehicle that runs everything locally
offers its processor 1.1 times its capacity: its local queue grows without
bound. E5b gives each vehicle one processor that carries unfinished local
work forward (FIFO; processor sharing as a sensitivity check) and repeats
the E5 comparison.

Policies (label: simulator policy, prior)
-----------------------------------------
  local, greedy, indep, gipa_slot, la, rbr_b20, rbr_b100, gipa_b20  as in E5
  jsq_live        near-live shortest expected delay; the local estimate
                  includes the vehicle's own queue
  plan            GIPA with planned prior, the E5 plan (ignores local queues)
  plan_q          GIPA with planned prior, plan with M/D/1 local queues
                  (async_sim.plan_async(local='fifo', eps_loc=1): stability
                  limit 0.95, no local miss constraint)
  gipa_slot+own   GIPA with the heuristic prior; local score and deadline
                  mask use the vehicle's own predicted local latency
  plan_e5+own     the E5 plan with the own-queue mask (ablation)
  plan+own        GIPA with planned prior and the own-queue mask; the plan
                  accounts for tasks that the mask moves to the RSUs
                  (plan_async(local='fifo_own'), closed-form admission model
                  async_sim.local_admit_stats)
The own-queue mask uses only the vehicle's own state: no communication.

What it runs
------------
0. Checks: FIFO M/D/1 waiting time against Erlang's formula; the closed-form
   admission model against simulation (background and safety).
1. Regimes A and B, local='fifo', fleets 55-240, 30 seeds, 20 s per run,
   paired with E5 (same classes, arrivals and decision uniforms).
2. Sensitivity: Regime A, local='ps', N = 80, 160, 240, eight policies.
3. Summaries over seeds and paired differences (seed-level bootstrap,
   10,000 resamples, seed 20260804).

Output
------
results/res_async_local.json: "checks", "plans", "main" (model -> regime ->
N -> label -> summary), "paired", "per_seed".

Runtime
-------
About 60-90 minutes on a 2-core laptop.

Run (from the repository root)
------------------------------
    PYTHONPATH=src python experiments/exp_async_local.py
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from async_sim import check_admit, check_md1, plan_async, run_async
from calibration import make_calibration
from gipa_core import CFG, REG

ROOT = Path(__file__).resolve().parents[1]
FLEETS = [55, 80, 87, 104, 160, 240]
SEEDS = list(range(CFG['NSEED']))
BOOT_REPS, BOOT_SEED = 10000, 20260804

# label -> (simulator policy, plan key or None)
POLICIES = {
    'local': ('local', None), 'greedy': ('greedy', None), 'indep': ('indep', None),
    'gipa_slot': ('gipa_slot', None), 'gipa_slot+own': ('gipa_slot+own', None),
    'plan': ('plan', 'e5'), 'plan_q': ('plan', 'q'), 'plan_e5+own': ('plan+own', 'e5'), 'plan+own': ('plan+own', 'own'),
    'la': ('la', None), 'rbr_b20': ('rbr_b20', None), 'rbr_b100': ('rbr_b100', None), 'gipa_b20': ('gipa_b20', None),
    'jsq_live': ('jsq_live', None),
}
PS_POLICIES = ['local', 'gipa_slot', 'gipa_slot+own', 'plan', 'plan_q', 'plan+own', 'la', 'jsq_live']
PS_FLEETS = [80, 160, 240]
PAIRS = [('plan+own', 'plan'), ('plan+own', 'plan_q'), ('plan+own', 'plan_e5+own'), ('plan+own', 'gipa_slot+own'),
         ('plan+own', 'la'), ('plan+own', 'jsq_live'), ('gipa_slot+own', 'gipa_slot')]


def make_plans(cal):
    out = {}
    for reg in ['A', 'B']:
        out[reg] = {}
        for N in FLEETS:
            out[reg][N] = dict(e5=plan_async(N, REG[reg], cal),
                               q=plan_async(N, REG[reg], cal, local='fifo', eps_loc=1.0),
                               own=plan_async(N, REG[reg], cal, local='fifo_own'))
    return out


def job(spec):
    model, reg, N, label, seed, plan, la_cfg = spec
    pol = POLICIES[label][0]
    m = run_async(pol, N, seed, REG[reg], make_calibration(), plan=plan, la_cfg=la_cfg, local=model)
    return (model, reg, N, label, seed), m


def paired_ci(x, y):
    d = np.asarray(x, float) - np.asarray(y, float)
    rg = np.random.default_rng(BOOT_SEED)
    bs = d[rg.integers(0, len(d), (BOOT_REPS, len(d)))].mean(1)
    return dict(mean=float(d.mean()), ci95=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))])


def summarise(rs):
    out = {}
    for k in rs[0]:
        vals = [r[k] for r in rs]
        if isinstance(vals[0], bool):
            out[k + '_share'] = float(np.mean(vals))
        elif isinstance(vals[0], list):
            out[k] = np.mean(np.array(vals, float), 0).tolist()
        elif all(v is None for v in vals):
            out[k] = None
        else:
            out[k] = float(np.mean([v for v in vals if v is not None]))
    out['lat_median'] = float(np.median([r['lat'] for r in rs]))
    return out


def main():
    ap = argparse.ArgumentParser(description='E5b asynchronous arrivals with local queues')
    ap.add_argument('--out', type=Path, default=ROOT / 'results' / 'res_async_local.json')
    ap.add_argument('--workers', type=int, default=2)
    args = ap.parse_args()

    checks = dict(md1=check_md1(lam=0.03, n=400000), admit_B=check_admit(f=0.2, L=22.0, D=40.0),
                  admit_S=check_admit(f=0.0, L=10.0, D=12.0))
    print('0. checks', json.dumps(checks), flush=True)

    cal = make_calibration()
    plans = make_plans(cal)
    for reg in ['A', 'B']:
        for N in FLEETS:
            s = []
            for k, p in plans[reg][N].items():
                s.append(f"{k}: " + ('infeasible' if p is None else f"x_S {p['x_S']:.3f} x_B {p['x_B']:.3f} J {p['J']:.2f}"))
            print(f'   plan {reg} N={N}: ' + '; '.join(s), flush=True)
    la_chosen = json.load(open(ROOT / 'results' / 'res_la.json'))['chosen']

    specs = []
    for model, regs, fleets, labels in [('fifo', ['A', 'B'], FLEETS, list(POLICIES)),
                                        ('ps', ['A'], PS_FLEETS, PS_POLICIES)]:
        for reg in regs:
            c = la_chosen[f'{reg}|ps|500']['lri']
            la_cfg = dict(algo='lri', resp='binary', rate=c['rate'])
            for N in fleets:
                for label in labels:
                    key = POLICIES[label][1]
                    plan = plans[reg][N][key] if key else None
                    if key and plan is None:
                        continue
                    for s in SEEDS:
                        specs.append((model, reg, N, label, s, plan, la_cfg))
    print(f'{len(specs)} runs', flush=True)
    with Pool(args.workers) as pool:
        res = dict(pool.map(job, specs, chunksize=4))

    main_tab, paired, per_seed = {}, {}, {}
    for model, regs, fleets, labels in [('fifo', ['A', 'B'], FLEETS, list(POLICIES)),
                                        ('ps', ['A'], PS_FLEETS, PS_POLICIES)]:
        main_tab[model], paired[model] = {}, {}
        for reg in regs:
            main_tab[model][reg], paired[model][reg] = {}, {}
            for N in fleets:
                t = main_tab[model][reg][N] = {}
                for label in labels:
                    rs = [res[(model, reg, N, label, s)] for s in SEEDS if (model, reg, N, label, s) in res]
                    if not rs:
                        continue
                    t[label] = summarise(rs)
                    per_seed[f'{model}|{reg}|{N}|{label}'] = [dict(lat=r['lat'], miss=r['miss'], unstable=r['unstable'])
                                                              for r in rs]
                pr = {}
                for a, b in PAIRS:
                    if a in t and b in t:
                        la_ = [res[(model, reg, N, a, s)]['lat'] for s in SEEDS]
                        lb_ = [res[(model, reg, N, b, s)]['lat'] for s in SEEDS]
                        ma = [res[(model, reg, N, a, s)]['miss'] for s in SEEDS]
                        mb = [res[(model, reg, N, b, s)]['miss'] for s in SEEDS]
                        pr[f'{a} - {b}'] = dict(lat=paired_ci(la_, lb_), miss=paired_ci(ma, mb))
                paired[model][reg][N] = pr

                def f(p):
                    v = t.get(p)
                    if v is None:
                        return '   n/a'
                    s = f"{min(v['lat'], 9999):7.2f}/{100 * v['miss']:5.2f}%"
                    if v['unstable_share'] > 0:
                        s += f"(U{v['unstable_share']:.0%})"
                    return s
                print(f"{model} {reg} N={N:3d} lat/miss: " + ' | '.join(f'{p} {f(p)}' for p in labels), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(dict(checks=checks, fleets=FLEETS, plans=plans, main=main_tab, paired=paired,
                       per_seed=per_seed, horizon_ms=20000.0, seeds=SEEDS), fh)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
