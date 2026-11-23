"""
exp_complexity.py
=================
Workplan experiment E8: decision complexity, measured decision time,
runtime communication and training cost (Reviewer 1 comment 8, "end-to-end
decision cost"; also supports the scalability claims of the manuscript).

STATUS: written for the first revision of manuscript ITS-2026-08-0579. Not
part of the submitted v1.0.0 results.

What it measures (Regime A, affine model, seed index 0)
-------------------------------------------------------
1. Per-slot decision time of every policy for the whole fleet at
   N = 80, 160, 240 and 320, as implemented (vectorised NumPy). Each
   measurement repeats the slot until at least 10^5 vehicle decisions have
   been timed; reported: median and 95th percentile per slot, and the
   median divided by N (amortised time per decision). The environment step
   is not included; the learning automata include their update.
2. Single-vehicle decision time (10^5 calls, median and p95) for the
   policies a vehicle can run on its own: GIPA, masked softmax variants,
   greedy, planner (one categorical draw), learning automata (draw +
   update), Q-learning lookup and the MAPPO actor forward pass. Two
   implementations of GIPA are timed: the NumPy operator called with N = 1
   (dominated by NumPy call overhead) and a pure-Python scalar version
   (``gipa_scalar`` below), which is checked to choose the same action as
   ``gipa_core.gipa`` for the same uniform draw.
3. Off-line / slow-time-scale computations: Stage 0 heuristic prior,
   self-consistent prior, planner grid search (affine and Eq. 5), and the
   EWMA latency predictor update of E1, with the per-slot cost when
   refreshed every T_p = 20 slots.
4. Training cost: tabular Q-learning (one 15,000-step run at N = 160),
   MAPPO (seconds per iteration, measured over 5 iterations, times the 400
   iterations of the protocol), and the learning-automata warm-up (500
   slots of compute and of real operating time at 20 ms per slot).
5. Runtime communication, application payload only (float32 values, slot
   20 ms, refresh period 20 slots, M = 3, as in the manuscript's Section
   on the optional prior refresh, which gives 90 B/s for GIPA). Computed
   analytically per policy; headers and MAC/PHY overhead excluded.

The full computing environment (processor, cores, memory, virtualisation,
operating system, kernel, C library, Python build, NumPy/SciPy/Matplotlib
versions, BLAS and thread settings) is recorded with the results under
"hardware" -> "full" (src/sysinfo.py). Timings depend on the machine and on
background load; run on an otherwise idle machine with
OPENBLAS_NUM_THREADS=1 so that every timing is single-threaded.

Output
------
results/res_complexity.json (path configurable with --out).

Runtime
-------
About 2-5 minutes on a laptop CPU (single process; run it on an otherwise
idle machine).

Run (from the repository root)
------------------------------
    PYTHONPATH=src python experiments/exp_complexity.py
"""

import argparse
import json
import math
import platform
import time
from pathlib import Path

import numpy as np

from gipa_core import (CFG, REG, RBR, Env, QL, alllocal, gipa, greedy, masked_softmax,
                       prior_heur, price_rr, samp, sc_prior, sig)
from learning_automata import Automata
from planner import plan_allocation, planned_pmf
from sysinfo import system_info

ROOT = Path(__file__).resolve().parents[1]
DENSITIES = [80, 160, 240, 320]
DECISIONS = 100_000
T_SLOT = 0.020          # s, manuscript
T_P = 20                # slots between prior refreshes, manuscript
F32 = 4                 # bytes per float32


def hardware():
    """Processor, memory, operating system and software versions (see src/sysinfo.py).

    The short keys ``cpu``, ``logical_cores``, ``python``, ``numpy`` and
    ``system`` are kept for the plotting script; ``full`` holds the complete
    record stated in the manuscript.
    """
    info = system_info()
    return dict(cpu=info['processor']['model'], logical_cores=info['processor']['logical_cpus'],
                python=info['python']['version'], numpy=info['packages']['numpy'],
                system=platform.platform(), full=info)


def timeit(fn, reps, warm=50):
    """Return per-call times in seconds (after ``warm`` untimed calls)."""
    for _ in range(warm):
        fn()
    out = np.empty(reps)
    pc = time.perf_counter
    for i in range(reps):
        t0 = pc()
        fn()
        out[i] = pc() - t0
    return out


def stats_us(ts):
    return dict(median_us=float(np.median(ts) * 1e6), p95_us=float(np.quantile(ts, 0.95) * 1e6), n=int(len(ts)))


# ----------------------------------------------------------------------
# Pure-Python single-vehicle GIPA (what an on-board unit would run)
# ----------------------------------------------------------------------
def gipa_scalar(q, cb, dl, lam, u):
    """GIPA for one vehicle without NumPy. Returns the chosen action index.

    ``q`` and ``cb`` are lists of length M + 1, ``dl`` the deadline, ``lam``
    the class diffusion rate and ``u`` one uniform draw. Same steps as
    ``gipa_core.gipa`` (Eqs. 7 and 13-20).
    """
    n = len(q)
    m = [(-q[a] + cb[a]) <= dl for a in range(n)]
    if not any(m):
        m[0] = True
    qt = [q[a] - CFG['ETA'] * cb[a] for a in range(n)]
    Fs = sum(m)
    z = [qt[a] / CFG['TAU'] if m[a] else -math.inf for a in range(n)]
    zmax = max(z)
    p = [math.exp(v - zmax) if m[a] else 0.0 for a, v in enumerate(z)]
    s = sum(p)
    p = [v / s for v in p]
    rm = max(qt[a] for a in range(n) if m[a])
    for t in range(1, CFG['K'] + 1):
        th = CFG['TH0'] * CFG['GA'] ** t
        for a in range(n):
            if m[a]:
                arg = max(-40.0, min(40.0, (qt[a] - rm + th) / CFG['NU']))
                p[a] *= math.exp(CFG['BETA'] / (1.0 + math.exp(-arg)))
        s = sum(p)
        p = [(1 - lam) * (v / s) + (lam / Fs if m[a] else 0.0) for a, v in enumerate(p)]
    c = 0.0
    for a in range(n):
        c += p[a]
        if u <= c:
            return a
    return n - 1


def check_gipa_scalar(N=160, seeds=range(5)):
    """True if gipa_scalar picks the same action as gipa_core.gipa for every vehicle."""
    for s in seeds:
        env = Env(N, s, REG['A'])
        cb = prior_heur(N)
        q = env.q(N)
        a_ref = gipa(env, q, cb, np.random.default_rng(123 + s), N)
        u = np.random.default_rng(123 + s).random((N, 1))[:, 0]
        for i in range(N):
            lam = CFG['LAM_S'] if env.isS[i] else CFG['LAM_B']
            if gipa_scalar(list(q[i]), list(cb), env.dl[i], lam, u[i]) != a_ref[i]:
                return False
    return True


# ----------------------------------------------------------------------
# Measurements
# ----------------------------------------------------------------------
def make_mappo_actor(rng):
    from mappo import MLP, obs_dims, DEFAULTS
    d_a, _ = obs_dims('base', 'central')
    h = DEFAULTS['hidden']
    return MLP([d_a, h, h, CFG['M'] + 1], rng)


def fleet_timings():
    """Part 1: per-slot fleet decision time at each density."""
    from mappo import build_obs, feasibility_mask, masked_probs
    R = REG['A']
    res = {}
    for N in DENSITIES:
        env = Env(N, 0, R)
        q = env.q(N)
        cb = prior_heur(N)
        rg = np.random.default_rng(0)
        reps = math.ceil(DECISIONS / N)
        ql = QL(CFG['M'] + 1)
        rbr = RBR()
        plan = plan_allocation(N, R, z=2.0)
        P = planned_pmf(env, N, plan)
        la = Automata(env, N, 'lri', 'binary', 0.1, 0)
        la.activate(N)
        lat_fixed, _ = env.lat(np.zeros(N, int), N)
        actor = make_mappo_actor(np.random.default_rng(1))
        dmask = feasibility_mask(env, q, cb, N, 'deadline')
        prev = np.full(CFG['M'] + 1, 1.0 / (CFG['M'] + 1))

        def la_step():
            a = la.act(rg, N)
            la.update(a, lat_fixed, N)

        def mappo_step():
            obs = build_obs(env, q, cb, dmask, prev, N, 'base')
            logits, _ = actor.forward(obs)
            samp(masked_probs(logits, dmask), rg)

        pols = {
            'gipa': lambda: gipa(env, q, cb, rg, N),
            'msoft': lambda: masked_softmax(env, q, cb, rg, N),
            'msoft_prior': lambda: masked_softmax(env, q, cb, rg, N, use_prior=True),
            'indep': lambda: masked_softmax(env, q, cb * 0, rg, N),
            'greedy': lambda: greedy(env, q, cb, rg, N),
            'local': lambda: alllocal(env, q, cb, rg, N),
            'price': lambda: price_rr(env, q, cb, rg, N),
            'rbr': lambda: rbr.act(env, q, cb, rg, N),
            'ql': lambda: ql.act(env, q, cb, rg, N),
            'plan': lambda: samp(P, rg),
            'la_lri': la_step,
            'mappo': mappo_step,
        }
        res[N] = {}
        for name, fn in pols.items():
            ts = timeit(fn, reps)
            st = stats_us(ts)
            st['per_decision_us'] = st['median_us'] / N
            res[N][name] = st
        print(f'1. N={N}: ' + ', '.join(f"{k} {v['median_us']:.0f}" for k, v in res[N].items()) + ' us/slot', flush=True)
    return res


def single_vehicle_timings():
    """Part 2: one vehicle's decision, 10^5 calls (half safety, half background)."""
    from mappo import build_obs, feasibility_mask, masked_probs
    R = REG['A']
    N = 160
    cb = prior_heur(N)
    out = {}
    envs = []
    for s in range(100):                       # find one safety and one background vehicle 0
        e = Env(N, s, R)
        if not any(bool(x.isS[0]) == bool(e.isS[0]) for x in envs):
            envs.append(e)
        if len(envs) == 2:
            break
    rg = np.random.default_rng(0)
    actor = make_mappo_actor(np.random.default_rng(1))
    ql = QL(CFG['M'] + 1)
    prev = np.full(CFG['M'] + 1, 1.0 / (CFG['M'] + 1))
    plan = plan_allocation(N, R, z=2.0)
    half = DECISIONS // 2
    calls = {}
    for e in envs:
        q1 = e.q(1)
        la = Automata(e, 1, 'lri', 'binary', 0.1, 0)
        la.activate(1)
        lat1 = np.array([e.loc[0]])
        P1 = planned_pmf(e, 1, plan)
        dm = feasibility_mask(e, q1, cb, 1, 'deadline')
        ql_list = list(q1[0])
        cbl = list(cb)
        lam = CFG['LAM_S'] if e.isS[0] else CFG['LAM_B']
        dl = float(e.dl[0])

        def la1():
            a = la.act(rg, 1)
            la.update(a, lat1, 1)

        def mappo1():
            obs = build_obs(e, q1, cb, dm, prev, 1, 'base')
            logits, _ = actor.forward(obs)
            samp(masked_probs(logits, dm), rg)

        fns = {
            'gipa_numpy': lambda: gipa(e, q1, cb, rg, 1),
            'gipa_scalar': lambda: gipa_scalar(ql_list, cbl, dl, lam, rg.random()),
            'msoft_prior': lambda: masked_softmax(e, q1, cb, rg, 1, use_prior=True),
            'greedy': lambda: greedy(e, q1, cb, rg, 1),
            'plan': lambda: samp(P1, rg),
            'la_lri': la1,
            'ql': lambda: ql.act(e, q1, cb, rg, 1),
            'mappo': mappo1,
        }
        for k, fn in fns.items():
            calls.setdefault(k, []).append(timeit(fn, half))
    for k, parts in calls.items():
        out[k] = stats_us(np.concatenate(parts))
    print('2. single vehicle: ' + ', '.join(f"{k} {v['median_us']:.1f}" for k, v in out.items()) + ' us', flush=True)
    return out


def offline_timings():
    """Part 3: Stage 0 prior, self-consistent prior, planner, EWMA predictor."""
    from calibration import make_calibration, plan_allocation_ps
    R = REG['A']
    cal = make_calibration()
    out = {}
    for N in DENSITIES:
        t = dict(
            prior_heur=stats_us(timeit(lambda: prior_heur(N), 10000)),
            sc_prior=stats_us(timeit(lambda: sc_prior(N, R), 20, warm=2)),
            plan_affine=stats_us(timeit(lambda: plan_allocation(N, R, z=2.0), 20, warm=2)),
            plan_ps=stats_us(timeit(lambda: plan_allocation_ps(N, R, cal, z=2.0), 20, warm=2)),
        )
        l_hat = np.full((N, CFG['M'] + 1), 10.0)
        l_t = np.full((N, CFG['M'] + 1), 11.0)
        rows = np.arange(N)
        a = np.random.default_rng(0).integers(0, CFG['M'] + 1, N)

        def ewma():
            l_hat[rows, a] = 0.8 * l_hat[rows, a] + 0.2 * l_t[rows, a]
        t['ewma_update_slot'] = stats_us(timeit(ewma, math.ceil(DECISIONS / N)))
        for k in ['prior_heur', 'sc_prior', 'plan_affine', 'plan_ps']:
            t[k]['per_slot_at_refresh_Tp_us'] = t[k]['median_us'] / T_P
        out[N] = t
        print(f"3. N={N}: plan {t['plan_affine']['median_us'] / 1e3:.1f} ms (affine), "
              f"{t['plan_ps']['median_us'] / 1e3:.1f} ms (Eq. 5), sc_prior {t['sc_prior']['median_us'] / 1e3:.1f} ms, "
              f"prior_heur {t['prior_heur']['median_us']:.1f} us, EWMA {t['ewma_update_slot']['median_us']:.1f} us/slot", flush=True)
    return out


def training_cost():
    """Part 4: Q-learning, MAPPO and LA warm-up."""
    import mappo
    from learning_automata import run_la
    R = REG['A']
    out = {}
    ql = QL(CFG['M'] + 1)
    t0 = time.perf_counter()
    ql.train(160, R)
    out['ql_run_s'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    mappo.train(160, R, run_seed=0, iters=5)
    s_it = (time.perf_counter() - t0) / 5
    out['mappo_s_per_iter'] = s_it
    out['mappo_run_s_400_iters'] = 400 * s_it
    t0 = time.perf_counter()
    run_la('lri', 'binary', 0.1, 160, 0, R, warm=500)
    out['la_episode_545_slots_s'] = time.perf_counter() - t0
    out['la_warmup_operating_time_s'] = 500 * T_SLOT
    print(f"4. QL run {out['ql_run_s']:.1f} s; MAPPO {s_it:.3f} s/iter ({out['mappo_run_s_400_iters'] / 60:.1f} min per run); "
          f"LA 545-slot episode {out['la_episode_545_slots_s']:.2f} s compute, warm-up = {out['la_warmup_operating_time_s']:.0f} s of operation",
          flush=True)
    return out


def communication(N=160):
    """Part 5: runtime application payload, bytes per second (analytic)."""
    M = CFG['M']
    slot_rate = 1 / T_SLOT
    refresh_rate = 1 / (T_P * T_SLOT)
    pricing_iters = 30
    rows = {
        'gipa': dict(Bps=M * M * F32 * refresh_rate, formula='M RSUs x M float32 every T_p slots (optional refresh; 0 with a fixed prior)'),
        'plan': dict(Bps=M * 2 * F32 * refresh_rate, formula='M RSUs x (x_S, x_B) float32 every T_p slots (0 if fixed)'),
        'msoft / msoft_prior / entreg': dict(Bps=M * M * F32 * refresh_rate, formula='as GIPA (same prior)'),
        'indep / greedy / local': dict(Bps=0.0, formula='none'),
        'la': dict(Bps=0.0, formula="none: each vehicle uses only its own task's realised latency"),
        'rbr': dict(Bps=M * F32 * slot_rate, formula='M float32 load shares every slot'),
        'ql': dict(Bps=F32 * slot_rate, formula='largest RSU share (1 float32) every slot'),
        'mappo (base)': dict(Bps=M * M * F32 * refresh_rate, formula='prior refresh as GIPA; training needs the global state'),
        'mappo (load variant)': dict(Bps=M * F32 * slot_rate, formula='per-slot load shares, as RBR'),
        'price': dict(Bps=pricing_iters * (M * F32 + N * (M + 1) * F32) * slot_rate,
                      formula=f'{pricing_iters} iterations per slot x (M prices down + N PMFs of M+1 float32 up); N = {N}'),
    }
    for k, v in rows.items():
        print(f"5. {k:30s} {v['Bps']:12.1f} B/s  ({v['formula']})")
    return dict(assumptions=dict(T_slot_s=T_SLOT, T_p_slots=T_P, M=M, bytes_per_value=F32, N_for_price=N), rows=rows)


COMPLEXITY = {
    'gipa': dict(time='O(K |A|)', memory='O(|A|)', runtime_comm='prior refresh, O(M) per T_p slots', training='none (prior offline)'),
    'plan': dict(time='O(|A|) per decision; grid search O(G_S G_B M) per refresh', memory='O(|A|)', runtime_comm='plan refresh, O(M)', training='none'),
    'msoft / entreg': dict(time='O(|A|)', memory='O(|A|)', runtime_comm='prior refresh', training='none'),
    'greedy / indep': dict(time='O(|A|)', memory='O(|A|)', runtime_comm='none', training='none'),
    'price': dict(time='O(I N |A|) per slot, centralised', memory='O(N |A|)', runtime_comm='I rounds of prices and PMFs per slot', training='none'),
    'rbr': dict(time='O(|A|)', memory='O(|A|)', runtime_comm='load broadcast every slot', training='none'),
    'ql': dict(time='O(|A|) lookup', memory='O(|S| |A|)', runtime_comm='load statistic every slot', training='O(T N |A|), T = 15,000 slots'),
    'la': dict(time='O(|A|) draw + update', memory='O(|A|) (pursuit: 3 |A|)', runtime_comm='none (own feedback)', training='online; hundreds of slots to converge'),
    'mappo': dict(time='O(P) forward pass, P = actor parameters', memory='O(P)', runtime_comm='prior refresh (base) or load broadcast (load)', training='O(E P) with E = samples x epochs; 400 iterations'),
}


def main():
    ap = argparse.ArgumentParser(description='E8 complexity and decision cost')
    ap.add_argument('--out', type=Path, default=ROOT / 'results' / 'res_complexity.json')
    args = ap.parse_args()
    hw = hardware()
    print('hardware:', hw, flush=True)
    ok = check_gipa_scalar()
    print('0. gipa_scalar matches gipa_core.gipa:', ok, flush=True)
    from mappo import obs_dims, DEFAULTS
    d_a, _ = obs_dims('base', 'central')
    h = DEFAULTS['hidden']
    P = d_a * h + h + h * h + h + h * (CFG['M'] + 1) + CFG['M'] + 1
    res = dict(hardware=hw, gipa_scalar_check=ok, mappo_actor_parameters=P,
               fleet=fleet_timings(), single_vehicle=single_vehicle_timings(),
               offline=offline_timings(), training=training_cost(), communication=communication(),
               complexity=COMPLEXITY)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(res, fh, indent=1)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
