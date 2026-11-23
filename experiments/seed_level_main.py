"""
seed_level_main.py
==================
Regenerates ``results/res_main_seed_level.json``: the principal experiment
(both regimes, N = 80-320, the 15 policies and ablations of
``experiments/exp.py``) with one record per seed, including the task, miss
and offload counts behind the Clopper-Pearson bounds.

STATUS: rebuilt for the first revision of manuscript ITS-2026-08-0579. The
original "instrumented run" was not in the repository. This script repeats
``experiments/exp.py`` exactly (same policy order, same Q-learner trained
once per regime at N = 160 and carried across densities and seeds, same
random streams) and only adds counting; it does not change any decision.

Per-seed record
---------------
seed_index, environment_seed (1000 + s), action_seed (5000 + s), the ten
``gipa_core.run`` metrics, and over the 45 evaluated slots:
nS, nB (safety and background task-slots), missS_count, missB_count,
offS_count, offB_count, evaluated_slots, fair_defined_slots (slots in which
some RSU received a task, i.e. Jain fairness is defined).

Check
-----
``--check`` compares the output with the archived file: byte identity
(MD5 8727bda798913aae39beeda4efc1f707) and, if that fails, value by value.
The simulator is deterministic, so byte identity is expected on any machine
with the pinned NumPy.

Run (from the repository root)
------------------------------
    PYTHONPATH=src python experiments/seed_level_main.py --check     # several minutes
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from gipa_core import (CFG, REG, QL, RBR, Env, RNG_ACT_BASE, RNG_ENV_BASE, alllocal, gipa, greedy,
                       masked_softmax, prior_heur, price_rr, safe_mean)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp import DENSITIES, POLICIES  # noqa: E402  (same policy list and order as the principal run)

ARCHIVED_MD5 = '8727bda798913aae39beeda4efc1f707'


def run_counts(pol, N, seed, R, cb=None, **kw):
    """``gipa_core.run`` with counting added; decisions and random draws are unchanged."""
    env = Env(N, seed, R)
    rg = np.random.default_rng(RNG_ACT_BASE + seed)
    cb = prior_heur(N) if cb is None else cb
    rbr = RBR()
    out = dict(mS=[], mB=[], lat=[], fair=[], off=[], offS=[], offB=[], en=[], p95=[], maxl=[])
    cnt = dict(nS=0, nB=0, missS_count=0, missB_count=0, offS_count=0, offB_count=0,
               evaluated_slots=0, fair_defined_slots=0)
    for t in range(CFG['SLOTS']):
        q = env.q(N)
        if pol == 'gipa':
            a = gipa(env, q, cb, rg, N, **kw)
        elif pol == 'greedy':
            a = greedy(env, q, cb, rg, N)
        elif pol == 'indep':
            a = masked_softmax(env, q, cb * 0, rg, N)
        elif pol == 'msoft':
            a = masked_softmax(env, q, cb, rg, N)
        elif pol == 'msoft_prior':
            a = masked_softmax(env, q, cb, rg, N, use_prior=True)
        elif pol == 'entreg':
            a = masked_softmax(env, q, cb, rg, N, use_prior=True, tau=kw.get('tau', 1.2))
        elif pol == 'price':
            a = price_rr(env, q, cb, rg, N)
        elif pol == 'rbr':
            a = rbr.act(env, q, cb, rg, N)
        elif pol == 'ql':
            a = kw['ql'].act(env, q, cb, rg, N)
        elif pol == 'local':
            a = alllocal(env, q, cb, rg, N)
        else:
            raise ValueError(pol)
        cost, c = env.lat(a, N)
        e = env.en(a, N)
        if t >= CFG['WARM']:
            S = env.isS[:N]
            miss = cost > env.dl[:N]
            out['mS'].append(miss[S].mean() if S.any() else 0.)
            out['mB'].append(miss[~S].mean() if (~S).any() else 0.)
            out['lat'].append(cost.mean())
            out['p95'].append(np.percentile(cost, 95))
            rl = c[1:]
            out['fair'].append(np.nan if rl.sum() == 0 else rl.sum() ** 2 / (len(rl) * (rl ** 2).sum()))
            out['off'].append((a > 0).mean())
            out['offS'].append((a[S] > 0).mean() if S.any() else 0.)
            out['offB'].append((a[~S] > 0).mean() if (~S).any() else 0.)
            out['en'].append(e.mean())
            out['maxl'].append(rl.max())
            cnt['nS'] += int(S.sum())
            cnt['nB'] += int((~S).sum())
            cnt['missS_count'] += int(miss[S].sum())
            cnt['missB_count'] += int(miss[~S].sum())
            cnt['offS_count'] += int((a[S] > 0).sum())
            cnt['offB_count'] += int((a[~S] > 0).sum())
            cnt['evaluated_slots'] += 1
            cnt['fair_defined_slots'] += int(rl.sum() > 0)
    m = {k: safe_mean(v) for k, v in out.items()}
    rec = dict(seed_index=seed, environment_seed=RNG_ENV_BASE + seed, action_seed=RNG_ACT_BASE + seed)
    for k in ['mS', 'mB', 'lat', 'fair', 'off', 'offS', 'offB', 'en', 'p95', 'maxl']:
        rec[k] = m[k]
    rec.update(cnt)
    return rec


def build():
    res = {}
    for reg in ['A', 'B']:
        R = REG[reg]
        res[reg] = {}
        ql = QL(CFG['M'] + 1)
        ql.train(160, R)
        for N in DENSITIES:
            res[reg][str(N)] = {}
            for name, kw in POLICIES:
                base = 'gipa' if name.startswith('gipa') else name
                kwargs = {'ql': ql} if name == 'ql' else dict(kw)
                res[reg][str(N)][name] = [run_counts(base, N, s, R, None, **kwargs) for s in range(CFG['NSEED'])]
            print(f'{reg} N={N} done', flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description='Rebuild res_main_seed_level.json')
    ap.add_argument('--out', type=Path, default=ROOT / 'results' / 'res_main_seed_level_rebuilt.json')
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    res = build()
    with open(args.out, 'w') as fh:
        json.dump(res, fh, indent=1)
    md5 = hashlib.md5(open(args.out, 'rb').read()).hexdigest()
    print(f'wrote {args.out} (MD5 {md5})')
    if args.check:
        from compare_results import compare
        print('byte check against the archive (MD5 ' + ARCHIVED_MD5 + '):',
              'IDENTICAL' if md5 == ARCHIVED_MD5 else 'not byte-identical')
        n, worst, bad = compare(json.load(open(ROOT / 'results' / 'res_main_seed_level.json')), res, 0)
        print(f'value check: {n} numbers, largest scaled difference {worst:.2g}, structure',
              'identical' if bad is None else f'DIFFERENT ({bad})')


if __name__ == '__main__':
    main()
