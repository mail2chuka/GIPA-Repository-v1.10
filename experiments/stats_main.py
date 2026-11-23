"""
stats_main.py
=============
Regenerates ``results/res_main_statistics.json`` from
``results/res_main_seed_level.json``: the seed-level bootstrap summaries,
Clopper-Pearson zero-miss bounds, nine paired comparisons at Regime A,
N = 160, and the 1 ms latency TOST behind Sections VI-G and VII and Figs. 3-8.

STATUS: rebuilt for the first revision of manuscript ITS-2026-08-0579. The
original script was not in the repository. This one reproduces every value
of the archived file: all 1,200 summary intervals, 90 paired intervals, the
TOST and the Clopper-Pearson bounds, including the shared bootstrap stream.
Under the pinned environment (NumPy 2.4.4, SciPy 1.17.1, requirements.txt)
the output is expected to be byte-identical (MD5
2e30d3a6f2d71875da620563f4043077). With newer NumPy/SciPy, 32 values differ
in the last printed digit (at most 4e-15 relative), from changes in
floating-point summation and in scipy.stats.beta; --check reports both.

Order of computation (it fixes the shared bootstrap stream)
-----------------------------------------------------------
1. Summaries: regime A then B; N = 80, 160, 240, 320; policies in the order
   of the seed-level file; metrics mS, mB, lat, p95, fair, off, offS, offB,
   en, maxl.
2. Zero-miss bounds (no random numbers): summed miss and task counts per
   class, one-sided 95% Clopper-Pearson upper bound.
3. Paired comparisons at Regime A, N = 160, GIPA minus each comparator, in
   the order of COMPARISONS, metrics lat, p95, mS, mB, off, offS, offB,
   fair, en, maxl; the first comparison (all-local) also gets the 1 ms TOST.

Run (from the repository root)
------------------------------
    PYTHONPATH=src python experiments/stats_main.py                 # writes results/res_main_statistics_rebuilt.json
    PYTHONPATH=src python experiments/stats_main.py --check         # also compares with the archived file
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from stats import BOOT_REPS, BOOT_SEED, Bootstrap, cp_upper, tost_paired

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))   # for compare_results
SUMMARY_METRICS = ['mS', 'mB', 'lat', 'p95', 'fair', 'off', 'offS', 'offB', 'en', 'maxl']
COMPARISON_METRICS = ['lat', 'p95', 'mS', 'mB', 'off', 'offS', 'offB', 'fair', 'en', 'maxl']
COMPARISONS = [('gipa_vs_all_local', 'local'), ('gipa_vs_reactive_best_response', 'rbr'),
               ('gipa_vs_masked_softmax', 'msoft'), ('gipa_vs_prior_adjusted_masked', 'msoft_prior'),
               ('gipa_vs_entropy_regularized', 'entreg'), ('gipa_vs_fixed_weights', 'gipa_fixed'),
               ('gipa_vs_binary_weights', 'gipa_binary'), ('gipa_vs_no_diffusion', 'gipa_nodiff'),
               ('gipa_vs_no_prior', 'gipa_noprior')]
ARCHIVED_MD5 = '2e30d3a6f2d71875da620563f4043077'


def build(seed_level):
    bs = Bootstrap(BOOT_SEED, BOOT_REPS)
    summaries = {}
    for reg, byN in seed_level.items():
        summaries[reg] = {}
        for N, byPol in byN.items():
            summaries[reg][N] = {pol: {m: bs.summary([r[m] for r in recs]) for m in SUMMARY_METRICS}
                                 for pol, recs in byPol.items()}
    bounds = {}
    for reg, byN in seed_level.items():
        bounds[reg] = {}
        for N, byPol in byN.items():
            bounds[reg][N] = {}
            for pol, recs in byPol.items():
                out = {}
                for cls, mk, nk in [('safety', 'missS_count', 'nS'), ('background', 'missB_count', 'nB')]:
                    x = int(sum(r[mk] for r in recs))
                    n = int(sum(r[nk] for r in recs))
                    out[cls] = dict(misses=x, trials=n, upper95=cp_upper(x, n))
                bounds[reg][N][pol] = out
    comps = {}
    g = seed_level['A']['160']['gipa']
    for name, other in COMPARISONS:
        o = seed_level['A']['160'][other]
        comps[name] = {m: bs.paired([r[m] for r in g], [r[m] for r in o]) for m in COMPARISON_METRICS}
        if other == 'local':
            comps[name]['latency_tost_1ms'] = tost_paired([r['lat'] for r in g], [r['lat'] for r in o], 1.0)
    return dict(bootstrap_seed=BOOT_SEED, bootstrap_reps=BOOT_REPS, confidence_level=0.95,
                summaries=summaries, zero_miss_bounds=bounds, comparisons=comps)


def main():
    ap = argparse.ArgumentParser(description='Rebuild res_main_statistics.json')
    ap.add_argument('--seed-level', type=Path, default=ROOT / 'results' / 'res_main_seed_level.json')
    ap.add_argument('--out', type=Path, default=ROOT / 'results' / 'res_main_statistics_rebuilt.json')
    ap.add_argument('--check', action='store_true', help='compare with results/res_main_statistics.json')
    args = ap.parse_args()
    res = build(json.load(open(args.seed_level)))
    with open(args.out, 'w') as fh:
        json.dump(res, fh, indent=1)
    md5 = hashlib.md5(open(args.out, 'rb').read()).hexdigest()
    print(f'wrote {args.out} (MD5 {md5})')
    if args.check:
        import numpy, scipy
        from compare_results import compare
        n, worst, bad = compare(json.load(open(ROOT / 'results' / 'res_main_statistics.json')), res, 0)
        print(f'NumPy {numpy.__version__}, SciPy {scipy.__version__}')
        print('byte check against the archive (MD5 ' + ARCHIVED_MD5 + '):',
              'IDENTICAL' if md5 == ARCHIVED_MD5 else 'not byte-identical')
        print(f'value check: {n} numbers, largest scaled difference {worst:.2g}, structure',
              'identical' if bad is None else f'DIFFERENT ({bad})')


if __name__ == '__main__':
    main()
