"""
compare_mappo.py
================
Compare two merged MAPPO result files (written by ``exp_mappo.py --stage
merge``) produced on different machines, to verify that independent runs of
the same experiment agree.

Results are not expected to be bit-identical across machines: the networks'
matrix products go through the CPU's BLAS library, whose rounding differs
between processors, and those tiny differences propagate through training.
Agreement is therefore judged within a tolerance (default 0.05 ms on the
mean latency of each group of 5 training runs).

Usage (from the repository root)
--------------------------------
    python experiments/compare_mappo.py results/res_mappo.json \\
        results/res_mappo_dmask_reference.json [--tol 0.05]

Only groups (model|variant|critic|regime[|deadline]) and training densities
present in both files are compared. For each, it prints the mean latency
and miss rates of each file, the difference of the group means, the largest
per-run difference (runs matched by run seed), and a verdict. Exit status is
0 if every compared group agrees within the tolerance, 1 otherwise.
"""

import argparse
import json
import sys

import numpy as np


def _runs(d, key):
    """{(N_train, run_seed): evaluation mean dict} for one group."""
    return {(r['N_train'], r['run_seed']): r['eval']['mean'] for r in d['groups'].get(key, [])}


def main():
    """Compare the common groups of two merged files and print a verdict."""
    ap = argparse.ArgumentParser(description='Compare two merged MAPPO result files')
    ap.add_argument('file_a')
    ap.add_argument('file_b')
    ap.add_argument('--tol', type=float, default=0.05, help='tolerance on group mean latency, ms')
    args = ap.parse_args()
    a, b = json.load(open(args.file_a)), json.load(open(args.file_b))
    common = sorted(set(a['groups']) & set(b['groups']))
    if not common:
        print('No common groups to compare.')
        return 1
    ok_all = True
    print(f'A = {args.file_a}\nB = {args.file_b}\ntolerance {args.tol} ms on group mean latency\n')
    for key in common:
        ra, rb = _runs(a, key), _runs(b, key)
        for N in sorted({k[0] for k in ra} & {k[0] for k in rb}):
            seeds = sorted({k[1] for k in ra if k[0] == N} & {k[1] for k in rb if k[0] == N})
            if not seeds:
                continue
            la = np.array([ra[(N, s)]['lat'] for s in seeds])
            lb = np.array([rb[(N, s)]['lat'] for s in seeds])
            mA = np.mean([100 * (ra[(N, s)]['mS'] or 0) for s in seeds]), np.mean([100 * (ra[(N, s)]['mB'] or 0) for s in seeds])
            mB = np.mean([100 * (rb[(N, s)]['mS'] or 0) for s in seeds]), np.mean([100 * (rb[(N, s)]['mB'] or 0) for s in seeds])
            diff = abs(la.mean() - lb.mean())
            ok = diff <= args.tol
            ok_all &= ok
            print(f'{key:30s} N={N:3d} ({len(seeds)} runs): A {la.mean():6.2f} ms, miss S/B {mA[0]:.2f}/{mA[1]:.2f}% | '
                  f'B {lb.mean():6.2f} ms, miss S/B {mB[0]:.2f}/{mB[1]:.2f}% | '
                  f'|mean diff| {diff:.3f}, max per-run {np.abs(la - lb).max():.3f} -> {"AGREE" if ok else "DIFFER"}')
    print('\nOverall:', 'all compared groups agree' if ok_all else 'some groups differ beyond the tolerance')
    return 0 if ok_all else 1


if __name__ == '__main__':
    sys.exit(main())
