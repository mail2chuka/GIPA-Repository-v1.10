"""
compare_results.py
==================
Check that two JSON result files agree number by number.

Used for cross-machine verification of deterministic experiments (E6, E9):
every float must agree within ``--tol`` (default 1e-9) relative to
max(1, |a|, |b|), so large values (e.g. backlog drift) are compared
relatively and small ones absolutely; every other
value (keys, strings, integers, nulls, list lengths) must be identical.

Run (from the repository root)
------------------------------
    python experiments/compare_results.py results/res_asym.json results/res_asym_laptop.json
"""

import argparse
import json
import math
import sys


def compare(a, b, tol):
    """Return (numbers compared, largest float difference, first structural mismatch or None)."""
    state = dict(n=0, worst=0.0, bad=None)

    def walk(x, y, path):
        if state['bad']:
            return
        if isinstance(x, dict) and isinstance(y, dict):
            if set(x) != set(y):
                state['bad'] = f'{path}: different keys'
                return
            for k in x:
                walk(x[k], y[k], f'{path}/{k}')
        elif isinstance(x, list) and isinstance(y, list):
            if len(x) != len(y):
                state['bad'] = f'{path}: different lengths'
                return
            for i, (u, v) in enumerate(zip(x, y)):
                walk(u, v, f'{path}[{i}]')
        elif isinstance(x, float) or isinstance(y, float):
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                state['bad'] = f'{path}: {x!r} vs {y!r}'
                return
            state['n'] += 1
            if not (math.isnan(x) and math.isnan(y)):
                state['worst'] = max(state['worst'], abs(x - y) / max(1.0, abs(x), abs(y)))
        elif x != y:
            state['bad'] = f'{path}: {x!r} vs {y!r}'

    walk(a, b, '')
    return state['n'], state['worst'], state['bad']


def main():
    ap = argparse.ArgumentParser(description='Compare two JSON result files')
    ap.add_argument('file_a')
    ap.add_argument('file_b')
    ap.add_argument('--tol', type=float, default=1e-9)
    args = ap.parse_args()
    n, worst, bad = compare(json.load(open(args.file_a)), json.load(open(args.file_b)), args.tol)
    if bad:
        print('DIFFERENT (structure):', bad)
        sys.exit(1)
    ok = worst <= args.tol
    print(f'{n} numbers compared, largest scaled difference {worst:.3g}:', 'IDENTICAL' if ok else 'DIFFERENT')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
