"""
aggregate_complexity.py
=======================
Combines repeated runs of experiments/exp_complexity.py (workplan E8,
decision timings for Section VII-H and Fig. R8) into one result file.

STATUS: written for the first revision of manuscript ITS-2026-08-0579.

Why repeat
----------
Timings on a shared or virtualised machine vary from run to run (about
+-30% between two runs on the reference machine), even though each run
already takes the median of 10^5 decisions. Repeating the whole timing
experiment and reporting the median across runs, with the run-to-run
range, makes the reported numbers stable and states their spread.

What it does
------------
Reads every JSON file given on the command line (each one a complete
output of exp_complexity.py from the same machine). For every numeric
value under "fleet", "single_vehicle", "offline" and "training" it
writes the median across runs in the same place, so the output has the
same structure as a single run and plotting/revision_figures.py reads it
unchanged. It adds:

* "repeats": the number of runs combined;
* "range": the same tree with [min, max] across runs for every value;
* "runs": the input file names.

"hardware", "communication", "complexity", "gipa_scalar_check" and
"mappo_actor_parameters" are taken from the first run (they do not vary).
It stops with an error if the runs were made on different processors or
software versions.

Run (from the repository root)
------------------------------
    python experiments/aggregate_complexity.py results/complexity_runs/run*.json \
        --out results/res_complexity.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

TIMED = ('fleet', 'single_vehicle', 'offline', 'training')


def _combine(trees):
    """Median and [min, max] across a list of identically shaped trees."""
    t0 = trees[0]
    if isinstance(t0, dict):
        med, rng = {}, {}
        for k in t0:
            med[k], rng[k] = _combine([t[k] for t in trees])
        return med, rng
    if isinstance(t0, bool) or not isinstance(t0, (int, float)):
        return t0, None
    vals = np.array([float(t) for t in trees])
    return float(np.median(vals)), [float(vals.min()), float(vals.max())]


def main():
    ap = argparse.ArgumentParser(description='Combine repeated E8 timing runs')
    ap.add_argument('runs', nargs='+', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    data = [json.load(open(p)) for p in args.runs]
    key = lambda d: (d['hardware']['cpu'], d['hardware']['python'], d['hardware']['numpy'],
                     d['hardware'].get('full', {}).get('kernel'))
    if len({key(d) for d in data}) != 1:
        raise SystemExit('runs come from different machines or software versions; not combining')
    out = {k: v for k, v in data[0].items() if k not in TIMED}
    rng = {}
    for k in TIMED:
        out[k], rng[k] = _combine([d[k] for d in data])
    out['repeats'] = len(data)
    out['range'] = rng
    out['runs'] = [str(p) for p in args.runs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(out, fh, indent=1)
    f = out['fleet']['160']
    r = rng['fleet']['160']
    print(f'combined {len(data)} runs -> {args.out}')
    for pol in ('plan', 'gipa', 'la_lri', 'mappo', 'ql', 'rbr', 'price'):
        print(f'  N=160 {pol:7s} median {f[pol]["median_us"]:8.1f} us/slot  range {r[pol]["median_us"][0]:8.1f}-{r[pol]["median_us"][1]:8.1f}')


if __name__ == '__main__':
    main()
