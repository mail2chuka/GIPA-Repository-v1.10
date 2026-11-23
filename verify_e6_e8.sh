#!/usr/bin/env bash
# Optional check on a second machine of experiments E6 (learning automata)
# and E8 (complexity).
#
#   1. reruns experiments/exp_la.py into results/res_la_laptop.json and checks
#      that every number agrees with results/res_la.json (reference run) to
#      1e-9; E6 is deterministic, so they should be identical
#   2. reruns the E8 timings five times on this machine (run_complexity_repeats.sh)
#      into results/res_complexity_laptop.json; timings are machine dependent,
#      so they are printed, not compared. The manuscript reports the reference
#      machine's timings (results/res_complexity.json); Fig. R8 is not redrawn.
#
# Run from the repository root with the virtual environment active. Close
# other programs first so the timings are clean. About 10-20 minutes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
python experiments/exp_la.py --out results/res_la_laptop.json | tee results/exp_la_laptop_log.txt
python - <<'PY'
import json, math
a = json.load(open('results/res_la.json'))
b = json.load(open('results/res_la_laptop.json'))
worst, n = 0.0, 0
def walk(x, y, path):
    global worst, n
    if isinstance(x, dict):
        assert set(x) == set(y), path
        for k in x: walk(x[k], y[k], path + '/' + k)
    elif isinstance(x, list):
        assert len(x) == len(y), path
        for i, (u, v) in enumerate(zip(x, y)): walk(u, v, path)
    elif isinstance(x, float) and isinstance(y, (int, float)):
        n += 1
        if not (math.isnan(x) and math.isnan(y)):
            worst = max(worst, abs(x - y))
    else:
        assert x == y, (path, x, y)
walk(a, b, '')
print(f'E6 check: {n} numbers compared, largest difference {worst:.3g}:',
      'IDENTICAL' if worst <= 1e-9 else 'DIFFERENT')
PY
NOPLOT=1 OUT=results/res_complexity_laptop.json bash run_complexity_repeats.sh
