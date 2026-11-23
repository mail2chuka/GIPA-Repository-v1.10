#!/usr/bin/env bash
# Laptop check of the rebuilt seed-level run and statistics script.
#
#   1. experiments/seed_level_main.py reruns the principal experiment with
#      counting and writes results/res_main_seed_level_rebuilt.json; the
#      expected verdict is "byte check ...: IDENTICAL" (deterministic).
#   2. experiments/stats_main.py computes the statistics from that rebuilt
#      file and compares them with results/res_main_statistics.json. With
#      the pinned NumPy 2.4.4 and SciPy 1.17.1 (requirements.txt) the
#      expected verdict is again IDENTICAL; newer versions give
#      "not byte-identical" with a largest scaled difference near 1e-15.
#
# Run from the repository root with the virtual environment active.
# About 1-3 minutes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
python experiments/seed_level_main.py --check
python experiments/stats_main.py --seed-level results/res_main_seed_level_rebuilt.json --check
