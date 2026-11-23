#!/usr/bin/env bash
# Laptop verification of experiment E12 (heterogeneous tasks, GIPA on the plan).
#
# Reruns experiments/exp_hetero.py into results/res_hetero_laptop.json and
# compares every number with results/res_hetero.json (reference run). The
# experiment is deterministic; the expected verdict is IDENTICAL. If only the
# planner-derived rows differ in the last digits (SciPy version), rerun the
# comparison with --tol 1e-6.
#
# Run from the repository root with the virtual environment active.
# About 10-20 minutes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
python -W ignore experiments/exp_hetero.py --out results/res_hetero_laptop.json | tee results/exp_hetero_laptop_log.txt
python experiments/compare_results.py results/res_hetero.json results/res_hetero_laptop.json
