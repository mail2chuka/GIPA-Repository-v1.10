#!/usr/bin/env bash
# Laptop verification of experiment E9 (asymmetric RSUs).
#
# Reruns experiments/exp_asym.py into results/res_asym_laptop.json and checks
# that every number agrees with results/res_asym.json (reference run). E9 is
# deterministic, so the expected verdict is IDENTICAL. The planner uses
# SciPy's SLSQP; a different SciPy version could move plan values in the last
# digits, in which case rerun the comparison with a looser tolerance, e.g.
#     python experiments/compare_results.py results/res_asym.json results/res_asym_laptop.json --tol 1e-6
#
# Run from the repository root with the virtual environment active.
# About 5-10 minutes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
python experiments/exp_asym.py --out results/res_asym_laptop.json | tee results/exp_asym_laptop_log.txt
python experiments/compare_results.py results/res_asym.json results/res_asym_laptop.json
