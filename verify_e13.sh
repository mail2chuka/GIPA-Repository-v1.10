#!/usr/bin/env bash
# Laptop verification of experiment E13 (slotted experiments with carried-over
# local work).
#
# Usage: ./verify_e13.sh own|blind
#   1. checks the local-queue chain against direct simulation
#      (experiments/check_carry.py);
#   2. checks that with GIPA_LOCAL_CARRY unset the planner experiment
#      reproduces results/res_planner.json (expected verdict: IDENTICAL);
#   3. reruns every slotted experiment in the chosen mode into
#      results/carry_<mode>_laptop/ and compares each file with
#      results/carry_<mode>/ (expected verdict: IDENTICAL).
#
# Run from the repository root with the virtual environment active.
# About 2 hours (own) or 2.5 hours (blind) on a 2-core laptop.
set -euo pipefail
MODE="${1:?usage: verify_e13.sh own|blind}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
GIPA_LOCAL_CARRY= python experiments/check_carry.py
GIPA_LOCAL_CARRY= python experiments/exp_planner.py --out results/res_planner_idle_laptop.json
python experiments/compare_results.py results/res_planner.json results/res_planner_idle_laptop.json
./run_carry.sh "$MODE" "$ROOT/results/carry_${MODE}_laptop" | tee "results/run_carry_${MODE}_laptop_log.txt"
for f in res_main res_sc res_planner res_calibration res_capacity res_prediction res_prediction_obs res_asym res_la res_hetero res_qlearning; do
  echo "== $f"
  python experiments/compare_results.py "results/carry_${MODE}/$f.json" "results/carry_${MODE}_laptop/$f.json"
done
