#!/usr/bin/env bash
# Laptop verification of experiment E1b (prediction observability and planned
# policies that read per-task predictions).
#
# Reruns experiments/exp_prediction_obs.py into
# results/res_prediction_obs_laptop.json and checks every number against
# results/res_prediction_obs.json (reference run). Expected verdict: IDENTICAL.
#
# Run from the repository root with the virtual environment active.
# About 10-15 minutes on a 2-core laptop.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
python experiments/exp_prediction_obs.py --out results/res_prediction_obs_laptop.json | tee results/exp_prediction_obs_laptop_log.txt
python experiments/compare_results.py results/res_prediction_obs.json results/res_prediction_obs_laptop.json
