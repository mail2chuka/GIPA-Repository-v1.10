#!/usr/bin/env bash
# Laptop verification of experiment E5b (asynchronous arrivals with a
# persistent on-board processor per vehicle).
#
# Reruns experiments/exp_async_local.py into results/res_async_local_laptop.json
# and checks every number against results/res_async_local.json (reference run).
# The simulator is deterministic, so the expected verdict is IDENTICAL. The run
# also prints the M/D/1 and local-admission checks.
#
# Run from the repository root with the virtual environment active.
# About 60-90 minutes on a 2-core laptop.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
python experiments/exp_async_local.py --out results/res_async_local_laptop.json | tee results/exp_async_local_laptop_log.txt
python experiments/compare_results.py results/res_async_local.json results/res_async_local_laptop.json
