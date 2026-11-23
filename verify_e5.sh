#!/usr/bin/env bash
# Laptop verification of experiment E5 (asynchronous arrivals).
#
# Reruns experiments/exp_async.py into results/res_async_laptop.json and
# checks every number against results/res_async.json (reference run). The
# event-driven simulator is deterministic, so the expected verdict is
# IDENTICAL. The run also prints the rebuilt Table III check.
#
# Run from the repository root with the virtual environment active.
# About 20-40 minutes on a 2-core laptop.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
python experiments/exp_async.py --out results/res_async_laptop.json | tee results/exp_async_laptop_log.txt
python experiments/compare_results.py results/res_async.json results/res_async_laptop.json
