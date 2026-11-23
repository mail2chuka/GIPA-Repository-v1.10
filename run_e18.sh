#!/usr/bin/env bash
# Experiment E18: queue priority at the original targets. Backlog screening
# (E17, run_e17.sh) re-optimises the offload target and changes which
# background vehicles offload; E18 keeps the targets of the carry-aware planned
# prior (E13) and changes only the order, so the two effects can be separated.
#
# Usage: ./run_e18.sh          (about two minutes on one core)
# Output: results/res_priority.json, results/priority_summary.md
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$ROOT/src:$ROOT/experiments${PYTHONPATH:+:$PYTHONPATH}" GIPA_LOCAL_CARRY=own
cd "$ROOT"
python experiments/exp_priority.py
