#!/usr/bin/env bash
# Experiment E17: backlog-aware screening of the planned prior with carried-over
# local work (identical tasks, Regime A). Source: src/backlog.py; driver:
# experiments/exp_backlog.py. Tunes the minimum random offload probability q_min at
# N = 160 on seeds 2000-2009, evaluates N = 80-320 on the 30 archived seeds and
# compares with the carry-aware planned prior (E13) and SLA-MA-PPO (E16).
# Writes results/res_backlog.json and results/backlog_summary.md. About 2 minutes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" GIPA_LOCAL_CARRY=own
cd "$ROOT"
python experiments/exp_backlog.py | tee results/run_e17_log.txt
