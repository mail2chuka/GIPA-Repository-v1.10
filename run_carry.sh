#!/usr/bin/env bash
# Experiment E13: rerun the slotted experiments with carried-over local work.
#
# Usage: ./run_carry.sh own|blind [output directory]
#   own   each vehicle also uses its own local backlog (the specified GIPA:
#          own queue in the mask (8); planned policies redraw among the plan's
#          RSUs when local execution is predicted to miss)
#   blind  decisions as in the paper; only realised latency includes the wait
#
# Sets GIPA_LOCAL_CARRY (see src/gipa_core.py) and writes every result file to
# results/carry_<mode>/ (or the given directory) with the same name as the idle-processor result. The
# planners become carry-aware (src/carry.py). MAPPO is not rerun: its trained
# actors are not stored, and its observation has no backlog.
# The archived checks inside the scripts (symmetric reproduction, archived
# Q-learning protocol) compare with idle-processor files and are expected to
# report differences under this mode.
set -euo pipefail
MODE="${1:?usage: run_carry.sh own|blind}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="${2:-$ROOT/results/carry_$MODE}"
mkdir -p "$OUT"
export GIPA_LOCAL_CARRY="$MODE"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$OUT"
run() { echo "== $(date +%H:%M:%S) $*"; "$@"; }
run python "$ROOT/experiments/exp.py"                                  # writes res_main.json, res_sc.json here
run python "$ROOT/experiments/exp_planner.py"       --out "$OUT/res_planner.json"
run python "$ROOT/experiments/exp_calibration.py"   --out "$OUT/res_calibration.json"
run python "$ROOT/experiments/exp_capacity.py"      --out "$OUT/res_capacity.json"
run python "$ROOT/experiments/exp_prediction.py"    --out "$OUT/res_prediction.json"
run python "$ROOT/experiments/exp_prediction_obs.py" --out "$OUT/res_prediction_obs.json"
run python "$ROOT/experiments/exp_asym.py"          --out "$OUT/res_asym.json"
run python "$ROOT/experiments/exp_la.py"            --out "$OUT/res_la.json"
run python -W ignore "$ROOT/experiments/exp_hetero.py" --out "$OUT/res_hetero.json"
run python "$ROOT/experiments/exp_qlearning.py"     --out "$OUT/res_qlearning.json"
echo "== $(date +%H:%M:%S) done"
