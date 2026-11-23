#!/usr/bin/env bash
# Run every experiment and regenerate the figures.
#
#   1. experiments/exp.py          principal experiment, prior audit and Regime B
#                                  prior variants; writes res_main.json and
#                                  res_sc.json to the repository root
#   2. experiments/exp_planner.py  planner versus GIPA and all-local; writes
#                                  results/res_planner.json
#   3. experiments/exp_capacity.py capacity experiment E3, soft and admission
#                                  control; writes results/res_capacity.json
#   4. experiments/exp_prediction.py prediction-error experiment E1; writes
#                                  results/res_prediction.json
#   5. experiments/exp_calibration.py Eq. (5) experiment E2; writes
#                                  results/res_calibration.json
#   6. experiments/exp_qlearning.py Q-learning experiment E4; writes
#                                  results/res_qlearning.json
#   7. experiments/exp_la.py       learning-automata experiment E6; writes
#                                  results/res_la.json
#   8. experiments/exp_complexity.py complexity and timing E8; writes
#                                  results/res_complexity.json (timings are
#                                  machine dependent)
#   9. experiments/exp_asym.py     asymmetric-RSU experiment E9; writes
#                                  results/res_asym.json
#  10. experiments/exp_async.py    asynchronous-arrival experiment E5; writes
#                                  results/res_async.json (about 20-40 minutes)
#  10b. experiments/exp_async_local.py E5b: asynchronous arrivals with a
#                                  persistent on-board processor per vehicle;
#                                  writes results/res_async_local.json (about 60-90 minutes)
#  11. experiments/exp_hetero.py   heterogeneous-task experiment E12; writes
#                                  results/res_hetero.json
#  12. plotting/generate_figures.py  Figs. 3-8 from results/res_main_statistics.json
#  13. plotting/revision_figures.py  revision Figs. R1-R9 from the result files
#
# See docs/SCRIPTS.md for details. Roughly 1-1.5 hours on a laptop CPU. MAPPO (E7) runs separately: train_mappo.sh.
# E13 (the slotted experiments with carried-over local work) also runs separately:
# ./run_carry.sh own, ./run_carry.sh blind, then python experiments/compile_carry.py.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
python experiments/exp.py
python experiments/exp_planner.py
python experiments/exp_capacity.py
python experiments/exp_prediction.py
python experiments/exp_prediction_obs.py
python experiments/exp_calibration.py
python experiments/exp_qlearning.py
python experiments/exp_la.py
python experiments/exp_complexity.py
python experiments/exp_asym.py
python experiments/exp_async.py
python experiments/exp_async_local.py
python -W ignore experiments/exp_hetero.py
python plotting/generate_figures.py
python plotting/revision_figures.py
