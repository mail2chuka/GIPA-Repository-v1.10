#!/bin/bash
# E14: value-based multi-agent DRL baselines (independent DQN, VDN) and Regime B overload rates. Run from the repository root.
set -e
python -I experiments/exp_marl_value.py --src src --out results/res_marl_value.json --N 80 160 240 320 --runs 5 --slots 4000 > results/exp_marl_value_log.txt 2>&1
python -I experiments/exp_marl_value.py --src src --out results/res_marl_value_dmask.json --N 240 320 --runs 5 --slots 4000 --mask deadline > results/exp_marl_value_dmask_log.txt 2>&1
python -I experiments/check_overload_regimeB.py src > results/check_overload_regimeB_log.txt 2>&1
