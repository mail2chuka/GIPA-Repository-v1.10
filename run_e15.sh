#!/bin/bash
# E15: GIPA's per-task layer against a re-solved probabilistic plan (Tan-style). Run from the repository root.
set -e
SRC=$(pwd)/src
python -I experiments/exp_tan.py --src "$SRC" --part tune  --out results/tune_batch.json --workers 2
python -I experiments/exp_tan.py --src "$SRC" --part batch --chosen results/tune_batch.json --out results/res_tan_batch.json --workers 2
GIPA_LOCAL_CARRY=own python -I experiments/exp_tan.py --src "$SRC" --part carry --chosen results/carry_own/res_hetero.json --out results/res_tan_carry.json --workers 2
python -I experiments/exp_tan.py --src "$SRC" --part fluct --chosen results/tune_batch.json --out results/res_tan_fluct.json --workers 2
TAN_RELAX=1 GIPA_LOCAL_CARRY=own python -I experiments/sens.py
python -I experiments/summarise.py > /dev/null
