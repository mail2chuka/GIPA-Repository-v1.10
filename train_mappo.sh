#!/usr/bin/env bash
# Workplan experiment E7: train and evaluate the MAPPO baseline.
#
# Designed for a 2-core laptop: training runs in two processes in parallel
# (one thread each). Run from the repository root, inside the virtual
# environment, ideally with sleep blocked and the laptop plugged in:
#
#   source .venv/bin/activate
#   systemd-inhibit --what=sleep:idle nohup ./train_mappo.sh > train_mappo_log.txt 2>&1 &
#   tail -f train_mappo_log.txt          # watch progress; Ctrl+C stops watching, not training
#
# Steps
#   1. bench   time a short run and print the expected duration
#   2. tune    6-configuration grid at N = 80 (results/mappo/tune.json)
#   3. train   5 seeds x 4 densities, affine model (the submitted model)
#   4. train   5 seeds x 4 densities, Eq. (5) model from experiment E2
#   5. train   'load' variant (actor also sees last slot's RSU loads), N = 160
#   6. merge   -> results/res_mappo.json
#
# Expected duration on an Intel i5-4210M: very roughly 1-3 hours in total
# (step 1 prints this machine's own estimate). See docs/SCRIPTS.md.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ROOT"
E=experiments/exp_mappo.py

echo "== $(date) bench";  python $E --stage bench
echo "== $(date) tune";   python $E --stage tune

for MODEL in affine ps; do
  echo "== $(date) train model=$MODEL"
  python $E --stage train --model $MODEL --seeds 0 1 2 --part a &
  python $E --stage train --model $MODEL --seeds 3 4   --part b &
  wait
done

echo "== $(date) train variant=load (N = 160)"
python $E --stage train --variant load --densities 160 --seeds 0 1 2 --part a &
python $E --stage train --variant load --densities 160 --seeds 3 4   --part b &
wait

echo "== $(date) merge";  python $E --stage merge
echo "== $(date) done"
