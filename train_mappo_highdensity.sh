#!/usr/bin/env bash
# Workplan experiment E7, follow-up: MAPPO at N = 240 and 320 with the
# deadline-only mask, under both latency models, plus a check against the
# reference run.
#
# Why: the main run gave MAPPO GIPA's prior-inclusive deadline mask, which at
# N >= 240 forbids every RSU, so MAPPO could only stay local there. With the
# deadline-only mask MAPPO can offload and learns congestion from the miss
# penalty. Uses the tuned configuration in results/mappo/tune.json.
#
# Run from the repository root inside the virtual environment:
#   source .venv/bin/activate
#   systemd-inhibit --what=sleep:idle nohup ./train_mappo_highdensity.sh > train_mappo_highdensity_log.txt 2>&1 &
#   tail -f train_mappo_highdensity_log.txt
#
# Expected duration on an Intel i5-4210M: roughly 1.5-2 hours.
# At the end, results/res_mappo.json also contains the new groups (keys ending
# in '|deadline'), and they are compared with the reference run in
# results/res_mappo_dmask_reference.json (tolerance 0.05 ms).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ROOT"
E=experiments/exp_mappo.py

for MODEL in affine ps; do
  echo "== $(date) train model=$MODEL mask=deadline N=240,320"
  python $E --stage train --model $MODEL --mask deadline --densities 240 320 --seeds 0 1 2 --part a &
  python $E --stage train --model $MODEL --mask deadline --densities 240 320 --seeds 3 4   --part b &
  wait
done

echo "== $(date) merge";  python $E --stage merge
if [ -f results/res_mappo_dmask_reference.json ]; then
  echo "== $(date) compare with the reference run"
  python experiments/compare_mappo.py results/res_mappo.json results/res_mappo_dmask_reference.json || true
fi
echo "== $(date) done"
