#!/usr/bin/env bash
# run_complexity_repeats.sh
# -------------------------
# Workplan E8 (decision timings, manuscript Section VII-H and Fig. R8).
# Runs experiments/exp_complexity.py REPEATS times (default 5) on an
# otherwise idle machine, single-threaded, then combines the runs with
# experiments/aggregate_complexity.py (median across runs, with the
# run-to-run range) and redraws Fig. R8.
#
# STATUS: written for the first revision of ITS-2026-08-0579.
# Runtime: about 1 minute per repeat on the reference machine (2 vCPU
# Xeon); a few minutes per repeat on a 2014 laptop CPU.
#
# Usage (from the repository root):
#   bash run_complexity_repeats.sh            # 5 repeats -> results/res_complexity.json
#   REPEATS=3 OUT=results/res_complexity_mine.json NOPLOT=1 bash run_complexity_repeats.sh
# NOPLOT=1 skips redrawing Fig. R8 (use it when timing a second machine).
set -euo pipefail
cd "$(dirname "$0")"
REPEATS="${REPEATS:-5}"
OUT="${OUT:-results/res_complexity.json}"
DIR="results/complexity_runs$( [ "$OUT" = results/res_complexity.json ] || echo _$(basename "$OUT" .json) )"
mkdir -p "$DIR"
export PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for i in $(seq 1 "$REPEATS"); do
  echo "== $(date) timing run $i of $REPEATS"
  python experiments/exp_complexity.py --out "$DIR/run$i.json" > "$DIR/run$i.log" 2>&1
done
python experiments/aggregate_complexity.py $(for i in $(seq 1 "$REPEATS"); do echo "$DIR/run$i.json"; done) --out "$OUT"
if [ -z "${NOPLOT:-}" ]; then
  python plotting/revision_figures.py --figure 8 --complexity "$(basename "$OUT")"
fi
