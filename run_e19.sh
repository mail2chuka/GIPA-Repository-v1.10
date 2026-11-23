#!/usr/bin/env bash
# Experiment E19: SLA-MA-PPO with completion-gated feedback. The learner uses a
# task's latency only once the task has completed and its deadline outcome only
# once the task has completed or its deadline has passed (GIPA_SLA_FEEDBACK=gated,
# see src/sla_mappo.py). E16 (run_e16.sh) gives every outcome of a slot at the
# next decision, including outcomes of tasks still running.
#
# Usage: ./run_e19.sh [job]
#   no argument  every stage below in order (about 8.5 hours on one core;
#                about 4.5 hours if the six train jobs run two at a time)
#   reeval       archived E16 actors evaluated with gated feedback (a few minutes)
#   1 ... 6      one training job, as in run_e16.sh, with the E16 tuned settings
#                (results/sla/tune_<setting>.json); each about 60-90 minutes
#   compile      results/sla_gated_summary.{json,md} (experiments/compile_sla_gated.py)
#
# Run files go to results/sla_gated/; runs that already have a file are skipped.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" GIPA_SLA_FEEDBACK=gated
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ROOT"
E="experiments/exp_sla.py --outdir results/sla_gated"
mkdir -p results/sla_gated
tr() { echo "== $(date +%H:%M:%S) $*"; }
job() {
  case "$1" in
    reeval) tr reeval
            for st in homog het50; do
              python $E --stage reeval --setting $st
              GIPA_LOCAL_CARRY=own python $E --stage reeval --setting $st
            done ;;
    1) tr job 1; python $E --stage train --setting homog --densities 80 160 240 ;;
    2) tr job 2; python $E --stage train --setting homog --densities 320
                 GIPA_LOCAL_CARRY=own python $E --stage train --setting homog --densities 80 160 ;;
    3) tr job 3; GIPA_LOCAL_CARRY=own python $E --stage train --setting homog --densities 240 320 ;;
    4) tr job 4; python $E --stage train --setting het50 --densities 80 160 240 ;;
    5) tr job 5; python $E --stage train --setting het50 --densities 320
                 GIPA_LOCAL_CARRY=own python $E --stage train --setting het50 --densities 80 160 ;;
    6) tr job 6; GIPA_LOCAL_CARRY=own python $E --stage train --setting het50 --densities 240 320 ;;
    compile) tr compile; python experiments/compile_sla_gated.py ;;
    *) echo "unknown job $1"; exit 1 ;;
  esac
}
if [ $# -eq 0 ]; then
  for j in reeval 1 2 3 4 5 6 compile; do job "$j"; done
else
  job "$1"
fi
tr done
