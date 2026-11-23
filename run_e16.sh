#!/usr/bin/env bash
# Experiment E16: SLA-MA-PPO (Georgiades et al., IEEE TNSM 2026), a recent
# task-specific vehicular DRL method, against GIPA's planned policies.
#
# Usage: ./run_e16.sh [job]
#   no argument  every stage below in order (about 8 hours on one core;
#                about 4 hours if the six train jobs run two at a time)
#   tune         tuning grid, homog and het50 in parallel (about 20 minutes)
#   1 ... 6      one training job (each about 60-90 minutes on one core):
#                  1 homog idle N = 80 160 240     2 homog idle N = 320, homog own N = 80 160
#                  3 homog own N = 240 320         4 het50 idle N = 80 160 240
#                  5 het50 idle N = 320, het50 own N = 80 160
#                  6 het50 own N = 240 320
#   merge        results/res_sla.json, then results/sla_summary.{json,md}
#   diag         load-spreading diagnostics at N = 160 (results/sla_diagnostics.json)
#
# 'own' sets GIPA_LOCAL_CARRY=own (carried-over local work, the vehicle knows
# its own backlog). Runs that already have a result file are skipped, so a job
# can be restarted. Source: src/sla_mappo.py; driver: experiments/exp_sla.py;
# comparison: experiments/compile_sla.py. See docs/SCRIPTS.md.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ROOT"
E=experiments/exp_sla.py
mkdir -p results/sla
tr() { echo "== $(date +%H:%M:%S) $*"; }
job() {
  case "$1" in
    tune) tr tune
          python $E --stage tune --setting homog > results/sla/tune_homog_log.txt 2>&1 &
          python $E --stage tune --setting het50 > results/sla/tune_het50_log.txt 2>&1 &
          wait ;;
    1) tr job 1; python $E --stage train --setting homog --densities 80 160 240 ;;
    2) tr job 2; python $E --stage train --setting homog --densities 320
                 GIPA_LOCAL_CARRY=own python $E --stage train --setting homog --densities 80 160 ;;
    3) tr job 3; GIPA_LOCAL_CARRY=own python $E --stage train --setting homog --densities 240 320 ;;
    4) tr job 4; python $E --stage train --setting het50 --densities 80 160 240 ;;
    5) tr job 5; python $E --stage train --setting het50 --densities 320
                 GIPA_LOCAL_CARRY=own python $E --stage train --setting het50 --densities 80 160 ;;
    6) tr job 6; GIPA_LOCAL_CARRY=own python $E --stage train --setting het50 --densities 240 320 ;;
    merge) tr merge; python $E --stage merge; python experiments/compile_sla.py ;;
    diag) tr diag; python experiments/diag_sla.py; GIPA_LOCAL_CARRY=own python experiments/diag_sla.py ;;
    *) echo "unknown job $1"; exit 1 ;;
  esac
}
if [ $# -eq 0 ]; then
  for j in tune 1 2 3 4 5 6 merge diag; do job "$j"; done
else
  job "$1"
fi
tr done
