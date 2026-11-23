#!/usr/bin/env bash
# Laptop verification of experiment E16 (SLA-MA-PPO, ref. [24]).
#
# Usage: ./verify_e16.sh
#   1. retrains SLA-MA-PPO for one fleet size and run (identical tasks, idle
#      processor, N = 80, run 0) into results/sla_verify/ and compares the
#      evaluation with results/sla/homog_idle_N80_r0.json (expected verdict:
#      IDENTICAL on the same NumPy version; floating-point differences across
#      versions can change the sampled actions and the checkpoint chosen);
#   2. recompiles results/sla_summary.json from results/res_sla.json into
#      results/sla_summary_check.json and compares the two.
# About 3 minutes on one core. Run from the repository root.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ROOT"
rm -f results/sla_verify/homog_idle_N80_r0.json
python experiments/exp_sla.py --stage train --setting homog --densities 80 --seeds 0 --outdir results/sla_verify
python - <<'EOF'
import json
a = json.load(open('results/sla/homog_idle_N80_r0.json'))
b = json.load(open('results/sla_verify/homog_idle_N80_r0.json'))
json.dump(dict(mean=a['mean'], per_seed=a['per_seed'], chosen_iter=a['chosen_iter']), open('results/sla_verify/archived_eval.json', 'w'))
json.dump(dict(mean=b['mean'], per_seed=b['per_seed'], chosen_iter=b['chosen_iter']), open('results/sla_verify/rerun_eval.json', 'w'))
EOF
python experiments/compare_results.py results/sla_verify/archived_eval.json results/sla_verify/rerun_eval.json
cp results/sla_summary.json results/sla_summary_archived.json
python experiments/compile_sla.py > /dev/null
mv results/sla_summary.json results/sla_summary_check.json
mv results/sla_summary_archived.json results/sla_summary.json
python experiments/compare_results.py results/sla_summary.json results/sla_summary_check.json
