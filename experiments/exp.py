"""
exp.py
======
Principal experiment for the GIPA manuscript (IET ITS ITS-2026-08-0579):
matched classical baselines, GIPA ablations and class-resolved metrics in
two workload regimes and four fleet densities, plus the self-consistent
prior audit.

What it runs
------------
1. For each regime (A, B):
   * trains one tabular Q-learner at N = 160 (see ``gipa_core.QL``);
   * for each density N in {80, 160, 240, 320} and each of the 15 policies
     in ``POLICIES`` (10 policies + 5 GIPA ablations), runs 30 matched
     seeds with ``gipa_core.agg`` and stores the across-seed means.
2. Self-consistent prior audit (manuscript Proposition 3, Appendix G.3):
   damped iteration of ``gipa_core.sc_prior`` for every regime and density,
   with convergence verdicts.
3. Regime B prior variants: GIPA with the static, self-consistent,
   self-consistent + 10% margin and workload-derived priors, plus all-local.

Outputs (written to the CURRENT WORKING DIRECTORY, not to results/)
--------------------------------------------------------------------
res_main.json
    {regime: {N: {policy: {metric: mean}}}}; metrics as documented in
    ``gipa_core.run``. Matches results/res_main.json in the archive.
res_sc.json
    {'sc': self-consistent prior audit per regime and N,
     'regB': Regime B prior-variant results per N}.
Console
    Summary tables for N = 160 and the prior audit.

``run_all.sh`` changes into the repository root before calling this script,
so the files appear in the root directory.

What it does NOT produce
------------------------
* results/res_main_seed_level.json (per-seed records with miss and offload
  counts) and results/res_main_statistics.json (bootstrap CIs, paired
  comparisons, TOST, Clopper-Pearson bounds). These were produced by an
  instrumented run and a statistics script that are not in this archive;
  see docs/SCRIPTS.md.

Seeds
-----
Environment 1000-1029, action sampling 5000-5029, Q-learning exploration
5999 (see gipa_core).

Runtime
-------
Several minutes on a laptop CPU; dominated by the 30-seed loops and the
Q-learning training (15,000 slots per regime).

Run
---
    PYTHONPATH=src python experiments/exp.py      # from the repository root
or  ./run_all.sh
"""

import json
import numpy as np

from gipa_core import (CFG, REG, QL, agg, prior_heur, prior_workload,
                       sc_prior, safe_mean)

# (result key, keyword arguments). Keys starting with 'gipa' run the GIPA
# operator with the given ablation switch; 'ql' is filled with the trained
# learner for each regime.
POLICIES = [
    ('greedy',        {}),
    ('indep',         {}),
    ('msoft',         {}),
    ('msoft_prior',   {}),
    ('price',         {}),
    ('entreg',        {}),
    ('rbr',           {}),
    ('ql',            None),          # filled per regime with the trained learner
    ('local',         {}),
    ('gipa',          {}),
    ('gipa_noamp',    {'amp_on': False}),
    ('gipa_nodiff',   {'lam_on': False}),
    ('gipa_fixed',    {'anneal': False}),
    ('gipa_binary',   {'graded': False}),
    ('gipa_noprior',  {'prior_on': False}),
]

DENSITIES = [80, 160, 240, 320]


def main():
    """Run the principal experiment, the prior audit and the Regime B variants."""
    results = {}
    for reg in ['A', 'B']:
        R = REG[reg]
        results[reg] = {}
        # One Q-learner per regime, trained at N = 160 and reused at every density.
        ql = QL(CFG['M'] + 1)
        ql.train(160, R)

        for N in DENSITIES:
            results[reg][N] = {}
            for name, kw in POLICIES:
                base = 'gipa' if name.startswith('gipa') else name
                kwargs = {'ql': ql} if name == 'ql' else dict(kw)
                m, _ = agg(base, N, R, None, **kwargs)
                results[reg][N][name] = m

    with open('res_main.json', 'w') as fh:
        json.dump(results, fh, indent=1)
    print('MAIN DONE')

    for reg in ['A', 'B']:
        print(f'--- regime {reg}  N=160 ---')
        for k, v in results[reg][160].items():
            fair = v['fair']
            fair_s = 'N/A ' if np.isnan(fair) else f'{fair:.2f}'
            print(f"  {k:14s} mS={100*v['mS']:6.1f} mB={100*v['mB']:6.1f} "
                  f"lat={v['lat']:6.2f} p95={v['p95']:6.1f} off={v['off']:.2f} "
                  f"offS={v['offS']:.2f} offB={v['offB']:.2f} fair={fair_s}")

    # ---------------- self-consistent prior ----------------
    print('\n=== SELF-CONSISTENT PRIOR ===')
    SC = {}
    for reg in ['A', 'B']:
        R = REG[reg]
        SC[reg] = {}
        for N in DENSITIES:
            cb, it, res = sc_prior(N, R)
            converged = res < 1e-6
            SC[reg][N] = {'cbar': cb.tolist(), 'iters': it, 'resid': float(res),
                          'converged': bool(converged),
                          'heur': prior_heur(N).tolist()}
            flag = '' if converged else '   <-- DID NOT CONVERGE'
            print(f' reg{reg} N={N}: sc_prior={np.round(cb[1:], 2)} '
                  f'(iters {it}, resid {res:.1e})  '
                  f'heuristic={np.round(prior_heur(N)[1:], 2)}{flag}')

    # ---------------- Regime B prior variants ----------------
    print('\n=== REGIME B with self-consistent prior (and margins) ===')
    OUT = {}
    for N in DENSITIES:
        OUT[N] = {}
        variants = [
            ('static',   prior_heur(N)),
            ('sc',       np.array(SC['B'][N]['cbar'])),
            ('sc_m10',   np.array(SC['B'][N]['cbar']) * 1.10),
            ('workload', prior_workload(N, REG['B'])),
        ]
        for tag, cb in variants:
            m, _ = agg('gipa', N, REG['B'], cb)
            OUT[N][tag] = m
            fair_s = 'N/A ' if np.isnan(m['fair']) else f"{m['fair']:.2f}"
            print(f" N={N} {tag:9s} mS={100*m['mS']:6.1f} mB={100*m['mB']:5.1f} "
                  f"lat={m['lat']:6.2f} offS={m['offS']:.2f} offB={m['offB']:.2f} "
                  f"fair={fair_s}")
        ml, _ = agg('local', N, REG['B'])
        print(f" N={N} {'ALL-LOCAL':9s} mS={100*ml['mS']:6.1f} "
              f"mB={100*ml['mB']:5.1f} lat={ml['lat']:6.2f}")

    with open('res_sc.json', 'w') as fh:
        json.dump({'sc': SC, 'regB': OUT}, fh, indent=1)


if __name__ == '__main__':
    main()
