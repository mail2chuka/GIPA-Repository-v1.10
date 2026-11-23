# Script catalogue

This catalogue documents every script in the GIPA reproducibility repository: what it does, what it reads and writes, which seeds it uses, how long it takes, and which manuscript results it produces. It also lists the scripts the manuscript refers to that are not yet in the repository, and the known issues in the archived code.

Public version 1.10: 23 November 2026. The original catalogue and historical verification notes are retained below, with later E13–E19 entries. For the current paper, start with `MANUSCRIPT_MAP.md`; E19 is the primary SLA-MA-PPO comparison.

## At a glance

| Script | Role | Status | Writes | Runtime (laptop CPU) |
| --- | --- | --- | --- | --- |
| `src/gipa_core.py` | Simulator, GIPA operator, baselines, priors, episode runner | v1.0.0 code, documented | Nothing (library) | — |
| `experiments/exp.py` | Principal experiment, prior audit, Regime B prior variants | v1.0.0 code, documented | `res_main.json`, `res_sc.json` in the working directory | Several minutes |
| `plotting/generate_figures.py` | Figs. 3–8 from archived statistics | v1.0.0 code, reformatted and documented | `figures/fig3`–`fig8` (PNG, PDF, SVG) | Under 1 minute |
| `src/planner.py` | System-optimal planned allocation | New, exploratory (revision) | Nothing (library) | — |
| `experiments/exp_planner.py` | Planner versus GIPA and all-local; misspecification sweep | New, exploratory (revision) | `results/res_planner.json` | A few minutes |
| `src/capacity.py` | RSU capacity, admission layer wrapping any policy, selected vs admitted load | New (revision, E3) | Nothing (library) | — |
| `experiments/exp_capacity.py` | E3: every policy in soft and admission-control mode | New (revision, E3) | `results/res_capacity.json` | Several minutes |
| `src/prediction.py` | Predicted instead of oracle latencies: error arms, drifting truth, EWMA predictor, guard band | New (revision, E1) | Nothing (library) | — |
| `experiments/exp_prediction.py` | E1: prediction-error robustness, 45 conditions | New (revision, E1) | `results/res_prediction.json` | A few minutes |
| `src/calibration.py` | Physical calibration; Eq. (5) processor-sharing environment; class-aware prior; Eq. (5) planner | New (revision, E2) | Nothing (library) | — |
| `experiments/exp_calibration.py` | E2: comparison, density sweep, ablation under Eq. (5); RSU capacity sweep | New (revision, E2) | `results/res_calibration.json` | A few minutes |
| `src/qlearning.py` | Configurable tabular Q-learning: multiple training runs, disjoint training environments, episode reset, discount, two-level bootstrap | New (revision, E4) | Nothing (library) | — |
| `experiments/exp_qlearning.py` | E4: 10 training runs per configuration, both regimes and latency models | New (revision, E4) | `results/res_qlearning.json` | About 5–15 minutes |
| `src/learning_automata.py` | Per-vehicle learning automata (L_R-I, L_R-εP, pursuit), episode and fleet-size-schedule runners | New (revision, E6) | Nothing (library) | — |
| `experiments/exp_la.py` | E6: tuning grid, main comparison with warm-up 25 and 500, density-change test | New (revision, E6) | `results/res_la.json` | About 2–10 minutes |
| `experiments/exp_complexity.py` | E8: per-slot and per-vehicle decision time, Stage 0 / planner cost, training cost, runtime communication | New (revision, E8) | `results/res_complexity.json` | About 1–5 minutes (idle machine) |
| `src/hetero.py` | Heterogeneous tasks, uneven links and partial coverage; GIPA on the plan (Framing A), plan with per-task ranking (Framing B) | New (revision, E12) | Nothing (library) | — |
| `experiments/exp_hetero.py`, `verify_e12.sh` | E12: six scenarios, tuning of GIPA on the plan, nine policies, paired CIs | New (revision, E12) | `results/res_hetero.json` | About 6–20 minutes |
| `src/stats.py`, `experiments/stats_main.py` | Seed-level statistics (bootstrap, paired differences, TOST, Clopper–Pearson); regenerates `res_main_statistics.json` | Rebuilt (revision) | `results/res_main_statistics_rebuilt.json` | Seconds |
| `experiments/seed_level_main.py` | Principal experiment with per-seed counts; regenerates `res_main_seed_level.json` byte for byte | Rebuilt (revision) | `results/res_main_seed_level_rebuilt.json` | About 1 minute |
| `verify_stats.sh` | Laptop check: seed-level rebuild, then statistics from it, both against the archive | New (revision) | Both rebuilt files | 1–3 minutes |
| `src/async_sim.py` | Event-driven simulator: Poisson arrivals, persistent processor-sharing RSU queues, broadcast and feedback policies, asynchronous planner; rebuilds the Table III check | New (revision, E5) | Nothing (library) | — |
| `experiments/exp_async.py` | E5: asynchronous arrivals with carried-over backlog, 20 policy variants, both regimes, six fleet sizes, 30 seeds | New (revision, E5) | `results/res_async.json` | About 20–40 minutes |
| `experiments/exp_prediction_obs.py`, `verify_e1b.sh` | E1b: what a vehicle observes (EWMA on the uncongested breakdown, on realised latency, or on realised latency minus the prior's congestion) and the planned policy when it reads per-task predictions (mask, advantage, margin-load mask) | New (revision, E1b, Nov 2026) | `results/res_prediction_obs.json` | About 10–15 minutes |
| `experiments/exp_async_local.py`, `verify_e5b.sh` | E5b: E5 with a persistent on-board processor per vehicle (FIFO; processor sharing as sensitivity), own-queue-aware GIPA and a local-queue-aware planner | New (revision, E5b, Nov 2026) | `results/res_async_local.json` | About 60–90 minutes |
| `src/carry.py` | Carried-over local work in the slotted simulators: stationary local-queue chain, sampling share under own-queue redirects, carry-aware fixed-point planning | New (revision, E13, Nov 2026) | Nothing (library) | — |
| `run_carry.sh`, `experiments/check_carry.py`, `experiments/compile_carry.py`, `verify_e13.sh` | E13: every slotted experiment except MAPPO rerun with carried-over local work (`own` and `blind` modes); chain check; summary tables | New (revision, E13, Nov 2026) | `results/carry_own/`, `results/carry_blind/`, `results/carry_check.json`, `results/carry_summary.{json,md}` | About 2 hours (own) or 2.5 hours (blind) on 2 cores |
| `src/sla_mappo.py` | SLA-MA-PPO of Georgiades et al. (IEEE TNSM 2026, manuscript ref. [24]) in the slotted simulators: shared actor and critic, SLA admissibility mask, learned full-service proxy and risk, SLA reward with fleet-KPI penalties, checkpoint selection on tuning seeds | New (revision, E16, Nov 2026) | Nothing (library) | — |
| `run_e16.sh`, `experiments/exp_sla.py`, `experiments/compile_sla.py`, `experiments/diag_sla.py`, `verify_e16.sh` | E16: tuning, 80 training runs (identical and heterogeneous tasks, idle and carried-over local work, N = 80–320, five runs), paired comparison with the planned policies, load-spreading diagnostics | New (revision, E16, Nov 2026) | `results/sla/`, `results/res_sla.json`, `results/sla_summary.{json,md}`, `results/sla_diagnostics.json`, `results/sla_long/` | About 8.4 core-hours (six jobs of 60–90 minutes; two at a time on 2 cores) |
| `src/backlog.py`, `experiments/exp_backlog.py`, `run_e17.sh` | E17: backlog-aware screening of the planned prior with carried-over local work (identical tasks): exact backlog chain, screened plan, simulation, paired comparison with the carry-aware planned prior and SLA-MA-PPO | New (revision, E17, Nov 2026) | `results/res_backlog.json`, `results/backlog_summary.md`, `results/run_e17_log.txt` | About 2 minutes |
| `experiments/exp_priority.py`, `run_e18.sh` | E18: backlog screening at the offload targets of the carry-aware planned prior (queue priority only), paired with the planned prior | New (revision, E18, Nov 2026) | `results/res_priority.json`, `results/priority_summary.md` | About 2 minutes |
| `run_e19.sh`, `experiments/compile_sla_gated.py` (and `GIPA_SLA_FEEDBACK=gated` in `src/sla_mappo.py`, `--stage reeval` in `experiments/exp_sla.py`) | E19: SLA-MA-PPO with completion-gated feedback: archived actors re-evaluated, 80 runs retrained; comparison with fixed GIPA variants with seed-level and two-level bootstrap intervals | New (revision, E19, Nov 2026) | `results/sla_gated/`, `results/sla_gated_summary.{json,md}`, `results/sla_gated_*_log.txt` | Re-evaluation a few minutes; retraining about 8.5 core-hours |
| `src/asymmetric.py` | Asymmetric RSUs: speed factors for both latency models, uniform and capacity priors, asymmetric planner, herding metrics | New (revision, E9) | Nothing (library) | — |
| `experiments/exp_asym.py` | E9: anti-herding under asymmetric RSUs, all policies, both regimes and latency models | New (revision, E9) | `results/res_asym.json` | About 5–10 minutes |
| `verify_e5.sh` | Laptop check of E5 | New (revision, E5) | `results/res_async_laptop.json` and log | About 20–40 minutes |
| `verify_e9.sh`, `experiments/compare_results.py` | Laptop check of E9; generic number-by-number JSON comparison | New (revision, E9) | `results/res_asym_laptop.json` and log | About 5–10 minutes |
| `verify_e6_e8.sh` | Laptop check: reruns E6 and compares with the reference file; times E8 on a second machine (not used in the paper) | New (revision, E6/E8) | `results/res_la_laptop.json`, `results/res_complexity_laptop.json` and logs | About 10–20 minutes |
| `src/sysinfo.py` | Records the computing environment (processor, cores, memory, virtualisation, OS, kernel, C library, Python build, library versions, BLAS, thread settings) | New (revision, E8) | Nothing (library) | — |
| `run_complexity_repeats.sh`, `experiments/aggregate_complexity.py` | E8 timings: five single-threaded runs, combined as the median across runs with the run-to-run range; redraws Fig. R8 | New (revision, E8) | `results/res_complexity.json`, `results/complexity_runs/` | About 3–4 minutes on the reference machine |
| `src/mappo.py` | MAPPO baseline in pure NumPy (parameter sharing, centralised critic, masked actions) | New (revision, E7) | Nothing (library) | — |
| `experiments/exp_mappo.py` | E7 stages: bench, tune, train, merge | New (revision, E7) | `results/mappo/*.json`, `results/res_mappo.json` | See `train_mappo.sh` |
| `train_mappo.sh` | Runs all E7 stages, two training processes in parallel | New (revision, E7) | As above | 2 h 51 min on an i5-4210M (27 Oct run) |
| `train_mappo_highdensity.sh` | E7 follow-up: N = 240 and 320 with the deadline-only mask, both latency models, then a comparison with the reference-machine run | New (revision, E7) | `results/mappo/train_*_dmask_*.json`, `results/res_mappo.json` | Roughly 1.5–2 hours on an i5-4210M |
| `experiments/compare_mappo.py` | Checks that two machines' MAPPO results agree within a tolerance | New (revision, E7) | Console only | Seconds |
| `plotting/revision_figures.py` | Nine summary figures for the revision (Figs. R1–R9, manuscript Figs. 3–11), with seed-level bootstrap CIs; all wording in one EDIT HERE block | New (revision) | `figures/revision/fig_r*.{png,pdf,svg,tif}`, `results/res_revision_figures.json` | About 15 seconds |
| `plotting/gipa_framework.py` | GIPA framework diagram (manuscript Fig. 2); all wording in one EDIT HERE block | New (revision) | `figures/revision/fig_gipa_framework.{png,pdf,svg,tif}` | Seconds |
| `plotting/concept_figures.py` | Seven explanatory figures C1–C7 (architecture, framework, herding, equilibrium against optimum, operator walkthrough, deadline margin, processor sharing); wording in the TEXT block, numbers in PARAMS; historical concept figures retained as assets | New (revision) | `figures/concept/fig_c*.{png,pdf,svg,tif}` | About 10 seconds |
| `plotting/export_submission_tiffs.py` | Copies the manuscript figures to journal-ready TIFFs numbered as in the manuscript, checks size and width, writes a manifest | New (revision) | `figures/submission/Figure_1.tif` … `Figure_8.tif`, `MANIFEST.txt` (pass the clean manuscript .docx: Figs. 2 and 8 are taken from it) | Seconds |
| `run_all.sh` | Runs the experiments, then the figures | Updated | As above | Roughly 15–30 minutes |

All Python scripts need `src/` on the import path. From the repository root:

```bash
python -m pip install -r requirements.txt
PYTHONPATH=src python experiments/exp.py
PYTHONPATH=src python experiments/exp_planner.py
PYTHONPATH=src python experiments/exp_capacity.py
PYTHONPATH=src python experiments/exp_prediction.py
PYTHONPATH=src python experiments/exp_calibration.py
PYTHONPATH=src python experiments/exp_qlearning.py
PYTHONPATH=src python experiments/exp_la.py
PYTHONPATH=src python experiments/exp_complexity.py
PYTHONPATH=src python experiments/exp_asym.py
PYTHONPATH=src python experiments/exp_async.py
PYTHONPATH=src python experiments/exp_async_local.py
PYTHONPATH=src python experiments/exp_prediction_obs.py
PYTHONPATH=src python experiments/exp_hetero.py
python plotting/generate_figures.py
PYTHONPATH=src python plotting/revision_figures.py
python plotting/gipa_framework.py
python plotting/export_submission_tiffs.py path/to/ITS-2026-08-0579_R1_manuscript_clean.docx
```

or simply `./run_all.sh`.

## Conventions used everywhere

- **Units.** Latencies, deadlines, priors and intrinsic scores are in milliseconds. Energy is in dimensionless stylized units. Offload and miss rates are fractions in the JSON files and percentages in the figures.
- **Seeds.** Seed index *s* = 0…29. The environment uses seed 1000 + *s*, action sampling uses 5000 + *s*, prediction noise and drift (E1) use 7000 + *s*, and admission order (E3) uses 9000 + *s*. Q-learning exploration uses 5999 (archived) and 6000–6009 (E4). Learning-automata tuning uses seed indices 4500–4509 and the pursuit tie-break jitter uses 9800 + *s* (E6). The bootstrap uses 20260804 with 10,000 resamples. Every policy sees the same environment and action seeds, so results are paired across policies.
- **Episodes.** 70 slots per episode; the first 25 are warm-up; 45 are evaluated.
- **Regimes.** Regime A: local execution meets every deadline. Regime B: local safety execution (14 ms) misses the 12 ms safety deadline. Values are in `gipa_core.REG`.
- **Metric keys.** `mS`, `mB` miss rate (safety, background); `lat` mean latency; `p95` per-slot 95th-percentile latency averaged over slots; `fair` Jain index over RSU loads; `off`, `offS`, `offB` offload ratio (overall, safety, background); `en` stylized energy per task; `maxl` largest RSU load per slot, averaged over slots.

## `src/gipa_core.py`

**Purpose.** The single shared module. It holds the configuration (`CFG`, `REG`), the controlled environment (`Env`), the congestion priors, the GIPA operator, every baseline policy, the self-consistent prior iteration, and the episode runner and aggregator.

**What the environment is.** A controlled abstract-congestion model, not a network simulator. Local execution costs a class-specific constant. An RSU task costs a class-specific uncongested constant plus 0.55 ms for every other task at the same RSU in the slot, whatever its class. There are no CPU rates, cycles, data sizes or channels. Each vehicle's class is drawn once per episode and repeats every slot.

**Main functions.**

| Name | Manuscript reference | Notes |
| --- | --- | --- |
| `kappa(load)` | Sections IV-B, VI-A | Affine congestion, 0.55 ms per extra co-located task |
| `Env.q`, `Env.lat`, `Env.en` | Section VI-A | Oracle intrinsic scores, realised latency and load, stylized energy |
| `prior_heur(N)` | Eq. (13) | Uniform split over M + 1 actions |
| `prior_workload`, `prior_class` | Simplified Eq. (12) | Used only in the Regime B prior variants |
| `gipa(...)` | Algorithm 1, Eqs. (7), (14)–(20) | Ablation switches for anneal, graded, diffusion, amplification, mask and prior |
| `masked_softmax(...)` | Eqs. (45)–(46) | Also gives independent softmax and entropy-regularised selection |
| `price_rr(...)` | Eq. (47) | Uses fleet-wide expected load (centralised) |
| `greedy`, `alllocal` | Section VI-E | Deterministic baselines |
| `RBR` | Section VI-E | Reacts to last slot's realised load (assumes a load broadcast) |
| `QL` | Section VI-E | Shared tabular learner; full specification in its docstring |
| `Phi`, `sc_prior` | Proposition 3, Appendix G.3 | Count-based induced-congestion map and damped iteration |
| `run`, `agg` | Section VI-F | One episode; 30-seed average plus per-seed records |

**Seeds.** Environment 1000 + *s*, action 5000 + *s*, Q-learning 5999.

## `experiments/exp.py`

**Purpose.** Produces every number in the principal comparison, density sweep and ablation, plus the self-consistent prior audit and the Regime B prior variants.

**What it runs.**

1. For Regimes A and B, it trains one Q-learner at N = 160. Then, for N ∈ {80, 160, 240, 320}, it runs 15 policies (10 policies and 5 GIPA ablations) over 30 matched seeds.
2. The self-consistent prior iteration for every regime and density, with convergence verdicts.
3. Regime B: GIPA with static, self-consistent, self-consistent + 10% and workload-derived priors, plus all-local.

**Outputs.** These go to the **current working directory**, not `results/`.

- `res_main.json`: `{regime: {N: {policy: {metric: mean}}}}`.
- `res_sc.json`: `{"sc": prior audit, "regB": Regime B prior variants}`.

**Manuscript results produced.** Tables IV, V and VI; the values behind Figs. 3–8 (via the statistics file); Section VII-E-2 (Regime B) and VII-E-3 (prior diagnostics); Appendix G.3; Appendix H tables.

**Does not produce.** `results/res_main_seed_level.json` or `results/res_main_statistics.json`; those come from the rebuilt `experiments/seed_level_main.py` and `experiments/stats_main.py`.

## `plotting/generate_figures.py`

**Purpose.** Draws Figs. 3–8 from `results/res_main_statistics.json`. It never runs the simulator.

**Usage.** `python plotting/generate_figures.py [--figure 3..8] [--statistics PATH] [--output DIR]`. Defaults are the archived statistics file and `figures/`.

| Figure | File stem | Content |
| --- | --- | --- |
| 3 | `fig3_matched_policy_comparison_N160` | Regime A, N = 160: miss rate, latency, offload, Jain |
| 4 | `fig4_class_specific_offload_N160` | Safety and background offload, N = 160 |
| 5 | `fig5_gipa_density_all_metrics` | GIPA across N = 80–320 |
| 6 | `fig6_gipa_ablation_all_metrics` | GIPA ablations, N = 160 |
| 7 | `fig7_energy_and_peak_load_N160` | Stylized energy and peak RSU load |
| 8 | `fig8_regimeB_diagnostic_N160` | Regime B, N = 160 |

## `src/planner.py` (new, exploratory)

**Purpose.** Plans the deadline-constrained system-optimal allocation offline (Stage 0); each vehicle then samples from the planned per-class PMF (Stage 1). It is still communication-free: the inputs are the same as the existing prior's (anticipated fleet size, class mix, RSU count, latency constants).

**Why.** GIPA's heuristic prior leads the fleet to roughly the user equilibrium, where every used action costs about the same. That is why GIPA's mean latency equals all-local execution. Planning the system optimum instead reduces total latency.

**Main functions.** `plan_allocation(Nbar, R, z=0.0)` returns per-RSU selection probabilities `x_S`, `x_B` (or `None` if infeasible). `planned_pmf` builds the per-vehicle PMFs. `run_planned` and `agg_planned` mirror `gipa_core.run` and `agg` exactly (same seeds, slots and metrics). The optional margin `z` adds *z* standard deviations of the per-RSU load before the deadline check.

**Limitations.** It assumes identical RSUs and the affine latency model. It optimises expected latency, so realised loads fluctuate under independent sampling. It is infeasible when RSUs cannot absorb every task that local execution cannot serve (Regime B, N ≥ 160).

## `experiments/exp_planner.py` (new, exploratory)

**Purpose.** Compares the planner with GIPA and all-local on the archived seeds, and tests sensitivity to a wrong fleet-size estimate.

**Output.** `results/res_planner.json` (override with `--out`), containing `main`, `misspec` and `per_seed` blocks. Per-seed records are kept so paired statistics can be computed later.

**Headline results (27 Oct 2026 run, Regime A, 30 seeds, no observed misses).**

| N | Mean latency, GIPA → planner (ms) | Peak RSU load, GIPA → planner (tasks) |
| --- | --- | --- |
| 80 | 15.09 → 15.03 | 17.9 → 16.3 |
| 160 | 18.38 → 16.69 | 29.9 → 16.3 |
| 240 | 18.38 → 17.26 | 0.0 → 16.4 |
| 320 | 18.35 → 17.52 | 0.0 → 16.4 |

At N = 160, a plan built for 0.75–1.5 times the true fleet size stays between 16.69 and 16.91 ms; a plan built for half the true size falls back to 18.57 ms. In Regime B the planner cuts the N = 80 safety miss rate from 100% to 4.2% (margin z = 2), but no independent-sampling plan is feasible at N ≥ 160.

## `src/capacity.py` (new, revision experiment E3)

**Purpose.** Answers Reviewer 2's capacity comment by separating what vehicles *selected* from what RSUs *admitted*, and by applying one admission layer to every policy so that results under admission control reflect selection quality alone.

**Capacity.** With the affine model, a class-*k* task at an RSU holding *n* tasks finishes in E_k + 0.55 (*n* − 1) ms. The deadline capacity is Q_k = ⌊(D_k − E_k) / 0.55⌋ + 1, which gives **Q_S = 17** and **Q_B = 59** tasks per RSU per slot in both regimes. Because every co-located task delays every other, an RSU holding any safety task must stay within Q_S (the *effective capacity*).

**Admission rule.** Each RSU considers its selected tasks in random order (seed 9000 + *s*) and admits a task only if every admitted task still meets its deadline afterwards. A rejected task runs locally after a 1 ms handshake penalty. This uses only the vehicle-to-RSU link that offloading needs anyway.

**Main functions.** `capacities(R)`, `admit(...)`, `select(...)` (policy dispatch, including the planner), `run_ac(pol, N, seed, R, ac=..., ql_reset=...)`, `agg_ac(...)`.

**Metrics added.** `maxl_sel`, `maxl_adm` (largest selected and admitted RSU load); `over_sel` (share of RSU-slots whose selected load exceeds the effective capacity); `sev_sel` (mean excess over capacity); `over_adm` (0 by construction under admission control); `rej` (share of offload attempts rejected); `rej_miss` (share of rejected tasks that then missed).

**Verified.** In soft mode, `run_ac` reproduces `gipa_core.run` exactly for all 10 archived policies (1,200 values, both regimes, N = 80 and 160), and the planned policy matches `planner.run_planned`.

## `experiments/exp_capacity.py` (new, revision experiment E3)

**Purpose.** Runs all 10 archived policies plus the planner in soft and admission-control mode, in both regimes and at four densities, over the 30 matched seeds.

**Note on Q-learning.** The learner is trained once per regime at N = 160 as in `exp.py`, but its previous-slot load is reset at the start of every episode, which fixes a known issue. Its numbers can differ slightly from `res_main.json`.

**Output.** `results/res_capacity.json` (override with `--out`), containing `capacities`, `main` and `per_seed` blocks.

**Headline results (27 Oct 2026 run, Regime A, N = 160, 30 seeds; 4,050 RSU-slots per policy).**

| Policy | Selection overload | Peak selected load | Soft: mean latency, misses (S / B) | Admission control: mean latency, rejections |
| --- | --- | --- | --- | --- |
| Greedy | 33.3% | 160.0 | 93.94 ms, 100% / 100% | 18.95 ms, 89.3% |
| Independent softmax | 100% | 59.6 | 35.64 ms, 100% / 18.4% | 18.08 ms, 68.1% |
| Reactive best response | 51.1% | 30.5 | 27.20 ms, 51.1% / 9.3% | 18.23 ms, 34.8% |
| Congestion pricing | 36.2% | 41.9 | 23.12 ms, 6.4% / 0% | 21.01 ms, 16.0% |
| GIPA | 0% | 29.9 | 18.38 ms, 0% / 0% | 18.38 ms, 0% |
| Planner | 0% | 16.3 | 16.69 ms, 0% / 0% | 16.69 ms, 0% |

- GIPA and the planner never exceeded an RSU's effective capacity in any evaluated slot, at any density in Regime A, so admission control never had to act for them. For GIPA this follows from the prior keeping safety tasks local, not from a guarantee.
- Admission control rescues the herding policies' deadlines, but only by rejecting 35–89% of their offload attempts. The rejection rate measures the cost of poor selection.
- Admitted load never exceeded capacity under admission control (2,580 episodes checked).
- Regime B: only the planner with admission control serves most safety tasks (safety miss rate 9.1% at N = 80 and 9.0% at N = 160, against 100% for GIPA and all-local).

## `src/prediction.py` (new, revision experiment E1)

**Purpose.** Answers Reviewer 2's comment on oracle scores by replacing the true intrinsic latencies with predictions and isolating where the error enters.

**How.** The policies are re-implemented with two inputs: `qs` drives the scores and `qm` drives the deadline mask. Otherwise they are line-for-line copies of `gipa_core`. Error arms:
- **none:** oracle;
- **score (a):** prediction in the scores only;
- **mask (b):** prediction in the mask only;
- **both:** one predictor feeds both, as in a real deployment;
- **prior (c):** a wrong anticipated fleet size or congestion slope, applied through `prior_misspecified`.

**Predictors.**
- **Synthetic:** multiplicative error (1 + ε), with ε ~ N(*b*, σ²) clipped at −0.9.
- **EWMA:** each vehicle averages its own observed latency per action (smoothing 0.2); untaken actions go stale.
- **Oracle.**

**Truth models.**
- **Static:** the archived constants.
- **AR(1) drift:** each vehicle's per-action latency follows l₀ · max(1 + η, 0.1), with η an AR(1) process of correlation 0.9 and standard deviation σ_η. This gives the EWMA something to track.

**Guard band.** An action counts as feasible only if (prediction + prior) × (1 + guard) ≤ deadline.

**Verified.** With zero error, `run_pred` reproduces `gipa_core.run` (and the planner reproduces `planner.run_planned`) exactly: 2,400 values, both regimes, N = 80 and 160. Q-learning is excluded because it never uses latency predictions.

## `experiments/exp_prediction.py` (new, revision experiment E1)

**Purpose.** Runs 45 conditions in Regime A over 30 matched seeds, at N = 160 and N = 80: the three error arms, bias, guard band, prior misspecification, and drift with oracle and EWMA predictors. The full design is in the script's docstring.

**Output.** `results/res_prediction.json` (override with `--out`): a `conditions` list with parameters, means and per-seed records for each policy.

**Headline results (27 Oct 2026 run, Regime A, 30 seeds).**

| Condition | GIPA latency, safety misses | Masked softmax latency, safety misses | Planner latency, safety misses |
| --- | --- | --- | --- |
| N = 160, oracle | 18.38 ms, 0% | 22.79 ms, 0% | 16.69 ms, 0% |
| N = 160, combined error σ = 0.30 (24% mean error) | 18.56 ms, 0% | 22.61 ms, 0% | 16.69 ms, 0% |
| N = 80, oracle | 15.09 ms, 0% | 15.72 ms, 0% | 15.03 ms, 0% |
| N = 80, score error only, σ = 0.30 | 15.05 ms, 0% | 15.67 ms, 0% | 15.03 ms, 0% |
| N = 80, mask error only, σ = 0.30 | 15.30 ms, 2.63% | 16.34 ms, 11.98% | 15.03 ms, 0% |
| N = 80, combined σ = 0.30 | 15.26 ms, 2.30% | 16.29 ms, 11.92% | 15.03 ms, 0% |
| N = 80, combined σ = 0.30, guard 0.10 | 15.07 ms, 0.09% | 15.70 ms, 0.48% | 15.03 ms, 0% |
| N = 80, combined σ = 0.30, guard 0.20 | 15.06 ms, 0% | 15.67 ms, 0% | 15.03 ms, 0% |

- **Error in the mask, not the scores, causes misses.** Optimistic errors make an RSU look feasible for safety tasks when it is not. Score error alone caused no misses at any level tested.
- **GIPA is about five times more robust than masked softmax** under the same errors (2.3% against 11.9% safety misses at N = 80, σ = 0.30).
- **A 20% guard band removes every prediction-induced miss** at no latency cost.
- **At N = 160 GIPA is insensitive to intrinsic-prediction error** (no misses at any σ or bias tested, latency within 0.2 ms). This is because the congestion prior dominates the deadline margin; it should be described as robustness through a conservative prior, not as accuracy.
- **The planner is unaffected by per-task prediction error here** because, with identical RSUs, it never uses per-task latency predictions, only class membership and the planning inputs. Its exposure is to prior error (next point).
- **Prior error matters most.** Planning for half the true fleet raises GIPA's latency to 19.40 ms; planning for 1.5× the fleet switches its offloading off entirely. The planner degrades more gracefully (16.69–16.91 ms for 0.75–1.5×; 18.57 ms at 0.5×).
- **Drift misses are not prediction misses.** With drifting latencies, safety misses (2.4%, 15.9% and 25.1% at σ_η = 0.1, 0.2 and 0.3, N = 160) are identical under the oracle and the EWMA predictor. They come from local execution drifting past the 12 ms deadline, so the remedy is offloading safety tasks with reserved capacity, not better prediction.

## `src/calibration.py` (new, revision experiment E2)

**Purpose.** Answers Reviewer 2's latency-model comment. It calibrates the simulator in physical units and implements the paper's Eq. (5) with class-dependent compute and exact processor sharing.

**Calibration.**
- **Assumed anchors:** vehicle CPU f = 1 GHz and a 100 Mbit/s uplink.
- **Task compute:** from the archived local latencies, c_S = 10 Mcycles and c_B = 22 Mcycles (a 2.2× ratio).
- **RSU capacity:** F = 33.45 GHz, chosen so the class-mix average compute time per task equals the archived 0.55 ms. Per task this gives a_S = 0.299 ms and a_B = 0.658 ms of RSU compute.
- **Transfer times:** T_S = 2.701 ms and T_B = 7.342 ms, which keep the uncongested RSU latencies at the archived 3 and 8 ms. At 100 Mbit/s they correspond to about 0.27 and 0.73 Mbit of input data.

**Latency model.** All tasks at an RSU in a slot share the processor equally, so task *i* finishes after T_k + Σ_l min(c_l, c_i)/F. A safety task takes T_S + a_S·*n*; a background task takes T_B + a_S·*n*_S + a_B·*n*_B.

**Priors.**
- `agnostic`: the submitted prior, one value per RSU.
- `class`: a class-aware heuristic built from the same calibration.

The masked-softmax family and GIPA are re-implemented with a per-vehicle prior matrix. Reactive best response and congestion pricing keep their internal affine model. Q-learning is retrained in E4.

**Verified.** With class-independent compute (a_S = a_B = 0.55 ms), Eq. (5) reproduces the affine model of the submitted experiments for all 14 policy variants in both regimes (1,200 values; largest difference 2.8 × 10⁻¹⁴, floating-point rounding). The Eq. (5) planner matches `planner.py`. This is numerical proof that the submitted model is the c_S = c_B special case of Eq. (5).

## `experiments/exp_calibration.py` (new, revision experiment E2)

**Purpose.** Runs three parts:
- **(A)** the special-case check;
- **(B)** the principal comparison, density sweep and ablation under the calibrated Eq. (5) model, in both regimes at four densities, with the submitted and class-aware priors, plus the planner;
- **(C)** an RSU capacity sweep at N = 160, with F = 10, 18.4, 33.45 and 60 GHz.

**Output.** `results/res_calibration.json` (override with `--out`), containing `calibration`, `special_case`, `main`, `per_seed` and `capacity_sweep` blocks.

**Headline results (27 Oct 2026 run, 30 seeds).**

| Regime A, N = 160, Eq. (5) | Mean latency | p95 | Safety / background misses |
| --- | --- | --- | --- |
| All-local | 18.38 ms | 22.0 ms | 0% / 0% |
| GIPA, submitted prior | 19.70 ms | 27.0 ms | 0% / 0% |
| GIPA without diffusion | 18.37 ms | 22.0 ms | 0% / 0% (offloads nothing) |
| Masked softmax | 25.57 ms | 35.3 ms | 0% / 1.5% |
| Congestion pricing | 25.37 ms | 34.8 ms | 6.2% / 2.1% |
| Planner (Eq. 5) | 16.96 ms | 22.0 ms | 0% / 0% |

- **GIPA loses mean-latency parity under Eq. (5).** Its diffusion floor sends about 69% of background tasks to the RSUs, about 26 per RSU, where each now takes about 24.3 ms against 22 ms locally (each extra background task adds 0.66 ms, not 0.55). The ablation confirms the cause: without diffusion GIPA offloads nothing.
- **The planner keeps a 7.7% gain over all-local** (16.96 ms). Planning with the affine model still gives 17.04 ms, so the planner is robust to which latency model it assumes.
- **RSU capacity sweep (Regime A, N = 160).** GIPA offloads nothing at 10 and 18.4 GHz (equal to all-local), is worse than all-local at 33.45 GHz (19.70 ms) and better at 60 GHz (16.35 ms). The planner beats all-local at every capacity: 18.05, 17.65, 16.96 and 15.73 ms.
- **Class-aware prior.** It helps only where safety tasks must offload: in Regime B at N = 80, GIPA's safety miss rate falls from 100% to 2.2%. In Regime A at N = 80 it adds unnecessary safety offloading (0.5% safety misses, +0.9 ms). At N = 160 it changes nothing.
- **Regime B changes under Eq. (5).** Safety tasks add only 0.30 ms each at an RSU, so the planner serves them at N = 160: 0% safety misses with the affine plan (23.01 ms) and 3.7% with the Eq. (5) plan and margin z = 2 (22.28 ms). GIPA still misses 100% at N = 160, and no plan is feasible at N = 320.

## `src/mappo.py` (new, revision experiment E7)

**Purpose.** Answers Reviewer 1's request for a comparison with multi-agent DRL. It implements MAPPO: centralised training, decentralised execution, and one actor network shared by all vehicles.

**Implementation.** Pure NumPy: two hidden layers of 64 tanh units, Adam, PPO clipped surrogate (clip 0.2), entropy bonus 0.01 and gradient-norm clipping 0.5. Gradients are written by hand and checked against finite differences (`gradient_check()`; worst relative error 9 × 10⁻⁷). No PyTorch is needed.

**Inputs.**
- **Actor:** the information GIPA uses (class, intrinsic scores, the prior-inclusive deadline mask and the prior). Invalid actions are masked out of the logits.
- **Critic (training only):** the actor's input plus last slot's RSU loads, the safety fraction and the fleet size.
- **`load` variant:** the actor also sees last slot's loads, which is more information than GIPA has.
- **`--critic local`:** gives the IPPO fallback.

**Reward.** Per vehicle, r = −latency/10 − 3 × miss, in three modes:
- **team:** the fleet mean, shared by every vehicle;
- **individual:** each vehicle's own reward;
- **mixed:** the average of the two.

**Discount.** 0 by default: in this simulator actions do not change the next state, so the optimal policy is myopic. The `load` variant uses 0.9 with GAE 0.95.

**Seeds.** Disjoint from evaluation:
- training environments: seed indices 1000–1049;
- training sampling and initialisation: 8000 + run seed;
- tuning evaluation: indices 2000–2009;
- final evaluation: the 30 archived seeds, with actions sampled from the learned stochastic policy through the archived action generator, so results pair with every other policy.

**Latency models.** `affine` (submitted) or `ps` (Eq. 5 from E2).

**Masks.**
- **`mask='prior'` (default):** GIPA's prior-inclusive deadline mask. At N ≥ 240 it forbids every RSU, so MAPPO can only stay local there.
- **`mask='deadline'`:** checks deadlines against uncongested latency only, which allows every RSU in the archived regimes; MAPPO then learns congestion from the miss penalty. Use it for a fair comparison at high density. The default reproduces the earlier code exactly (identical curves, weights and evaluations).

## `experiments/exp_mappo.py` and `train_mappo.sh` (new, revision experiment E7)

**Stages.**
- **`bench`:** times a short run and estimates the full plan.
- **`tune`:** a 6-configuration grid (team, mixed or individual reward × learning rate 3 × 10⁻⁴ or 10⁻³) at N = 80, scored on the tuning seeds. It picks the lowest latency with at most 0.1% misses.
- **`train`:** trains each seed and density, then evaluates on the 30 archived seeds. N = 160 models are also evaluated at every density. Curves, plateau verdicts and networks are saved.
- **`merge`:** combines the outputs into `results/res_mappo.json`.

**`train_mappo.sh`** runs bench, tune, 5 seeds × 4 densities under both latency models, the `load` variant at N = 160, then merge. Training runs in two parallel single-thread processes. Run it from the repository root inside the virtual environment:

```bash
systemd-inhibit --what=sleep:idle nohup ./train_mappo.sh > train_mappo_log.txt 2>&1 &
```

**Full run (27 Oct 2026, Intel i5-4210M laptop, Ubuntu 24.04).** Total time 2 h 51 min (0.68 s per iteration at N = 160). The log is `results/train_mappo_log.txt` and the merged output `results/res_mappo.json`. Tuning chose team reward with learning rate 10⁻³ (14.92 ms at N = 80; mixed and individual rewards gave 15.57–15.59 ms). Base runs: 5 seeds × 4 densities × 2 latency models, all plateaued, with no deadline misses in any evaluation.

| Mean latency, ms (30 evaluation seeds) | N = 80 | N = 160 | N = 240 | N = 320 |
| --- | --- | --- | --- | --- |
| MAPPO, affine model (range over 5 seeds) | 15.03 (15.02–15.03) | 16.71 (16.70–16.74) | 18.38 | 18.35 |
| Planner, affine model | 15.03 | 16.69 | 17.26 | 17.52 |
| GIPA as submitted, affine model | 15.09 | 18.38 | 18.38 | 18.35 |
| MAPPO, Eq. (5) model | 15.59 | 17.00 | 18.38 | 18.35 |
| Planner, Eq. (5) model | 15.56 | 16.96 | 17.43 | 17.65 |
| GIPA as submitted, Eq. (5) model | 15.94 | 19.70 | 18.38 | 18.35 |
| All-local | 18.41 | 18.38 | 18.38 | 18.35 |

- **MAPPO with a shared fleet reward matches the planner and beats GIPA** (16.71 against 18.38 ms at N = 160). Its peak RSU load is 15.5 tasks against GIPA's 29.9.
- **N ≥ 240 is decided by the mask, not by learning.** MAPPO is given GIPA's prior-inclusive deadline mask, which forbids every RSU there (predicted 8 + 32.45 ms > 40 ms for background tasks), so it can only stay local. A fair test at those densities needs MAPPO trained with a deadline-only mask.
- **No generalisation across density.** Models trained at N = 160 give 15.94 ms at N = 80 (affine), against 15.03 ms when trained at N = 80, so MAPPO must be retrained per density. The planner just recomputes its allocation.
- **`load` variant** (actor also sees last slot's loads; discount 0.9): worse and less stable, 18.07 ms on average (17.27–18.56) with 2 of 5 runs not plateaued. More information did not help within 400 iterations.

**Deadline-mask follow-up and cross-machine check.** `train_mappo_highdensity.sh` trains N = 240 and 320 with `--mask deadline` under both latency models (5 seeds each). It then merges and runs `experiments/compare_mappo.py` against `results/res_mappo_dmask_reference.json`, the same experiment run independently on the reference machine. Results are not bit-identical across machines, because BLAS rounding differs between CPUs and propagates through training, so agreement is judged within 0.05 ms on each group's mean latency.

**Reference-machine run (27 Oct 2026, 45 min; log in `results/train_mappo_highdensity_log.txt`).** All 20 runs plateaued.

| Mean latency, ms (range over 5 seeds) | N = 240 | N = 320 |
| --- | --- | --- |
| MAPPO, deadline-only mask, affine model | 17.27 (17.27–17.29) | 17.53 (17.52–17.53) |
| Planner, affine model | 17.26 | 17.52 |
| MAPPO, deadline-only mask, Eq. (5) model | 17.45 (17.44–17.46) | 17.66 (17.65–17.66) |
| Planner, Eq. (5) model | 17.43 | 17.65 |
| GIPA as submitted, and MAPPO with GIPA's mask | 18.38 | 18.35 |

With a fair mask, MAPPO matches the planner at every density and under both latency models, to within 0.02 ms.

**Cross-machine verification (28 Oct 2026).** The same high-density experiment, run independently on the i5-4210M laptop with `train_mappo_highdensity.sh`, agrees with the reference-machine run in every group. Group means differ by at most 0.010 ms and single runs by at most 0.031 ms (tolerance 0.05 ms). Laptop means: affine model 17.28 ms at N = 240 and 17.53 ms at N = 320; Eq. (5) model 17.46 and 17.66 ms. `results/res_mappo.json` is now the laptop's complete merged file: the earlier groups are unchanged, and the four deadline-mask groups have been added. Safety misses are at most 0.01%. Under Eq. (5), MAPPO, like the planner, also offloads some safety tasks (7–15%).

## `src/qlearning.py` (new, revision experiment E4)

**Purpose.** Answers Reviewer 2's comment that Q-learning was trained only once. `QLearner` generalises the archived `gipa_core.QL`, which is left untouched.

**Fixes, each controlled by an argument.**
- **Disjoint training environments:** seed indices 3000–3006 instead of 0–6, which overlapped evaluation.
- **Episode reset:** last slot's load is cleared at the start of every evaluation episode.
- **Multiple training runs:** exploration seeds 6000–6009.
- **Optional discount:** γ = 0.9 with a bootstrapped next-state target.
- **Latency model:** `affine` or `ps` (Eq. 5).

**Verified.** In the archived configuration (seed 5999, environments 0–6, γ = 0, no reset) it reproduces `gipa_core.QL` exactly: identical Q-table and identical metrics on all 30 seeds (`check_archived()`).

**Two-level bootstrap.** `two_level_bootstrap()` resamples training runs, then evaluation seeds within each run (10,000 replicates, seed 20260804), so the interval reflects both sources of variability.

## `experiments/exp_qlearning.py` (new, revision experiment E4)

**Purpose.** Runs two parts:
- **(1)** isolates the fixes on the archived seed;
- **(2)** for both regimes and both latency models, trains 10 runs at N = 160 with γ = 0, 10 with γ = 0.9, and 10 per density at N = 80, 240 and 320. Every run is evaluated on the 30 archived seeds.

**Output.** `results/res_qlearning.json`: `fixes`, `grid` (mean, two-level 95% CI and between-run SD per metric), `runs` (Q-table, greedy policy, learning curve and evaluations per run) and `archived_reference`. The log is `results/exp_qlearning_log.txt`.

**Results (28 Oct 2026).**
- **The fixes alone change nothing for the archived seed.** Seed 5999 still gives 18.2003 ms with disjoint environments and the reset; the archived value reproduces exactly.
- **The single seed hid a bimodal learner.** Regime A, affine model, γ = 0, trained and evaluated at N = 160, 10 runs:
    - 7 runs learn all-local (18.20 ms);
    - 2 runs send every safety task to one RSU whenever the load bucket is low (21.07 ms, 49% safety misses);
    - 1 run does the same with background tasks (33.76 ms, 48% background misses).
  The mean is 20.33 ms (two-level 95% CI 18.23–23.51) with a between-run SD of 4.87 ms. The archived seed happened to land in the all-local group.
- **Why.** All vehicles of a class share one table and act greedily, so the learner cannot represent a mixed strategy. It either stays local or herds a whole class onto one RSU, which flips the load bucket and alternates slot by slot. Q-values never settle: the largest change over the final 500 steps is still about 4 in every configuration (median 4.21).
- **γ = 0.9** is slightly more reliable (8 of 10 runs all-local; 18.78 ms, CI 18.19–19.63) but has the same failure mode.
- **Per-density training** makes it worse at high density. At N = 320: 33.28 ms (CI 18.18–55.74), with 4.9% safety and 14.7% background misses.
- **Under Eq. (5)** it is similar: 21.58 ms at N = 160 (CI 18.19–26.37), with 2 of 10 runs herding.
- **Regime B:** about 98% safety misses in every configuration, because it never learns to offload safety tasks.
- **Implication for the paper.** Report Q-learning as the mean of 10 runs with the two-level CI and the run breakdown, and say why it fails. Its archived 18.20 ms was a lucky all-local outcome, not a tuned result.

## `src/learning_automata.py` (new, revision experiment E6)

**Purpose.** Answers Reviewer 1's request for a direct comparison with learning-automata offloading. Each vehicle keeps its own probability vector over its feasible actions, samples from it every slot with the archived action generator (so results are paired), and updates it from its own realised latency only. There is no communication, no prior and no load broadcast.

**Feasible set.** Deadline-only by default (uncongested cost within the deadline), as in MAPPO's fair comparison. `mask='prior'` uses GIPA's prior-inclusive mask, which forbids every RSU at N ≥ 240 in Regime A.

**Environment response.**
- `binary` (P-model, the workplan specification): β = 1 if the deadline is met and the latency is no worse than local.
- `snorm` (S-model): β = 1 − latency/deadline if the deadline is met, else 0.

**Update schemes** (rate *a*, chosen action *k*):
- `lri` (L_R-I): p ← p + a·β·(e_k − p); no change on penalty.
- `lrep` (L_R-εP, ε = 0.1): also, on penalty, p_k shrinks by εa and the other feasible actions move toward uniform.
- `pursuit`: per-action reward estimates (step max(1/count, 0.05), optimistic start 2), p ← (1 − a)·p + a·e_best. Ties are broken by a fixed per-vehicle jitter (seed 9800 + *s*).

**Runners.** `run_la` runs `warm` learning slots then the 45 evaluated slots (`warm = 25` is the archived 70-slot episode). `run_schedule` runs a fleet-size schedule for LA, GIPA (prior refreshed or stale), the planner (re-planned or stale) and all-local on the same seeds.

**Verified.** `check_updates()` reproduces hand-computed L_R-I and L_R-εP updates exactly.

## `experiments/exp_la.py` (new, revision experiment E6)

**Purpose.** Four parts:
- **(1)** self-check;
- **(2)** tuning grid on seed indices 4500–4509: 3 algorithms × 2 responses × rates {0.01, 0.05, 0.10}, for both regimes, both latency models, warm-up 25 and 500, at all four densities. Rule at N = 160: keep settings within 0.5 points of the lowest overall miss rate, then take the lowest latency;
- **(3)** evaluation on the 30 archived seeds, against GIPA, the planner (z = 2) and all-local computed on the same seeds, with paired bootstrap CIs (10,000 resamples, seed 20260804), learning curves and one row with GIPA's mask;
- **(4)** density-change test (Regime A, affine): N = 80 → 240 at slot 500 → 80 at slot 1000, with per-phase transient, steady state and recovery time.

**Output.** `results/res_la.json`: `check`, `tuning`, `chosen`, `main`, `prior_mask`, `per_seed`, `density_change` (schedule, mean per-slot traces and phase summaries). The log is `results/exp_la_log.txt`.

**Results (28 Oct 2026, Regime A).**

| Mean latency, ms (overall miss rate) | N = 80 | 160 | 240 | 320 |
| --- | --- | --- | --- | --- |
| L_R-I, binary, a = 0.1, warm-up 500 | 15.03 | 16.74 | 17.30 | 17.54 |
| L_R-εP, binary, a = 0.1, warm-up 500 | 15.03 | 16.73 | 17.28 | 17.53 |
| Pursuit, S-model, a = 0.1, warm-up 500 | 15.46 | 17.17 | 17.56 | 17.94 |
| L_R-I, binary, a = 0.1, warm-up 25 | 15.26 (0.8%) | 16.97 (1.1%) | 17.52 (1.2%) | 17.78 (1.3%) |
| Planner (z = 2) | 15.03 | 16.69 | 17.26 | 17.52 |
| GIPA as submitted | 15.09 | 18.38 | 18.38 | 18.35 |
| All-local | 18.41 | 18.38 | 18.38 | 18.35 |

Misses are zero unless shown.

- **LA matches the planner and MAPPO.** After warm-up, L_R-I and L_R-εP are within 0.01–0.04 ms of the planner at every density. LA beats GIPA by 1.6 ms at N = 160 (paired 95% CI 1.5–1.8 ms) and by 0.8–1.1 ms at N ≥ 240.
- **Convergence is fast.** The mean latency settles within about 75–100 slots (1.5–2 s at 20 ms per slot); learning curves are in `main[...]["best"]["curve"]` (25-slot blocks).
- **The price is the learning transient.** With only the archived 25-slot warm-up, L_R-I already gets 16.97 ms at N = 160, but with about 1% of all tasks missing their deadline, all of them safety tasks (2.6–4.3% of safety tasks).
- **Under Eq. (5),** the best LA (L_R-I, a = 0.05, warm-up 500) gets 15.92, 17.06, 17.49 and 17.67 ms, against 15.56, 16.96, 17.43 and 17.65 ms for the planner, and 16.83, 19.70, 18.38 and 18.35 ms for GIPA.
- **The S-model response is worse** for L_R-I and L_R-εP. It rewards any timely offload, so vehicles over-offload (on the tuning seeds at N = 160, 21.5–22.2 ms with 56–61% safety misses at a = 0.01).
- **GIPA's mask** limits LA as it limits MAPPO. With it, LA reaches 15.02 ms at N = 80 and 17.04 ms at N = 160, but offloads nothing at N ≥ 240.

**Regime B.** Under the affine model no decentralised method protects safety tasks: LA misses 23–100% of them (GIPA misses 100%), because background vehicles fill the RSUs up to their own break-even load (41 tasks). Under Eq. (5), pursuit LA with warm-up 500 has 0% safety misses at N = 80–240 (22.45 ms at N = 160, against 22.28 ms for the planner and 23.43 ms with 100% safety misses for GIPA). At N = 320 it misses 49% of safety tasks.

**Density-change test (Regime A, affine; mean over 30 seeds).**

| Policy | N = 240 phase: first 50 slots | N = 240 phase: last 100 slots | Recovery (slots) |
| --- | --- | --- | --- |
| L_R-I | 20.72 ms, 6.7% misses | 17.58 ms, 0% | 41 |
| L_R-εP | 20.54 ms, 6.6% | 17.44 ms, 0% | 41 |
| Pursuit | 20.62 ms, 8.0% | 17.22 ms, 0% | 21 |
| GIPA, prior refreshed to N | 18.38 ms, 0% (all-local) | 18.38 ms, 0% | 0 |
| GIPA, stale N = 80 prior | 23.76 ms, 0.3% | 23.81 ms, 0.4% | 0 |
| Planner, re-planned | 17.26 ms, 0% | 17.26 ms, 0% | 0 |
| Planner, stale N = 80 plan | 22.15 ms, 0.1% | 22.19 ms, 0% | 0 |

On the return to N = 80, every LA scheme is back at its steady state immediately (15.0 ms). LA needs no fleet-size estimate but pays a transient of about 40 slots (0.8 s) with misses after each jump. GIPA and the planner react instantly only if their prior or plan is refreshed with the new N; with a stale prior they are worse than all-local.

**Caveat.** The simulator keeps the same vehicles, classes and costs for the whole episode, a repeated game that favours a learner. With vehicle turnover, LA's learning cost recurs; the density-change test is a first measure of this.

**Implication for the paper.** A communication-free, prior-free learner matches the system-optimal planner after about two seconds of operation and beats GIPA as submitted. The comparison must be reported, and it weakens any claim that GIPA's advantage comes from being communication-free. GIPA's remaining argument is no learning transient, which holds only with a correctly refreshed prior. This bears on decision D5.

## `experiments/exp_complexity.py` (new, revision experiment E8)

**Purpose.** Decision complexity and end-to-end decision cost: asymptotic complexity, measured decision time, Stage 0 / planner and predictor cost, training cost and runtime communication.

**Method.** Regime A, affine model.
- **Per-slot fleet decision time** for every policy at N = 80–320, as implemented (vectorised NumPy), repeated until at least 10⁵ vehicle decisions are timed (median and p95 per slot, and median ÷ N).
- **Single-vehicle decision time** over 10⁵ calls, half safety and half background vehicles. GIPA is timed both as the NumPy operator with N = 1 and as a pure-Python scalar version (`gipa_scalar`), which is checked to choose the same action as `gipa_core.gipa` for the same uniform draw.
- **Slow-time-scale costs:** heuristic prior, self-consistent prior, planner grid search (affine and Eq. 5) and the E1 EWMA predictor update.
- **Training cost:** one 15,000-step Q-learning run, MAPPO seconds per iteration, and LA warm-up.
- **Communication:** application payload only, analytic, under the manuscript's assumptions (float32 values, 20 ms slots, refresh every 20 slots, M = 3), which give its 90 B/s for GIPA.

The complete computing environment is recorded in the output (`hardware` → `full`, from `src/sysinfo.py`). Timings depend on hardware and load, so `run_complexity_repeats.sh` runs the experiment five times on an idle machine, single-threaded (`OPENBLAS_NUM_THREADS=1`), and `experiments/aggregate_complexity.py` combines the runs: every timing is the median across runs, with the run-to-run range stored under `range`.

**Output.** `results/res_complexity.json` (combined; same structure as a single run plus `repeats`, `range`, `runs`), with the five single runs and their logs in `results/complexity_runs/`.

**Reference machine (29 Oct 2026; these are the numbers in the manuscript).**

| Item | Value |
| --- | --- |
| Processor | Intel Xeon Processor @ 2.10 GHz (model name as exposed by the hypervisor), GenuineIntel, x86-64; SSE4.2, AVX, AVX2, FMA, AVX-512F |
| Cores | 2 vCPU (2 cores, 1 thread per core, 1 socket); L2 2 MiB per core; L3 260 MiB (shared, as reported) |
| Virtualisation | KVM virtual machine, full virtualisation |
| Memory | 7.8 GiB |
| Operating system | Ubuntu 24.04.5 LTS, Linux kernel 6.18.44, glibc 2.39 |
| Python | CPython 3.13.17 (built with Clang 23.1.3) |
| Libraries | NumPy 2.5.3, SciPy 1.18.1, Matplotlib 3.11.2; BLAS: OpenBLAS 0.3.34 (scipy-openblas, DYNAMIC_ARCH) |
| Threads | One thread (OPENBLAS_NUM_THREADS = OMP_NUM_THREADS = MKL_NUM_THREADS = 1) |
| Protocol | 5 repeated runs; each timing is the median over at least 10⁵ decisions; reported value is the median across runs |

**Results (median across five runs; run-to-run range in brackets).**

| Policy | Per-vehicle decision, μs | Fleet per slot at N = 160, μs | Runtime payload, B/s | Training |
| --- | --- | --- | --- | --- |
| GIPA | 29.5 (scalar Python), 86.6 (NumPy, N = 1) | 187 [181–191] | 90 (optional refresh) | None |
| Planner | 4.6 (one draw) | 11.5 [11.1–12.2] | 60 (plan refresh); grid search 23–29 ms per refresh at N = 160 | None |
| Masked softmax (prior) | 16.5 | about 47 | 90 | None |
| Learning automata (L_R-I) | 14.8 (draw and update) | 28.5 [26.8–29.0] | 0 | Online, about 75–100 slots |
| Q-learning | 9.1 | 14.4 [13.8–15.3] | 200 (load statistic) | 2.8 s per run |
| MAPPO actor (5,380 parameters) | 30.5 | 171 [157–220] | 90 (base) or 600 (load variant) | 0.24 s per iteration, 400 iterations per run (0.68 s per iteration on the i5-4210M laptop) |
| RBR | — | 39.0 [37.5–39.2] | 600 | None |
| Pricing (30 iterations) | — (centralised) | 1,293 [1,277–1,532] | about 3.9 MB/s if distributed | None |

- **Every decision is far below the 20 ms slot.** The slowest per-slot computation is centralised pricing, about 1.3 ms at N = 160.
- **Slow-time-scale costs are small.** The heuristic prior takes about 1 μs and the planner grid search 23–29 ms per refresh; refreshed every 20 slots, the planner amortises to about 1.2–1.5 ms per slot at the RSU.
- **Earlier single run (28 Oct, another instance of the same machine type, Xeon @ 2.30 GHz):** about 13–30% faster throughout (for example 10 μs for the planner and 151 μs for GIPA per slot at N = 160). It is superseded by the five-run figures above, which state their spread.

## `src/asymmetric.py` (new, revision experiment E9)

**Purpose.** Stress-tests the core anti-herding claim. With identical RSUs there is no obvious best RSU to herd onto; here RSU 1 is made 30% faster.

**Model.** RSU *j* has a speed factor sp_j that scales its uncongested latency and its congestion. Scenario `fast1` is sp = (0.7, 1, 1); scenario `sym` is (1, 1, 1).
- `AsymEnv` applies this to the affine model.
- `AsymPSEnv` applies it to the Eq. (5) model.
- Intrinsic scores rank the fast RSU first.

**Priors.**
- `uniform` is the archived Eq. (13). It gives every RSU the same delay and ignores speed.
- `capacity` is a recalibrated Eq. (12). Load is split in proportion to capacity 1/sp_j, and each RSU's delay is computed at its own speed.

**Planner.** `plan_asym` finds the system-optimal per-RSU expected counts (s_j, b_j) with SLSQP from several starts, under the same deadline constraints (margin z = 2) as `planner.py`. With sp = (1, 1, 1) it reproduces the grid planner's objective; it is lower in one case, Regime B at N = 80 (3.6%), because it may split unevenly.

**Herding metrics** (per slot, averaged):
- `herd_fast`: fast-RSU load divided by its capacity-weighted fair share of the offloaded load. 1 is fair; 2.40 means every offload is on RSU 1.
- `cv_raw`, `cv_cap`: coefficient of variation of the RSU loads, raw and capacity-normalised.
- `same_choice`: probability that two vehicles of the same class choose the same action.
- `swing`: slot-to-slot load swing ÷ N; 2 is the maximum, with all offloaders moving every slot.
- `overload`: fraction of slots in which some RSU has a deadline miss.
- `load1`–`load3`: load per RSU.

**Verified.**
- With sp = (1, 1, 1) and the uniform prior, every policy reproduces the archived (affine) and E2 (Eq. 5) latencies exactly, in both regimes and at all four densities.
- `learning_automata.feasible_set` now reads intrinsic costs from `env.q` so that it handles per-RSU speeds. E6 results are unchanged; spot checks are bit-identical.

## `experiments/exp_asym.py` (new, revision experiment E9)

**Purpose.**
- **Scope:** scenarios `sym` and `fast1` × regimes A and B × latency models affine and Eq. (5) × N = 80–320, on the 30 archived seeds.
- **Policies:** local, greedy, independent softmax, RBR, the planner, and learning automata (the E6 'best' configuration with 500 warm-up slots). Masked softmax, prior-adjusted masked softmax, entropy-regularised softmax, GIPA and GIPA without diffusion each run with both priors.
- **Paired comparisons** (seed-level bootstrap): GIPA − planner, GIPA − GIPA without diffusion (latency and herding ratio), and LA − planner.

**Output.** `results/res_asym.json`: `checks`, `scenarios`, `plans`, `main`, `paired`, `per_seed`. The log is `results/exp_asym_log.txt`. `verify_e9.sh` reruns it on another machine and compares with `experiments/compare_results.py`.

**Results (28 Oct 2026, Regime A, RSU 1 30% faster).**

| Affine model | N = 80 | 160 | 240 | 320 |
| --- | --- | --- | --- | --- |
| Planner (z = 2) | 13.94 (herd 1.08) | 16.15 (1.09) | 16.90 (1.08) | 17.25 (1.08) |
| Learning automata | 14.25 (0.93) | 16.35 (1.21) | 17.00 (1.17) | 17.29 |
| GIPA, archived prior | 14.02 (0.96) | 17.30 (0.81) | **22.23 (2.40)**, 1% background misses | 18.35 (all-local) |
| GIPA, capacity prior | 14.91 (1.20), **17% safety misses** | 17.30 (0.81) | 20.40 (0.80) | 18.35 (all-local) |
| GIPA without diffusion, capacity prior | 21.90 (2.39) | 18.37 (≈ all-local) | 18.38 (all-local) | 18.35 |
| Prior-adjusted masked softmax, capacity prior | 22.01 (2.38) | 18.35 | 18.38 | 18.35 |
| Greedy | 34.97 (2.40) | 65.76 | 96.56 | 127.35 |
| RBR | 31.51 (1.17, swing 2.0) | 57.81 | 85.61 | 117.36 |
| All-local | 18.41 | 18.38 | 18.38 | 18.35 |

Mean latency in ms, with the fast-RSU herding ratio in brackets.

- **Diffusion is a real anti-herding mechanism.** Every non-diffused probability policy sends 99% of its offloads to the fast RSU: same-choice probability about 1 and herding ratio 2.4. GIPA keeps a herding ratio of 0.8–1.2 and a same-choice probability of 0.3–0.6. At N = 80, GIPA beats GIPA without diffusion by 7.8 ms (paired 95% CI 7.4–8.2, archived prior). Under Eq. (5) the gap is 14.6 ms (15.16 against 29.75 ms with the capacity prior; the no-diffusion variant also misses 90% of safety tasks).
- **GIPA does not allocate by capacity.** From N = 160 its diffusion splits the offloaded load equally (herding ratio 0.80), so the fast RSU is under-used. It is 1.15 ms worse than the planner at N = 160 (CI 1.07–1.23) and worse than all-local at N = 240.
- **The archived prior causes herding at N = 240.** The prior-inclusive mask leaves only the fast RSU feasible for background tasks, so every offload goes there: herding ratio 2.40, 22.23 ms, and 1% background misses (22% under Eq. 5). The recalibrated capacity prior removes this herding, but GIPA is then still worse than all-local (20.40 ms). At N = 80 the capacity prior has the reverse effect: only the fast RSU is feasible for safety tasks, so they herd onto it (17% safety misses). In both cases the herding is caused by the mask and the prior, not by the amplification.
- **The planner handles asymmetry.** It loads the fast RSU in proportion to its advantage (herding ratio about 1.08), has zero misses in Regime A, and is best at every density under both models (Eq. 5: 14.66, 16.51, 17.14 and 17.43 ms). Learning automata come within 0.1–0.3 ms of it.
- **Herding baselines.** Greedy and independent softmax herd onto the fast RSU (82–100% safety misses). RBR oscillates at the maximum swing (2.0), with the whole offloaded fleet switching RSU every slot.

**Regime B.**
- Under the affine model, safety tasks remain unprotected by GIPA (100% misses), as in the symmetric case. The planner has 5.6% safety misses at N = 80 and is infeasible above that. Learning automata miss 46–100% of safety tasks.
- Under Eq. (5), GIPA has 0% safety misses at N = 80 (16.65 ms) but 100% from N = 160. The planner misses 4.1–4.3% and learning automata 0–1.8% at N = 160–240.

**Implication for the paper (D5).**
- **Supported:** the anti-herding claim, in the sense that GIPA's diffusion prevents the collapse onto the best RSU that every non-diffused probability policy shows. The ablation isolates diffusion as the cause.
- **Not supported:** the claim that GIPA allocates well. It equalises load rather than matching capacity, and the prior-inclusive mask can itself create herding.
- **Strongest method:** the planned allocation, which is communication-free in operation and capacity-aware.

## `src/hetero.py` and `experiments/exp_hetero.py` (new, revision experiment E12)

**Purpose.** Tests decision D5. Can GIPA, run on top of a planned allocation and using each task's own information (Framing A), beat the planner (Framing B)? In the homogeneous simulator it cannot, by construction, so E12 adds the conditions where per-task information matters.

**Model.** E2's physical calibration in Regime A, with three sources of variation:
- every slot, each vehicle has a new task whose compute and data size are mean-one lognormal around the class means (σ_h);
- each vehicle–RSU link rate is a lognormal factor (σ_r, fixed per episode);
- optionally, each vehicle reaches only 2 of the 3 RSUs.

RSUs use batch processor sharing with per-task work. With σ = 0 and full coverage this is exactly E2's environment: the planner and GIPA reproduce E2's per-seed latencies (difference 0). Policies may see predicted features (lognormal error σ_p).

**Policies.**
- **Framing B:**
    - `plan`: class-level sampling from the E2 planner.
    - `plan_mask`: the plan restricted to each task's guarded deadline mask.
    - `plan_rank`: the plan's class offload fraction filled by the tasks with the largest predicted benefit. The threshold is set in Stage 0 by Monte Carlo. Local-infeasible tasks are always offloaded, and RSUs are chosen at random among reachable ones.
- **Framing A:**
    - `gipa_plan`: start from the plan, mask with a 10% guard, K = 4 amplification rounds on the task's score advantage over the class-average task, then diffusion toward the plan (λ). λ and β are tuned per scenario and density on seeds 4600–4609.
    - `gipa_plan_abs`: the same operator amplifying absolute scores.
- **Others:** GIPA as submitted, learning automata (500 warm-up slots) and all-local.

**Results (29 Oct 2026, 30 seeds; mean latency in ms with overall misses).**

| het50: task σ 0.5, link σ 0.3 | N = 80 | 160 | 240 | 320 |
| --- | --- | --- | --- | --- |
| All-local | 18.40 (13.4%) | 18.37 (13.3%) | 18.39 (13.3%) | 18.36 (13.3%) |
| GIPA as submitted | 14.67 (2.7%) | 17.25 (8.3%) | 16.73 (10.1%) | 18.36 (13.3%) |
| Planned allocation, class sampling | 15.09 (8.8%) | 16.74 (11.2%) | 17.28 (11.8%) | 17.54 (12.2%) |
| Plan restricted to feasible set | 14.16 (1.3%) | 15.72 (3.6%) | 16.60 (6.5%) | 17.25 (9.1%) |
| **Plan + per-task ranking** | 12.78 (1.4%) | **14.59 (2.2%)** | **15.54 (3.1%)** | **16.11 (4.2%)** |
| **GIPA on the plan (tuned)** | **12.02 (0.6%)** | 14.98 (2.7%) | 16.05 (5.5%) | 16.83 (8.3%) |
| Learning automata | 14.69 (4.5%) | 16.48 (7.8%) | 17.17 (9.3%) | 17.50 (10.1%) |

- **With identical tasks, per-task logic adds nothing.** In `homog` and `cov2` every plan-based policy coincides (differences ≤ 0.01 ms). Partial coverage alone changes nothing once the plan's mass for an unreachable RSU is moved to the reachable ones.
- **With heterogeneous tasks, per-task information matters a great deal.** Class-level sampling misses 9–12% of deadlines, because large safety tasks cannot run locally in 12 ms. Using each task's own information cuts this to 1–4% and lowers latency by 1.3–2.3 ms.
- **Framing A wins at light load and loses at heavier load.**
    - At N = 80, GIPA on the plan beats the ranking planner by 0.45–0.78 ms (paired 95% CI excludes 0 in every heterogeneous scenario), with fewer misses.
    - From N = 160 it is 0.36–0.79 ms worse, and its misses grow faster (8.3% against 4.2% at N = 320).
- **Why.** Amplification is worth 0.3–2.1 ms over the masked plan. But its mask and amplification push the offload ratio above the planned fraction (0.33 against 0.23 at N = 160), and the tasks it offloads are less well chosen: their benefit advantage over the average task is 7.7 ms, against 12.5 ms for the tasks chosen by ranking. When capacity is slack, extra offloading helps. When the plan's fraction binds, ranking (offload exactly the planned number, and the best tasks) wins.
- **Tuning.** The tuned diffusion was always λ = 0: diffusion toward the plan did not help, because nothing here invites herding (the RSUs are symmetric). Strong amplification (β = 4) was chosen at N = 80 and the default β = 0.4 above that.
- **Other policies.** Prediction error (σ_p = 0.2) costs the per-task methods at most 0.16 ms and 1.8 points of misses, with the same ordering. Learning automata cannot learn task-to-task variation and stay at 7.8–10% misses.

**Implication (D5).** E12 supports Framing B: per-task information should enter through the plan, by ranking tasks within the planned offload fraction, rather than through GIPA's amplification. Framing A is worthwhile only at light load. The natural refinement, amplification constrained to the planned fraction, is future work.

## `src/stats.py` and `experiments/stats_main.py` (rebuilt)

**Purpose.** These rebuild the missing statistics script behind `results/res_main_statistics.json`, which supplies every confidence interval, paired comparison, TOST and zero-miss bound in Sections VI-G and VII and Figs. 3–8. `src/stats.py` holds reusable functions for the revision's own tables.

**Conventions recovered.** Reproducing the archive exactly pins these down:
- **Bootstrap:** a percentile bootstrap over the 30 seeds, 10,000 resamples, interval from `np.percentile` at 2.5 and 97.5.
- **One shared stream:** a single generator, `default_rng(20260804)`, serves every interval in a fixed order: summaries (A then B, N ascending, policies in file order, metrics mS, mB, lat, p95, fair, off, offS, offB, en, maxl), then the nine paired comparisons at Regime A, N = 160 (metrics lat, p95, mS, mB, off, offS, offB, fair, en, maxl).
- **Undefined values:** NaN values (undefined Jain fairness) are dropped first.
- **Constant values:** when every value is equal, the interval is [mean, mean] and no random numbers are drawn.
- **TOST:** a paired t-test against a 1 ms margin, 90% CI.
- **Zero-miss bounds:** a one-sided 95% Clopper–Pearson bound, Beta(0.95; x + 1, n − x).

**Verified (28 Oct 2026).** All 1,200 summary intervals and all 90 paired intervals reproduce the archive, as do the TOST (90% CI −0.0940 to 0.0988 ms, p < 3 × 10⁻¹⁷) and every Clopper–Pearson bound. Here (NumPy 2.5.3, SciPy 1.18.1), 32 values differ in the last printed digit (at most 4 × 10⁻¹⁵ relative). Under the pinned NumPy 2.4.4 and SciPy 1.17.1 the file is expected to be byte-identical; `verify_stats.sh` checks this on the laptop.

**Run.** `PYTHONPATH=src python experiments/stats_main.py --check`.

## `experiments/seed_level_main.py` (rebuilt)

**Purpose.** Rebuilds the missing instrumented run behind `results/res_main_seed_level.json`. It repeats `experiments/exp.py` exactly (same policy order; the same Q-learner trained once per regime at N = 160 and carried across densities and seeds; the same random streams) and only adds counts. Per seed it records:
- the ten metrics;
- `nS` and `nB`;
- miss counts (`missS_count`, `missB_count`) and offload counts (`offS_count`, `offB_count`);
- `evaluated_slots` and `fair_defined_slots`.

**Verified (28 Oct 2026).** The output is byte-identical to the archived file (MD5 `8727bda798913aae39beeda4efc1f707`, 36,000 numbers). Feeding it to `experiments/stats_main.py` reproduces the archived statistics. The chain from simulator to seed-level records to statistics to Figs. 3–8 is now fully scripted.

## `src/async_sim.py` (new, revision experiment E5)

**Purpose.** Answers Reviewer 1's point that the system looks synchronous, and the request to carry unfinished work forward. It is an event-driven simulator in which every task is decided at its own arrival time and RSU queues persist for the whole run.

**Model.**
- **Arrivals:** each of N vehicles generates tasks as a Poisson process with a mean of one task per 20 ms, so the fleet produces N tasks per 20 ms on average. Classes are drawn as in `gipa_core.Env`.
- **Local execution:** the archived local latency, unchanged from the synchronous model.
- **Offloading:** the task spends the E2 transfer time T_k (2.70 or 7.34 ms), then joins RSU *j*'s queue. Each RSU is an egalitarian processor-sharing server at the E2 capacity: a task needs 0.299 ms (safety) or 0.658 ms (background) of service when alone. Queues never reset, so unfinished work carries forward.
- **Simulation:** exact, with a virtual-time engine (`PSQueue`). Queues are drained after the 20 s horizon.
- **Warm-up:** chosen by MSER-5 on the total backlog, at most half the run.
- **Admission control** (`ac=True`): the asynchronous rule of E3. On arrival, a task is admitted only if T_k + a_k(n + 1) ≤ D_k; otherwise it runs locally after a 1 ms handshake.

**Policies and the information they use.**
- **Static policies** (no runtime information; one PMF per class):
    - local, greedy and independent softmax;
    - GIPA and prior-adjusted masked softmax, each with three priors:
        - `slot`: the E2 synchronous prior;
        - `soj_heur`: a backlog-aware prior a_k ρ/(1 − ρ), with ρ from Stage 0's uniform split;
        - `soj_plan`: the same with ρ from the plan;
    - GIPA without diffusion;
    - the asynchronous planner `plan_async`, a grid search over per-class offload probabilities using the M/G/1-PS mean sojourn T_k + a_k/(1 − ρ), with ρ ≤ 0.9 and a 3× sojourn margin on deadlines.
- **Broadcast policies** (RSU backlog sent to every vehicle every 20 or 100 ms):
    - RBR, and GIPA with prior a_k n_j;
    - `jsq_live`, shortest expected delay with the backlog refreshed every 1 ms, as a centralised-information reference.
- **Feedback:** learning automata (E6 tuned L_R-I), updated when each vehicle's task completes, in 20 ms chunks.

**Seeds.** Classes 1000 + *s*, arrivals 11000 + *s*, decisions 5000 + *s* (one uniform per task, shared by all policies, so runs are paired).

**Verified.**
- **Table III rebuilt** (`check_table3`): 21.64, 45.92, 95.48 and 195.25 ms, against the manuscript's 21.64, 45.90, 95.48 and 195.25 ms. The original script was missing; this replaces it. The 0.02 ms difference at 10 tasks is Monte Carlo variation.
- **Simultaneous arrivals** reproduce the E2 batch formula Σ min(w_l, w_i) to 7 × 10⁻¹⁵.
- **M/M/1-PS** mean sojourn: 4.99 against a theoretical 5.00.
- **Per-class GIPA PMF** reproduces `calibration.gipa_pv` draws exactly.

## `experiments/exp_async.py` (new, revision experiment E5)

**Purpose.** Regimes A and B, fleets N = 55, 80, 87, 104, 160 and 240, 30 seeds, 20 s simulated per run.
- N = 80, 160 and 240 match the synchronous densities.
- N = 55, 87 and 104 give an all-offload RSU utilisation of 0.5, 0.8 and 0.95.
- 20 policy variants are run, including -AC variants of greedy, independent softmax, GIPA, the planner and RBR (20 ms).
- Paired bootstrap comparisons cover GIPA − planner, LA − planner, near-live reference − planner and GIPA − all-local.

**Output.** `results/res_async.json`: `checks`, `fleets`, `plans`, `main`, `paired`, `per_seed`. The log is `results/exp_async_log.txt` (19 minutes on 2 cores). `verify_e5.sh` reruns and compares it on another machine.

**Results (28 Oct 2026, Regime A, 30 seeds).**

| Mean latency, ms (overall misses; share of runs with unbounded backlog) | N = 55 | 80 | 104 | 160 | 240 |
| --- | --- | --- | --- | --- | --- |
| Asynchronous planner | 7.05 | 8.05 | 10.26 | 12.99 | 14.73 |
| Learning automata | 9.21 | 9.66 | 10.56 | 13.23 | 16.07 |
| GIPA, slot prior | 9.26 | 9.61 | 10.57 | 13.48 | 18.38 (all-local) |
| GIPA, backlog prior from Stage 0 (ρ_heur) | 9.26 | 9.61 | 10.34 | 18.37 (all-local) | 18.38 (all-local) |
| GIPA, backlog prior from the plan's ρ | 9.26 | 9.61 | 10.34 | unstable (100%) | unstable (100%) |
| GIPA, 20 ms backlog broadcast | 9.32 | 9.72 | 10.59 | 23.48 (19% misses) | 29.40 (32%) |
| RBR, 20 ms broadcast | 9.63 (0.4%) | 23.66 (23%) | 30.80 (50%) | 38.32 (43%) | 44.00 (36%; 17%) |
| RBR, 100 ms broadcast | 18.28 (20%) | 57.84 (46%; 20%) | 74.25 (54%; 50%) | 33.75 (39%) | 85.22 (34%; 27%) |
| Near-live shortest expected delay (1 ms) | 9.06 | 14.11 (0.7%) | 21.08 (8%) | 26.39 (34%) | 28.99 (35%) |
| Greedy | unstable (100%) | unstable (100%) | unstable (100%) | unstable (100%) | unstable (100%) |
| All-local | 18.38 | 18.42 | 18.41 | 18.37 | 18.38 |

- **Persistent queues are stable under planned randomisation.**
    - The planner, GIPA with the slot prior and learning automata never let the backlog grow, with zero misses (at most 0.1%).
    - The planner's design utilisation of 0.79 is the measured one (0.79).
    - The planner is best at every load: GIPA − planner is 1.56 ms at N = 80 (CI 1.53–1.59), 0.50 ms at N = 160 (CI 0.39–0.62) and 3.64 ms at N = 240. Learning automata are within 0.24–2.15 ms.
- **A backlog-aware prior does not control the load by itself.**
    - Built from Stage 0's uniform split, the prior implies ρ ≥ 1 from N = 160, so GIPA's mask forces all-local.
    - Built from the plan's ρ, it makes RSUs look cheap, but GIPA does not follow the plan's offload fraction. It offloads 80% of tasks, and every run's backlog grows without bound.
    - The offload fraction has to be planned, not just the delay estimate.
- **Herding persists without simultaneity.** Asynchronous arrivals do not remove herding when vehicles act on shared, stale information:
    - RBR on a 20 ms broadcast has 23% misses at N = 80, against 0% for the planner. On a 100 ms broadcast, 20–50% of runs become unstable.
    - Even the near-live 1 ms reference herds (6–14 ms worse than the planner from N = 80, with up to 35% misses), because tasks in flight (2.7–7.3 ms of transfer) are invisible: vehicles see the same snapshot and pick the same RSU.
    - GIPA on a backlog broadcast is fine up to N = 104 but herds from N = 160.
    - This is the response to R1-6: herding comes from shared stale information, not from exact simultaneity.
- **Admission control** keeps queues bounded for the herding baselines, but at a high price. Greedy-AC rejects 45–87% of offloads and still misses every safety deadline (the local fallback after a rejection takes 13.7 ms). RBR-AC (20 ms) rejects 12–69% of offloads, with 35–78% safety misses.

**Regime B.**
- The planner is best at every load (7.05–18.96 ms), with at most 0.4% safety misses.
- Learning automata are 0.5–2.8 ms behind, with up to 3.4% safety misses.
- GIPA with the slot prior is fine up to N = 104. At N = 160 it misses every safety deadline, with 20% of runs unstable; at N = 240 it is all-local (100% safety misses).

**Implication for the paper.** The asynchronous model strengthens the planner-led framing. Planned, communication-free randomisation is stable and best with persistent queues. Information-driven policies herd on stale state even without simultaneous decisions. GIPA needs a planned offload level to stay stable at high load.

## `experiments/exp_prediction_obs.py` (new, revision experiment E1b, November 2026)

**Why.** Two simplifications of E1. (1) Its EWMA predictor learns from the *uncongested* latency of the action taken, which assumes the RSU returns a timing breakdown (isolated service time) with each result. (2) Its planned policy samples the plan and never reads a per-task prediction, so its robustness to prediction error was by construction; GIPA with planned prior (Algorithm 2) reads predictions through the mask (8) and the advantage (16).

**Code (`src/prediction.py`; defaults reproduce E1 exactly).**
- `obs` for the EWMA predictor: `'breakdown'` (E1), `'total'` (realised latency, congestion included, used as is) and `'total_debiased'` (realised latency minus the congestion the broadcast prior anticipates for that action).
- Policies `'plan_mask'` (plan restricted to the predicted-feasible set) and `'gipa_plan'` (Algorithm 2: mask, start from the restricted plan, K rounds of amplification on the advantage over the class average, diffusion towards the restricted plan with λ_k). With oracle predictions both equal `'plan'` draw for draw (checked at N = 80, 160, 240).
- `mask_z`: the RSU part of the mask uses the margin load of P1, planned mean plus z standard deviations of the sampled load (`plan_congestion(plan, z)`), instead of the mean load.
- `R0`: Stage 0's latency model for the plan and the class averages (Stage 0 latency-model error).

**Design.** Regime A, affine model, 30 matched seeds, N = 80, 160, 240. A: per-task noise σ 0.1–0.3 (and σ 0.1 with 10% optimistic bias), guard 0 and 0.1. B: AR(1) drift σ_η 0.1–0.3 with oracle and the three EWMA observation models, guard 0 (and 0.1 at σ_η 0.2). C: Stage 0 L_k and E_k scaled by 0.8 or 1.2.

**Results (8 Nov 2026; misses are safety-class rates; background misses are 0 throughout).**
- **The plan sampled directly** ('plan', the E1 policy) is unaffected by per-task error by construction: 15.03 / 16.69 / 17.26 ms at N = 80 / 160 / 240. Under drift it misses what all-local misses (safety tasks stay local): 2.4%, 15–16% and 24–25% at σ_η 0.1, 0.2, 0.3.
- **GIPA with planned prior, mask at the mean planned load** ('gipa_plan'): per-task error of 20% and 30% adds 0.4–0.8 ms and 4–19% safety misses; a 10% guard band raises them to 5–35%. A false local-infeasible prediction sends a safety task to an RSU whose planned load leaves it no room. Under drift (σ_η 0.2–0.3) with the EWMA (breakdown) predictor it misses 26–66% (39% at N = 160, σ_η 0.2): a vehicle that once predicts local as infeasible stops observing local and keeps offloading.
- **Mask at the margin load of P1 (z = 2)** ('gipa_plan_z2'): never more misses than the plan. Per-task error: at most 1.2% safety misses (N = 80, σ 0.3) and 0.14 ms; with the guard band 0–0.1%. Drift, any predictor: misses equal to the plan's or lower (oracle σ_η 0.3: 21.6% against 25.1% at N = 160), latency 0–0.14 ms below the plan.
- **Observability.** With the margin-load mask the planned policy gives the same misses under all three observation models; its latency differs by at most 0.07 ms. GIPA with the heuristic prior on realised latency ('total') double counts congestion and offloads less (16.80 against 18.39 ms at N = 160, σ_η 0.2); subtracting the prior's congestion restores the breakdown behaviour (18.63 ms). Reactive best response that averages its own realised latency no longer herds (18.51 ms, 17% safety misses, close to all-local's 15.9%).
- **Stage 0 latency model ±20%:** the plan loses at most 0.69 ms (N = 80, L × 1.2, E × 0.8), with no misses; GIPA with planned prior recovers up to 0.17 ms of it through the advantage.

**Output.** `results/res_prediction_obs.json` and `results/exp_prediction_obs_log.txt`; `verify_e1b.sh` reruns and compares.

## `experiments/exp_async_local.py` (new, revision experiment E5b, November 2026)

**Why.** In E5 a task kept local takes L_k from its arrival, as if the vehicle's processor were always idle. A vehicle generates one task per 20 ms on average and a background task takes 22 ms locally (Regime A), so a background vehicle that runs everything locally loads its processor to 1.1: its queue grows without bound. E5b gives every vehicle one persistent processor and repeats E5.

**Model changes (`src/async_sim.py`, `local='fifo'` or `'ps'`; the default `'none'` reproduces E5 exactly).**
- One on-board processor per vehicle that carries unfinished local work forward: FIFO (main) or processor sharing (sensitivity). Tasks rejected by admission control join it after the handshake.
- Own-queue mask (`+own` policies): the vehicle predicts local latency as the time until its processor is free plus L_k (exact under FIFO), uses it as the local score and in the deadline mask of GIPA, and for the plan redraws among the RSUs when local execution is predicted to miss. It uses only the vehicle's own state.
- Planner modes: `plan_async(local='fifo')` adds the M/D/1 local queue (mean sojourn and Erlang's waiting-time distribution, `md1_wait_cdf`) with a stability limit; `plan_async(local='fifo_own')` adds the own-queue mask through the closed-form admission model `local_admit_stats` (admitted share 1/(λ(E[v] + L − θ) + 1), E[v] = θ − (1 − e^(−λθ))/λ, θ = D − L < L), so the plan accounts for tasks the mask moves to the RSUs.
- New metrics: local latency, wait and miss rate, local backlog per vehicle by class, local utilisation, and separate RSU and local instability flags.

**Checks (8 Nov 2026).** FIFO M/D/1 at λ = 0.03/ms, s = 22 ms: mean wait 21.47 against 21.35 ms; P(W ≤ 18) 0.581 against 0.583; P(W ≤ 40) 0.811 against 0.814. Admission model: background (f = 0.2) admitted share 0.7308 against 0.7317 and latency 27.18 against 27.17 ms; safety (f = 0) 0.7109 against 0.7118 and 10.098 against 10.097 ms. With `local='none'`, `run_async` reproduces E5's per-seed latencies exactly (spot checks: six policies, N = 80 and 160, seed 0).

**Results (8 Nov 2026, Regime A, FIFO, 30 seeds; mean latency in ms, deadline misses).**

| Policy | N = 55 | 80 | 104 | 160 | 240 |
| --- | --- | --- | --- | --- | --- |
| GIPA, planned prior, own-queue mask (`plan+own`) | 7.05, 0% | 8.05, 0% | 10.44, 0.02% | 14.11, 0.03% | 17.27, 0.07% |
| GIPA, planned prior, E5 plan (ignores local queues) | 7.05, 0% | 8.05, 0% | 10.72, 1.0% | 19.68, 13.9% | 46.02, 36.6% |
| GIPA, planned prior, M/D/1-aware plan, no own-queue mask (`plan_q`) | 7.05, 0% | 8.05, 0% | 10.56, 0.95% | 15.88, 11.3% | 22.44, 25.2% |
| E5 plan with own-queue mask (`plan_e5+own`) | 7.05, 0% | 8.05, 0% | 10.48, 0.04% | 15.48, 1.7% | unstable (RSUs) |
| GIPA, heuristic prior, own-queue mask | 9.37, 0% | 9.74, 0% | 10.70, 0% | 16.29, 14.3% | all-local, unstable |
| Learning automata | 7.20, 0.07% | 8.35, 0.16% | 12.97, 3.8% | 19.95, 16.3% | 27.05, 31.3% |
| Near-live shortest delay (1 ms; sees own queue) | 9.06, 0% | 14.11, 0.7% | 21.29, 8.1% | 28.34, 42.8% | 33.33, 47.7% |
| RBR, 20 ms broadcast | 9.63, 0.4% | 23.75, 23.6% | 31.97, 55.0% | 43.04, 57.7% | 55.29, 62.2% |
| All-local | unstable (about 1.15 s, 83%) | unstable | unstable | unstable | unstable |

- `plan+own` is best at every load; paired differences at N = 160: −5.57 ms against the E5 plan (CI −5.64 to −5.50), −1.77 ms against `plan_q`, −1.37 ms against `plan_e5+own`, −5.84 ms against learning automata.
- Both parts are needed: the own-queue mask removes local misses, and the planner must account for the tasks it moves (the E5 plan with the mask overloads the RSUs at N = 240).
- GIPA with the heuristic prior misses 14% at N = 160 because its prior masks the RSUs for safety tasks, which then queue locally; at N = 240 it keeps every task local and is unstable.
- Regime B: `plan+own` 17.15 ms with 0.87% misses at N = 160; at N = 240 no plan meets the RSU margin (the capacity limit of Section VII-F).
- Processor-sharing sensitivity (Regime A): `plan+own` remains best, 14.98 ms with 2.1% misses at N = 160 and 19.52 ms with 8.4% at N = 240; the own-queue prediction is no longer exact under processor sharing.

**Output.** `results/res_async_local.json` (`checks`, `fleets`, `plans`, `main` by model, regime, N and policy, `paired`, `per_seed`) and `results/exp_async_local_log.txt` (about 60 minutes on 2 cores). `verify_e5b.sh` reruns and compares it. Figure R5b: `plotting/revision_figures.py --figure 10`.

## `src/carry.py`, `run_carry.sh` and the E13 scripts (new, revision experiment E13, November 2026)

**Why.** In every slotted experiment a task kept local takes L_k from the start of its slot, as if the vehicle's processor were idle. A vehicle has one task per 20 ms slot and a background task takes 22 ms locally in Regime A (30 ms in Regime B), so this favours local execution. E5b removed the assumption in the asynchronous model; E13 removes it in the slotted experiments of Sections VII-A to VII-D, VII-G and VII-I.

**Simulator change (`src/gipa_core.py`).** The environment variable `GIPA_LOCAL_CARRY` selects the local model; unset, every simulator reproduces the archived results exactly.
- `blind`: each vehicle keeps a backlog W of unfinished local work, W' = max(W + L_k 1{local} − 20, 0), and a local task's latency is W + L_k. Decisions are those of the paper.
- `own`: as `blind`, and each vehicle also uses its own backlog, which it knows without communication: the local score and the local term of mask (8) use W + L_k, and the planned policies redraw among the plan's RSUs when local execution is predicted to miss.

**Planning (`src/carry.py`).** `local_chain` solves the stationary distribution of one vehicle's backlog on a 1 ms lattice and returns the share of tasks run locally, their mean latency and miss rate, and whether the queue is stable. `carry_plan` wraps the affine, Eq. (5) and asymmetric planners: it replaces L_k by the mean latency of local tasks under the current plan and re-plans until the plan stops changing (damped fixed point). In `own` mode the planner's offload share is the share the RSUs receive; the vehicle samples the RSUs with a lower probability so that sampling plus redirects gives it (`sampling_share`), and the redirects of a vehicle that never samples (1/11 of Regime A background tasks, 1/3 of Regime B background tasks, every task of a class with L_k > D_k) enter the planners as a minimum offload share. A plan is infeasible if a local queue is unstable or the redirects alone exceed the planned offload.

**Checks (8 Nov 2026).**
- `experiments/check_carry.py` simulates the backlog recursion for 2,000,000 slots: the chain matches it, for example own mode at f = 0, 0.9090 against 0.9091 local share and 30.99 against 31.00 ms (L 22), 0.6667 and 35.00 ms in both (L 30). In blind mode at f = 0.1, L 22 (local utilisation 0.99) the mean latency differs by 3% (123.5 against 119.9 ms), because the queue mixes slowly.
- With `GIPA_LOCAL_CARRY` unset, the planner experiment reproduces `results/res_planner.json` exactly (8,245 numbers), and the asymmetric planner check passes.
- An earlier version of the chain gave a local share of 1 in own mode at f = 0, where the chain is periodic; it now averages the iterates (Cesàro mean). The affected own-mode experiments were rerun; the superseded files are kept in `results/carry_own/pre_fix/`.

**Results (8 Nov 2026, 30 matched seeds; mean latency in ms; idle values in brackets; misses are background-class rates unless stated).**
- **All-local is unstable in both regimes:** in Regime A the background queue grows by 2 ms per slot (84 ms mean over a 70-slot run, every background task misses); in Regime B about 353 ms. The mean depends on the run length, so all-local is no longer a reference point.
- **GIPA with the heuristic prior** is within 0.2 ms of the idle model up to N = 160 (18.57 [18.38] ms at N = 160, Regime A). From N = 240 its prior masks every RSU, the task falls back to local execution, and the fleet is all-local and unstable (84 ms). Masked softmax with the prior, entropy-regularised softmax and every ablation variant behave the same way; in blind mode masked softmax with the prior already collapses at N = 160.
- **GIPA with planned prior (carry-aware plan), Regime A:** own mode 15.17 / 17.85 / 19.38 / 20.38 ms at N = 80 / 160 / 240 / 320 with no misses [15.03 / 16.69 / 17.26 / 17.52]; blind mode 15.17 / 17.89 / 19.55 / 20.92 ms with 0 / 0.2 / 1.6 / 4.5% background misses. Under Eq. (5): own 15.90 / 18.51 / 20.04 / 21.00 ms, no misses; blind up to 7.2% background misses at N = 320.
- **Regime B:** the plan is feasible only at N = 80 (z = 0: 21.67 ms, 55.5% safety misses, own; 25.88 ms, 18.9% background misses, blind). Under Eq. (5) also at N = 160 (24.50 ms, 39.3% safety misses, own). Heuristic GIPA: 17.73 and 22.10 ms at N = 80 and 160 (every safety task misses, as in the idle model), all-local and unstable from N = 240.
- **Capacity (admission control, Regime A):** the plan is unchanged from the planner experiment; reactive best response rises to 20.1–39.7 ms with 2–65% background misses (idle: 18.1–18.5 ms, none).
- **Prediction error (N = 160):** the plan is unaffected by per-task error, as in E1. Under drift with the EWMA predictor: 17.68 ms, 1.8% background misses (own), 18.88 ms, 6.8% (blind). With the margin-load mask (z = 2) GIPA with planned prior never has more misses than the plan (noise σ 0.3: 0% against 20.3% for the mean-load mask, own mode, N = 160).
- **Asymmetric RSUs (Regime A):** the plan is best (fast RSU, affine: 13.96 / 16.94 / 18.56 / 19.64 ms, own); learning automata are within 0.1–1.1 ms with ≤ 0.5% background misses (≤ 1.3% under processor sharing). GIPA with the capacity prior collapses to all-local at N = 320 (symmetric RSUs: from N = 240). In Regime B with a fast RSU the asymmetric planner (which separates the classes across RSUs) is feasible at N = 80 (20.55 ms, 5.3% safety misses, own) where the symmetric grid planner is not; `exp_asym.py` reports the gap as infinite.
- **Learning automata (Regime A):** the best configuration is within 0.4 ms of the plan, slightly lower at N = 320 (20.26 against 20.38 ms), with 1.1–1.6% safety misses where the plan has none. Regime B: 20.7–45.7 ms with 26–68% background misses from N = 240.
- **Q-learning:** mostly local, hence 60–78 ms (Regime A) and 241–336 ms (Regime B), identical in own and blind mode (the learner never reads the backlog).
- **Heterogeneous tasks (own mode):** GIPA on the plan remains best in latency (het50: 12.06 / 15.06 / 17.23 / 18.39 ms [12.02 / 14.98 / 16.05 / 16.83]), but its safety misses rise to 2.1 / 15.5 / 24.8 / 30.8% [2.0 / 8.9 / 17.8 / 25.9]. Heuristic GIPA collapses to all-local at N = 320. Blind mode: GIPA on the plan 12.11 / 15.13 / 17.70 / 19.00 ms with 2.0 / 14.4 / 22.9 / 29.8% safety and 0 / 0.2 / 2.3 / 3.9% background misses; class-level sampling of the plan reaches 30.43 ms with 34.4% background misses at N = 320 (own: 18.98 ms, 1.3%).
- **Own against blind:** using the own backlog matters for the planned policies at high load (background misses 0 against 4.5% at N = 320) and for the prior-masked baselines; it makes no difference to the policies that never keep a task local by choice.

**Not rerun.** MAPPO (E7): its trained actors are not stored and its observation has no backlog.

**Scripts and output.**
- `run_carry.sh own|blind [output directory]` sets `GIPA_LOCAL_CARRY` and reruns `exp.py`, `exp_planner.py`, `exp_calibration.py`, `exp_capacity.py`, `exp_prediction.py`, `exp_prediction_obs.py`, `exp_asym.py`, `exp_la.py`, `exp_hetero.py` and `exp_qlearning.py` into `results/carry_<mode>/`, with the file names of the idle-model results. Checks inside the scripts that compare with idle-model files (symmetric reproduction in `exp_asym.py`, the archived Q-learning protocol) report differences in this mode, as expected.
- `experiments/compile_carry.py` writes `results/carry_summary.json` and `results/carry_summary.md`: for each section, the main policies under the idle, own and blind models.
- `verify_e13.sh own|blind` runs the chain check, checks the idle reproduction, reruns one mode into `results/carry_<mode>_laptop/` and compares every file.
- Logs: `results/run_carry_own_log.txt`, `results/run_carry_own_refix_log.txt`, `results/run_carry_blind_log.txt` (the blind heterogeneous-task and Q-learning runs were restarted separately: `results/run_carry_blind_hetero_log.txt`, `results/run_carry_blind_ql_log.txt`).

## `src/sla_mappo.py`, `run_e16.sh` and the E16 scripts (new, revision experiment E16, November 2026)

**Why.** Reviewer 1 asked for a comparison with recent multi-agent DRL methods. MAPPO, independent DQN and VDN are generic learners; SLA-MA-PPO (M. Georgiades et al., "Enhanced vehicular edge offloading through SLA-based deep reinforcement learning and risk-aware bandits", IEEE Trans. Netw. Service Manag., 2026; manuscript reference [24]; accepted version on City Research Online, eprint 38473) is designed for vehicular offloading and gives its network and PPO settings. Its public repository holds only the EdgeCloudSim base, so the method is reimplemented here.

**Method (`src/sla_mappo.py`).** Taken from [24]: one actor and one critic shared by all vehicles (12–64–64), clipped PPO with GAE (learning rate 3e-3, gamma 0.99, lambda 0.95, clip 0.2, four epochs, minibatch 32), the SLA admissibility mask (an action is masked when its full-service proxy exceeds 0.95 D or its risk exceeds 0.16; unmasked if every action is masked), the full-service proxy (path cost plus a residual learned from the vehicle's own outcomes on the selected action), and the reward −DelayRatio − beta_tau (1{S > 0.95 D} + 1{S > D}) − global KPI penalties − beta_risk R(a) + early bonus, with FailPct, VMUtil, p99Over and the normalised fleet size broadcast to every vehicle. Chosen here (not given in [24]): the action space (local or one of three RSUs), the 12 state features, the KPI definitions and targets (VMUtil = mean RSU load / 17 tasks, the Regime A safety cap; FailPct* 0.01, VMUtil* 0.8), the risk (the vehicle's exponentially weighted miss rate, weight 0.1), fixed global-penalty weight 1, entropy 0.01, gradient-norm clip 0.5, reward scaling by the running standard deviation of discounted returns, one 70-slot episode of all vehicles per update, and β_E = 0 (no energy model). `Adapter` gives the same interface to `gipa_core.Env` (`homog`) and `hetero.HetEnv` (`het50`: σ_h 0.5, σ_r 0.3, full coverage); with `GIPA_LOCAL_CARRY=own` both carry unfinished local work (E13).

**Development notes (17 Nov 2026).** (1) The optional dual update of the KPI weight in [24] diverged here (multiplier above 400 after 800 iterations, because warm-up misses keep FailPct above target); the fixed-weight form is used. (2) With the paper's learning rate and the final iterate, a first grid gave 20.1–22.2 ms at N = 160 (identical tasks) and latency moved by up to 3 ms between iterations (`results/sla/old_tune/`). The learning rate was added to the grid, and each run now keeps its best checkpoint on the tuning seeds (every 25 iterations from iteration 225; lowest latency among checkpoints within 0.5 points of the lowest miss rate), which favours the baseline.

**Tuning (`exp_sla.py --stage tune`).** Grid: learning rate {3e-3, 3e-4} × beta_tau {0.5, 2}, beta_risk 0.5, N = 160, one run each, tuning seeds 2000–2009. Chosen: identical tasks 3e-4 and 2 (16.88 ms, 0.02% misses on the tuning seeds); heterogeneous tasks 3e-3 and 0.5 (14.86 ms, 3.92%). The configuration is reused at every N and with carried-over work. Files: `results/sla/tune_homog.json`, `results/sla/tune_het50.json`.

**Results (17 Nov 2026; 30 evaluation seeds, mean of five runs; reference GIPA arm in brackets; paired 95% bootstrap CIs in `results/sla_summary.md`).**
- **Identical tasks, idle processor:** 15.38 / 17.58 / 18.33 / 18.95 ms with up to 0.16% misses [planned prior 15.03 / 16.69 / 17.26 / 17.52 ms, no misses]; the plan is 0.36–1.43 ms faster.
- **Identical tasks, carried-over local work (own):** 15.54 / 18.07 / 18.68 / 19.35 ms, 0.01–0.15% misses [15.17 / 17.85 / 19.38 / 20.38 ms, no misses]. The plan is faster at N = 80 and 160 (0.37 and 0.22 ms); SLA-MA-PPO is faster at N = 240 and 320 (0.69 and 1.03 ms) with 0.05% and 0.15% misses.
- **Heterogeneous tasks (het50), idle:** 13.08 / 15.25 / 16.17 / 16.74 ms with 2.2 / 4.9 / 6.6 / 7.5% misses; the better GIPA variant (operator at 80, ranking from 160) is 0.63–1.06 ms faster with 1.6–3.5 points fewer misses. The operator alone is 0.09 ms slower at N = 320.
- **Heterogeneous tasks, carried-over work:** 13.01 / 15.48 / 16.96 / 17.67 ms, 2.1–9.6% misses; the better GIPA variant is 0.26–0.95 ms faster with 1.4–2.2 points fewer misses. The operator alone is 0.27 and 0.73 ms slower at N = 240 and 320.
- **Overall:** the reference GIPA arm has lower latency and fewer misses in 14 of 16 cells, with both paired intervals excluding zero; the exceptions are identical tasks with carried-over work at N = 240 and 320.
- **Load spreading (N = 160, identical tasks, `results/sla_diagnostics.json`):** the five runs offload 23–74 tasks per slot (plan 39.5), mean largest RSU load 12.0–29.3 (plan 16.3); the slot-to-slot standard deviation of the offloaded count is up to 2.8 times the binomial value, and with carried-over work the lag-one autocorrelation reaches 0.81.
- **Training:** 2.4–11.2 minutes per run on one core, 8.4 core-hours for the 80 runs; 32 of 80 runs meet the 1% plateau criterion, most oscillate.
- **Longer training (`results/sla_long/`):** two runs at N = 160 (identical tasks, idle) trained for 1,000 iterations reach 17.08 and 17.09 ms (checkpoints 825 and 950) against 16.89 and 17.42 ms after 400 iterations; the planned prior gives 16.69 ms.

**Scripts and output.**
- `run_e16.sh [tune|1|…|6|merge|diag]` runs one stage; without an argument, every stage in order. Training jobs: 1 identical idle N = 80–240; 2 identical idle N = 320 and identical own N = 80–160; 3 identical own N = 240–320; 4 heterogeneous idle N = 80–240; 5 heterogeneous idle N = 320 and own N = 80–160; 6 heterogeneous own N = 240–320. Existing run files are skipped.
- `experiments/exp_sla.py` writes one file per run (`results/sla/<setting>_<idle|own>_N<N>_r<seed>.json`: evaluation per seed, learning curve, checkpoints, trained actor) and merges them into `results/res_sla.json`. `--outdir` writes to another directory (sensitivity runs, verification).
- `experiments/compile_sla.py` pairs SLA-MA-PPO with the archived comparators (`res_planner.json`, `res_mappo.json`, `carry_own/res_planner.json`, `res_hetero.json`, `carry_own/res_hetero.json`) on the same seeds: `results/sla_summary.json`, `results/sla_summary.md` (Table XII).
- `experiments/diag_sla.py` replays the evaluation of every N = 160 run and records offloaded count, largest RSU load and Jain index per slot.
- `verify_e16.sh` retrains one run (identical tasks, idle, N = 80, run 0) and recompiles the summary, comparing both with the archived files.
- Logs: `results/sla/tune_*_log.txt`, `results/sla/job*_log.txt`, `results/sla/merge_log.txt`, `results/sla/diag_log.txt`. Jobs 3 and 6 report a shell syntax error after their last run because `run_e16.sh` was edited while they ran; every run file was written.

## `src/backlog.py`, `experiments/exp_backlog.py` and `run_e17.sh` (new, revision experiment E17, November 2026)

**Why.** In E16, SLA-MA-PPO was faster than the carry-aware planned prior with identical tasks and carried-over local work at N = 240 and 320. The planned prior uses the vehicle's own backlog only to redirect tasks predicted to miss; SLA-MA-PPO's state includes the backlog. E17 adds the backlog to GIPA's per-task screening without adding communication. In the manuscript as Section VII-J (backlog screening paragraph), a row of Table XII and Supplementary Section S15 (17 Nov 2026).

**Rule (`src/backlog.py`).** A background vehicle (L_B = 22 ms > the 20 ms slot, so its processor queues) offloads with probability 1 when its own backlog W >= theta or W + L_B > D_B, and otherwise with probability q; the RSU is drawn uniformly (symmetric plan). Safety vehicles (L_S = 10 ms) never queue and sample the plan as before. The random part q >= q_min keeps vehicles that start with empty queues from offloading in step.

**Planning.** `chain_stats` solves the stationary distribution of W under (theta, q) exactly (linear system on a 1 ms lattice). `best_screen` finds, for a target background offload share, the threshold and q (interpolated on 401 values) with the lowest mean local latency, subject to q >= q_min. `backlog_plan` repeats the deadline-constrained grid of `planner._plan_allocation` (z = 2) with that local latency for each candidate share. Without a threshold the rule is the carry-aware plan's own, so the planned cost is never higher.

**Tuning.** q_min from {0.05, 0.1, 0.2} at N = 160 on seeds 2000–2009 (rule of Section VI-E): 17.12, 17.10, 17.11 ms, no misses; chosen 0.1, used at every N.

**Results (17 Nov 2026; identical tasks, Regime A, affine, own mode, 30 evaluation seeds; `results/backlog_summary.md`).**
- Screened plan: 15.03 / 17.11 / 18.10 / 18.84 ms at N = 80 / 160 / 240 / 320, no misses (thresholds 1, 3, 3, 5 ms).
- Against the carry-aware planned prior (E13; 15.17 / 17.85 / 19.38 / 20.38 ms): −0.14 / −0.73 / −1.27 / −1.54 ms.
- Against SLA-MA-PPO (E16; 15.54 / 18.07 / 18.68 / 19.35 ms, 0.01–0.15% misses): −0.51 / −0.96 / −0.58 / −0.51 ms with fewer misses; every paired 95% bootstrap interval excludes zero.
- Deterministic threshold (q_min = 0): 15.03 / 20.63 / 22.71 / 25.02 ms with up to 22.9% misses — identical vehicles that start with empty queues offload in step (herding).
- Planned cost and simulation agree within about 0.1 ms (e.g. N = 240: 17.94 planned, 18.07 simulated with q_min = 0.05).

**Scripts and output.** `run_e17.sh` sets `GIPA_LOCAL_CARRY=own` and runs `experiments/exp_backlog.py` (tuning, evaluation, comparison; about 2 minutes), writing `results/res_backlog.json` (plans, per-seed results, paired differences), `results/backlog_summary.md` and `results/run_e17_log.txt`. Comparators are read from `results/carry_own/res_planner.json` and `results/res_sla.json`, unchanged. Rerunning reproduces `results/res_backlog.json` exactly.

## `experiments/exp_marl_value.py`, `compile_marl.py`, `check_overload_regimeB.py` and `run_e14.sh` (revision experiment E14, November 2026)

Independent DQN and VDN with one Q-network shared by all vehicles (NumPy, two hidden layers of 64 tanh units, the MAPPO
actor's inputs, reward −latency/10 − 3 × miss, discount 0). Training: 4,000 slots on environment seed indices 1000–1049,
ε-greedy exploration 1.0 → 0.05 over 70% of training, replay buffer 2,000 slots, 32 slots per Adam step (lr 1e-3, clip 5),
five runs per method and fleet size (seeds 8000–8004). Execution rule tuned on seeds 2000–2009 (ε-greedy 0.02 or Boltzmann
T ∈ {0.1, 0.25, 0.5, 1, 2}; lowest latency with at most 0.1% class-standardised misses). Evaluation on the 30 evaluation
seeds. `--mask deadline` uses the deadline-only mask (as MAPPO at N ≥ 240). `check_overload_regimeB.py` gives the Regime B
selected-overload rates of Section VII-B (z = 0: 47.2% of RSU-slots; z = 2: 2.8%). `compile_marl.py` writes
`results/marl_summary.json`. Run `./run_e14.sh` (about 40 min on two cores).

Outputs: `results/res_marl_value.json`, `results/res_marl_value_dmask.json`, `results/marl_summary.json`,
`results/exp_marl_value_log.txt`, `results/exp_marl_value_dmask_log.txt`, `results/check_overload_regimeB_log.txt`.

## `src/tan_style.py`, `experiments/exp_tan.py`, `sens.py`, `summarise.py` and `run_e15.sh` (revision experiment E15, November 2026)

A re-solved probabilistic plan in the manner of Tan et al. (one action distribution per class and coverage group, re-solved
every slot by SLSQP from that slot's true predicted aggregates, RSU deadline constraints on the expected load, z = 0 or
z = 2, sampled without a per-task check, local execution added) against GIPA's per-task layer, on the six E12 scenarios,
N = 80–320, 30 seeds, with an idle processor (`--part batch`), with carried-over local work (`--part carry`, run with
`GIPA_LOCAL_CARRY=own`) and with a fluctuating fleet (`--part fluct`). Operator settings tuned on seeds 4600–4609
(`--part tune`) by the rule of Section VI-E (lowest latency within 0.5 percentage points of the lowest miss rate).
`sens.py` repeats the carry cells with the deadline constraints dropped in infeasible slots (`TAN_RELAX=1`).
Run `./run_e15.sh` (about 45 min on two cores).

Outputs: `results/tune_batch.json`, `results/res_tan_batch.json`, `results/res_tan_carry.json`, `results/res_tan_fluct.json`,
`results/res_tan_carry_relax.json`, `results/E15_summary.md`, `results/run_e15_log.txt`.

## `plotting/revision_figures.py` (new)

**Purpose.** Draws the revision's summary figures from the result files; it never runs a simulator. Current manuscript mappings are in [`MANUSCRIPT_MAP.md`](MANUSCRIPT_MAP.md).

| Figure | File stem | Content |
| --- | --- | --- |
| R1 | `fig_r1_main_comparison` | Latency against N: planner, GIPA, learning automata, MAPPO, all-local; affine and Eq. (5) panels |
| R2 | `fig_r2_prediction_error` | E1: misses by error arm and guard band; prior fleet-size misspecification |
| R3 | `fig_r3_capacity` | E3: selected RSU load against capacity; rejections under admission control |
| R4 | `fig_r4_qlearning_runs` | E4: each of 10 Q-learning training runs |
| R5 | `fig_r5_asynchronous` | E5: latency and misses with asynchronous arrivals |
| R6 | `fig_r6_learning_automata` | E6: learning curve; density change 80 → 240 → 80 |
| R7 | `fig_r7_asymmetric_rsus` | E9: fast-RSU herding ratio; latency against N |
| R8 | `fig_r8_decision_cost` | E8: decision time and runtime payload |
| R9 | `fig_r9_heterogeneous_tasks` | E12: heterogeneous tasks; GIPA on the plan against per-task ranking |

**Statistics.** Seed-level 95% percentile-bootstrap intervals from `src/stats.py`. Each figure uses its own generator (seed 20260804), so any figure can be regenerated alone. Plotted values are written to `results/res_revision_figures.json`.

**Style.** Every method has one fixed colour and marker across all figures, from a categorical palette validated for colour-vision deficiency (adjacent ΔE ≥ 9.1). Distinct markers and legends mean identity never depends on colour alone.

**Editing legends, titles and labels.** Every word in a figure is set in the block marked `EDIT HERE` near the top of the script: `LEGEND` holds the method names used in every legend and bar label; `TEXT['R1']` … `TEXT['R9']` hold each figure's panel titles, axis labels, tick labels and notes; `PANEL_TITLES` switches panel titles between the full description (`'full'`), the letter only (`'letter'`, leaving the description to the caption) and none (`'none'`). Legend positions are on the lines marked `# EDIT: legend position`. The same applies to `plotting/gipa_framework.py` (its `TEXT` block). The values in `results/res_revision_figures.json` are keyed by legend label, so renaming a label renames its key.

**Outputs.** PNG (300 dpi, used by the Word build), PDF, SVG and TIFF (600 dpi, RGB, LZW). `plotting/export_submission_tiffs.py` then writes `figures/submission/Figure_n.tif` in manuscript order and checks each file against the Wiley figure guidelines used by IET journals (line art 600 dpi, width 80–180 mm, under 10 MB). Figures wider than 180 mm are re-tagged to print at 180 mm (pixels unchanged). The revised manuscript has eight figures; Figs. 2 and 8 were redrawn for the IET layout and are taken from the manuscript file, and Fig. 8 shows the persistent-queue experiment E5b.

| Manuscript figure | Source |
| --- | --- |
| 1 | `figures/fig_1.png` (raster original, low resolution) |
| 2 | `fig_gipa_framework` |
| 3 | R1 `fig_r1_main_comparison` |
| 4 | R2 `fig_r2_prediction_error` |
| 5 | R3 `fig_r3_capacity` |
| 6 | R5 `fig_r5_asynchronous` |
| 7 | R7 `fig_r7_asymmetric_rsus` |
| 8 | R4 `fig_r4_qlearning_runs` |
| 9 | R6 `fig_r6_learning_automata` |
| 10 | R8 `fig_r8_decision_cost` |
| 11 | R9 `fig_r9_heterogeneous_tasks` |

**Usage.**
- `PYTHONPATH=src python plotting/revision_figures.py [--figure 1..9] [--titles full|letter|none]`
- `python plotting/gipa_framework.py`
- `python plotting/export_submission_tiffs.py`
- Fig. R8 with another machine's timings: add `--complexity <file in results/>`.

## Results files

| File | Produced by | Content |
| --- | --- | --- |
| `results/res_main.json` | Extracted from the statistics file (see note) | Across-seed means; undefined Jain fairness stored as `null` |
| `results/res_main_seed_level.json` | Instrumented run (rebuilt: `experiments/seed_level_main.py`) | 30 per-seed records per regime, density and policy, with miss and offload counts, `nS`, `nB`, evaluated slots |
| `results/res_main_statistics.json` | Statistics script (rebuilt: `experiments/stats_main.py`) | Bootstrap means and 95% CIs, nine paired comparisons with TOST, Clopper–Pearson zero-miss bounds |
| `results/res_planner.json` | `experiments/exp_planner.py` | Planner results (see above) |
| `results/res_capacity.json` | `experiments/exp_capacity.py` | E3 capacity results: soft and admission-control modes for 11 policies |
| `results/res_prediction.json` | `experiments/exp_prediction.py` | E1 prediction-error results: 45 conditions, 10 policies |
| `results/res_qlearning.json`, `results/exp_qlearning_log.txt` | `experiments/exp_qlearning.py` | E4: Q-learning over 10 training runs per configuration, with two-level bootstrap CIs |
| `results/res_la.json`, `results/exp_la_log.txt` | `experiments/exp_la.py` | E6: learning-automata tuning grid, main comparison with paired CIs, learning curves, density-change traces |
| `results/res_complexity.json`, `results/complexity_runs/run1.log` | `experiments/exp_complexity.py` | E8: decision timings, slow-time-scale and training costs, communication payloads, asymptotic complexity (reference machine) |
| `results/res_async.json`, `results/exp_async_log.txt` | `experiments/exp_async.py` | E5: asynchronous arrivals, persistent PS queues, stability metrics, rebuilt Table III check |
| `results/res_hetero.json`, `results/exp_hetero_log.txt` | `experiments/exp_hetero.py` | E12: heterogeneous tasks and partial coverage; tuning grid, nine policies, paired CIs |
| `results/res_asym.json`, `results/exp_asym_log.txt` | `experiments/exp_asym.py` | E9: asymmetric RSUs, herding metrics, asymmetric plans, paired comparisons |
| `results/carry_own/`, `results/carry_blind/` | `run_carry.sh own`, `run_carry.sh blind` | E13: the slotted result files rerun with carried-over local work, same names and structure as the idle-model files; `carry_own/pre_fix/` keeps the own-mode files superseded by the chain fix |
| `results/carry_check.json`, `results/carry_summary.json`, `results/carry_summary.md` | `experiments/check_carry.py`, `experiments/compile_carry.py` | E13: local-queue chain against direct simulation; summary tables (idle, own, blind) |
| `results/sla/`, `results/res_sla.json`, `results/sla_summary.json`, `results/sla_summary.md`, `results/sla_diagnostics.json`, `results/sla_long/` | `experiments/exp_sla.py`, `experiments/compile_sla.py`, `experiments/diag_sla.py` | E16: SLA-MA-PPO runs, merged results, paired comparison with the planned policies (Table XII), load-spreading diagnostics, 1,000-iteration sensitivity runs |
| `results/res_calibration.json` | `experiments/exp_calibration.py` | E2 results under Eq. (5): special-case check, comparison, ablation, capacity sweep |
| `results/res_mappo_dmask_reference.json`, `results/train_mappo_highdensity_log.txt` | `exp_mappo.py --stage merge --out ...` (reference-machine run, 27 Oct) | E7 deadline-mask runs at N = 240 and 320, for the cross-machine check |
| `results/mappo/tune.json`, `results/mappo/train_*.json`, `results/res_mappo.json` | `experiments/exp_mappo.py` (via `train_mappo.sh`) | E7 tuning grid, per-process training outputs (intermediates, kept on the machine that ran them), merged results with curves and networks. The merged file and `results/train_mappo_log.txt` from the 27 Oct laptop run are archived here |

Note: running `exp.py` writes `res_main.json` with `NaN` where the archived file has `null`. The numbers are otherwise identical (see the verification log).

## Historical audit of the v1.0.0 manuscript

The following audit records gaps and subsequent repairs in the original v1.0.0 draft. Its manuscript locations refer to that earlier draft. Use `MANUSCRIPT_MAP.md` for the current revised paper; the historical table is retained to document the package's development.

| Missing script | Produced | Manuscript location |
| --- | --- | --- |
| ~~Instrumented principal run~~ **Rebuilt:** `experiments/seed_level_main.py`, byte-identical | `res_main_seed_level.json` (per-seed counts) | Section VI-G, all CIs |
| ~~Statistics script~~ **Rebuilt:** `experiments/stats_main.py` reproduces every value | `res_main_statistics.json`: bootstrap CIs, paired tests, TOST, Clopper–Pearson | Sections VI-G, VII; Figs. 3–8 |
| ~~Processor-sharing check~~ **Rebuilt (E5):** `async_sim.check_table3()` reproduces 21.64, 45.92, 95.48 and 195.25 ms | Discrete-event comparison, 4,000 replications, seed 0 | Table III, Section VI-C, Appendix H.1 |
| Exact-arithmetic verification | 60,000 sampled configurations, seed 11 | Section V-A, Appendix G.1 |
| Live-feedback experiment | Additive live congestion signal, 4,000 bootstrap resamples | Section VII-E-4, Appendix H.2 |
| Aggregate-load relaxation | Class-aggregated centralised reference | Section VII-E-1, Appendix H.3 |
| Regime B split-policy diagnostic | Safety-offloading split policy | Section VII-E-2, Appendix H.4 |

## Known issues in the archived code

These are documented, not fixed, so that v1.0.0 results still reproduce bit for bit. The four Q-learning issues are fixed in the revision's `src/qlearning.py` (E4), which leaves `gipa_core.QL` untouched.

- **Q-learning training seeds overlap evaluation seeds.** Training environments cycle seed indices 0–6, which are also evaluation seeds 0–6.
- **Q-learning has no discount term.** The update is Q ← Q + 0.1 (r − Q), i.e. discount 0, which makes it a contextual bandit. The manuscript should say so.
- **Q-learning state carries over.** `QL.mu` is not reset between evaluation episodes.
- **Single Q-learning training run.** One exploration seed (5999).
- **p95 definition.** `run` computes the 95th percentile across vehicles within each slot, then averages over slots. The manuscript describes a per-run p95.
- **Output location.** `exp.py` writes to the working directory rather than `results/`.
- **Class-independent congestion.** Every co-located task adds 0.55 ms regardless of class, while local latency differs by class (10 vs 22 ms). This is inconsistent with Eq. (5) with class-dependent compute; experiment E2 (`src/calibration.py`) shows the affine model is exactly its c_S = c_B special case and evaluates every policy under the calibrated class-dependent model.

## Verification log

27 October 2026, on CPython 3.12.3 with NumPy 2.4.4 and Matplotlib 3.10.8:

- `src/gipa_core.py`, `experiments/exp.py` and `plotting/generate_figures.py` have syntax trees identical to v1.0.0 once docstrings are removed. Comments, docstrings and layout changed; computation did not.
- A full rerun of `exp.py` produced `res_main.json` with MD5 `b4b3c084c3b8c97562c6c58097b6b923`. This is the fingerprint recorded in `docs/statistical_analysis.md`. All 1,156 numeric values equal the archived `results/res_main.json`; the remaining 44 entries differ only as `NaN` versus `null` for undefined Jain fairness.
- All six figures regenerated by the documented plotting script are pixel-identical to those from the original script.

28 October 2026, revision experiments E6 and E8 (reference machine, CPython 3.13.17, NumPy 2.5.3):

- `learning_automata.check_updates()` passes, and `exp_complexity.check_gipa_scalar()` confirms the scalar GIPA chooses the same action as `gipa_core.gipa` for every vehicle on five seeds.
- `exp_la.py` and `exp_complexity.py` leave `gipa_core.py` untouched. The GIPA, planner and all-local references recomputed inside `exp_la.py` equal the archived and E2 values (for example 18.38 ms for GIPA and 16.69 ms for the planner at N = 160, Regime A).
- `exp_la.py` is deterministic. A rerun on another machine should match `results/res_la.json` to rounding; `exp_complexity.py` timings depend on the machine.

28 October 2026, revision experiment E9 (reference machine, CPython 3.13.17, NumPy 2.5.3, SciPy 1.18.1):

- With sp = (1, 1, 1) and the uniform prior, `asymmetric.run_asym` reproduces `res_main.json` (affine) and `res_calibration.json` (Eq. 5) latencies exactly for local, greedy, independent softmax, masked softmax, prior-adjusted and entropy-regularised softmax, GIPA and RBR, in both regimes at all four densities (maximum difference 0).
- `plan_asym` with sp = (1, 1, 1) matches the grid planner's objective to within 7 × 10⁻⁶ (relative), or improves on it.
- After the `feasible_set` change in `learning_automata.py`, E6 spot checks (Regime A affine N = 160 L_R-εP; Regime B Eq. (5) N = 160 pursuit) are bit-identical to `res_la.json`.

28 October 2026, revision experiment E5 (reference machine, CPython 3.13.17, NumPy 2.5.3):

- `async_sim.check_table3()` reproduces the manuscript's Table III: 21.64, 45.92, 95.48 and 195.25 ms against 21.64, 45.90, 95.48 and 195.25 ms. The original script was missing.
- `check_batch()` agrees with the E2 batch formula to 7 × 10⁻¹⁵; `check_mm1()` gives 4.989 against 5.000; and `check_gipa_probs()` reproduces `calibration.gipa_pv` draws exactly.

28 October 2026, rebuilt statistics script (reference machine, NumPy 2.5.3, SciPy 1.18.1):

- `experiments/stats_main.py --check` reproduces all 3,983 numbers of `results/res_main_statistics.json` (largest scaled difference 3.5 × 10⁻¹⁵) with identical structure. The output is not byte-identical here because of last-digit differences in newer NumPy/SciPy; the byte check can be made in the pinned v1.0.0 environment with `verify_stats.sh`.
- `experiments/seed_level_main.py --check` regenerates `results/res_main_seed_level.json` byte for byte (MD5 `8727bda798913aae39beeda4efc1f707`), and `stats_main.py` run on that output reproduces the archived statistics (largest scaled difference 3.5 × 10⁻¹⁵ under NumPy 2.5.3 / SciPy 1.18.1).

29 October 2026, revision experiment E12: with no heterogeneity and full coverage, `hetero.run_het` reproduces E2's per-seed latencies for the planner and GIPA exactly (maximum difference 0).

8 November 2026, revision experiment E13: with `GIPA_LOCAL_CARRY` unset, `exp_planner.py` reproduces `results/res_planner.json` exactly (8,245 numbers) and the asymmetric planner check of `exp_asym.py` passes; `experiments/check_carry.py` agrees with 2,000,000-slot simulations of the backlog recursion (largest difference 3% in mean latency, blind mode at local utilisation 0.99).

17 November 2026, revision experiment E16: `verify_e16.sh` retrains SLA-MA-PPO for identical tasks, idle processor, N = 80, run 0, and reproduces the archived evaluation exactly (310 numbers, largest difference 0); recompiling `results/sla_summary.json` from `results/res_sla.json` reproduces it exactly (640 numbers). The comparators in `compile_sla.py` are read from the archived files, unchanged.

18 November 2026, revision experiments E14 and E15, rerun in full from the recovered code (`run_e15.sh`, then `run_e14.sh`): E15 reproduces the values quoted in Section VII-D and Table S20.4 (e.g. re-solved plan, identical tasks, N = 160: 16.75 ms, 10.9% misses; tuned β = 0.4); E14 reproduces 17.80 ms (independent DQN) and 17.49 ms (VDN) at N = 160, and `check_overload_regimeB.py` gives selected-overload rates of 47.2% (z = 0) and 2.8% (z = 2).


## `experiments/exp_priority.py` and `run_e18.sh` (new, revision experiment E18, November 2026)

**Purpose.** Separates the two changes made by backlog screening (E17): the offload target, which E17 re-optimises with the screened local latency, and the order in which background vehicles offload. E18 keeps the targets of the carry-aware planned prior of E13 (`planner.plan_allocation`, z = 2, own-queue mode): the safety plan is unchanged and background vehicles keep the plan's effective offload share, sampling plus redirects. `backlog.best_screen` then chooses the threshold and random offload probability that give this share with the lowest mean local latency (minimum random probability 0.1, as tuned in E17; the arm `priority_det` allows 0).

**Scripts and output.** `run_e18.sh` sets `GIPA_LOCAL_CARRY=own` and runs `experiments/exp_priority.py` on the 30 evaluation seeds (about 2 minutes), writing `results/res_priority.json` (screens, per-seed results, paired differences against `results/carry_own/res_planner.json`, 10,000 bootstrap resamples) and `results/priority_summary.md`.

## E19: completion-gated feedback for SLA-MA-PPO (`src/sla_mappo.py`, `experiments/exp_sla.py`, `run_e19.sh`, `experiments/compile_sla_gated.py`; revision, November 2026)

**Purpose.** In E16 each vehicle receives, at its next decision, the latency and deadline outcome of every task of the previous slot, including tasks still running. E19 uses an outcome only once it exists at decision time 20 t' ms: a latency (residual update of the full-service proxy, FailPct and p99Over) when the task completes, at 20 t + latency, and a deadline outcome (risk update) at completion or, for a late task, when its deadline passes. FailPct and p99Over are computed over the tasks completed during the previous slot (kept from the slot before if none completed); VMUtil uses the RSU association counts of the previous slot. Events are applied in time order (`GatedSLAState`). With `GIPA_SLA_FEEDBACK` unset the code reproduces E16 exactly (checked on archived run files).

**Scripts and output.** `run_e19.sh reeval` evaluates the 80 archived E16 actors with gated feedback (`results/sla_gated/reeval_*.json`). `run_e19.sh 1` … `6` retrain the 80 runs with gated feedback and the E16 tuned settings (`results/sla_gated/<setting>_<carry>_N<N>_r<run>.json`), in jobs of 60–90 minutes on one core. `run_e19.sh compile` runs `experiments/compile_sla_gated.py`, which pairs idealised (E16), gated and re-evaluated SLA-MA-PPO with fixed GIPA variants (planned prior; backlog screening; per-task ranking; amplification) and reports seed-level intervals and two-level intervals that resample the training runs and the evaluation seeds independently (`results/sla_gated_summary.{json,md}`).
