# Revised manuscript (R1): where every result comes from

Numbering follows the revised manuscript ITS-2026-08-0579 R1 (Figs 1–8, Tables 1–8) and its Supplementary
Material (Sections S1–S20). Experiment codes (E1–E19) are those of `docs/SCRIPTS.md`, which documents every script,
its inputs, outputs, run time and verification. All evaluations use the 30 matched evaluation seeds
(environment seeds 1000–1029, action-sampling seeds 5000–5029); tuning and training use disjoint seeds.

## Figures

| Revised paper | Content | Experiment | Script that runs it | Result file | Figure script (stem) |
| --- | --- | --- | --- | --- | --- |
| Fig. 1 | System model | — | — | — | `plotting/concept_figures.py` (`fig_c1_system_architecture`) |
| Fig. 2 | Planning and per-task rules, including two uses of local queues | — | — | — | Final revision-15 editable SVG: `figures/manuscript/figure2_gipa_four_rules.svg` (PNG/PDF alongside it). `plotting/gipa_framework.py` is the earlier diagram. |
| Fig. 3 | User equilibrium against system optimum | Analysis (Section 5) | — | — | `plotting/concept_figures.py` (`fig_c4_equilibrium_optimum`) |
| Fig. 4 | Mean latency against fleet size, affine model and Eq. (5) | E2, E6, E7, main | `experiments/exp.py`, `exp_planner.py`, `exp_calibration.py`, `exp_la.py`, `exp_mappo.py` | `results/res_main.json`, `res_planner.json`, `res_calibration.json`, `res_la.json`, `res_mappo.json` | `plotting/revision_figures.py --figure 1` (`fig_r1_main_comparison`) |
| Fig. 5 | Robustness to prediction error | E1 | `experiments/exp_prediction.py` | `results/res_prediction.json` | `revision_figures.py --figure 2` (`fig_r2_prediction_error`) |
| Fig. 6 | RSU capacity and admission | E3 | `experiments/exp_capacity.py` | `results/res_capacity.json` | `revision_figures.py --figure 3` (`fig_r3_capacity`) |
| Fig. 7 | Learning automata | E6 | `experiments/exp_la.py` | `results/res_la.json` | `revision_figures.py --figure 6` (`fig_r6_learning_automata`) |
| Fig. 8 | Asynchronous arrivals with persistent queues | E5, E5b | `experiments/exp_async.py`, `exp_async_local.py` | `results/res_async.json`, `res_async_local.json` | `revision_figures.py` (`fig_r5b_async_local_queue`) |
| Fig. S20.1 | Component ablation, heuristic prior | main | `experiments/exp.py` | `results/res_main.json` | `plotting/generate_figures.py` |
| Fig. S20.2 | Unequal RSUs (herding) | E9 | `experiments/exp_asym.py` | `results/res_asym.json` | `revision_figures.py --figure 7` (`fig_r7_asymmetric_rsus`) |
| Fig. S20.3 | Tabular Q-learning over ten training runs | E4 | `experiments/exp_qlearning.py` | `results/res_qlearning.json` | `revision_figures.py --figure 4` (`fig_r4_qlearning_runs`) |
| Fig. S20.4 | Decision-kernel benchmarks and payload | E8 | `experiments/exp_complexity.py`, `run_complexity_repeats.sh` | `results/res_complexity.json`, `results/complexity_runs/` | `revision_figures.py --figure 8` (`fig_r8_decision_cost`) |

Submission TIFF files: `plotting/export_submission_tiffs.py`.

## Tables

| Revised paper | Content | Experiment | Script | Result file |
| --- | --- | --- | --- | --- |
| Table 1 | Approach families | — (literature) | — | — |
| Table 2 | Planning latency model of the planned prior | E2 | `experiments/exp_calibration.py` | `results/res_calibration.json` |
| Table 3 | Asymptotic cost per method | E8 (measured times in Section 7.3) | `experiments/exp_complexity.py` | `results/res_complexity.json` |
| Table 4 | Methods compared | — | — | — |
| Table 5 | Matched comparison at N = 160 | main, E2, E4, E6, E7, E14, E19 (SLA row uses gated feedback) | `experiments/exp.py`, `exp_planner.py`, `exp_qlearning.py`, `exp_la.py`, `exp_mappo.py`, `exp_marl_value.py`, `exp_sla.py` | `results/res_main.json`, `res_main_statistics.json`, `res_planner.json`, `res_qlearning.json`, `res_la.json`, `res_mappo.json`, `res_marl_value*.json`, `marl_summary.json`, `res_sla.json`, `results/sla_gated/`, `sla_gated_summary.json` |
| Table 6 | Regime B under processor sharing | E2 | `experiments/exp_calibration.py`, `exp_planner.py` | `results/res_calibration.json`, `res_planner.json` |
| Table 7 | SLA-MA-PPO against GIPA (intervals over training runs and seeds) | E16, E17, E19 | `run_e16.sh`, `experiments/compile_sla.py`, `run_e17.sh`, `experiments/compile_sla_gated.py` | `results/sla_summary.md`, `res_sla.json`, `res_backlog.json`, `sla_gated_summary.md` |
| Table 8 | Extended scenarios at N = 160 | E5b, E12, E13, E15, E17 | `exp_async_local.py`, `exp_hetero.py`, `run_carry.sh`, `exp_tan.py`, `run_e17.sh` | `results/res_async_local.json`, `res_hetero.json`, `carry_summary.md`, `res_tan_*.json`, `E15_summary.md`, `res_backlog.json` |
| Table S20.1 | Full matched comparison | as Table 5 | as Table 5 | as Table 5 |
| Table S20.2 | Heterogeneous tasks | E12 | `experiments/exp_hetero.py` | `results/res_hetero.json` |
| Table S20.3 | Carried-over local work | E13 | `run_carry.sh`, `experiments/compile_carry.py` | `results/carry_summary.md` |
| Table S20.4 | Re-solved probabilistic plan | E15 | `exp_tan.py` | `res_tan_*.json`, `E15_summary.md` |
| Table S20.5 | Full comparison with SLA-MA-PPO (gated primary, idealised sensitivity) | E16, E17, E19 | `experiments/compile_sla.py`, `experiments/compile_sla_gated.py` | `results/sla_summary.md`, `results/sla_gated_summary.md` |
| Table S14.5 | SLA-MA-PPO with idealised and completion-gated feedback | E19 | `run_e19.sh`, `experiments/compile_sla_gated.py` | `results/sla_gated/`, `sla_gated_summary.{json,md}` |
| Section 7.4, Section S15 (queue priority at the original targets) | Backlog screening at the planned prior's offload targets | E18 | `run_e18.sh`, `experiments/exp_priority.py` | `results/res_priority.json`, `priority_summary.md` |

## Statistics

Seed-level bootstrap intervals, paired differences, TOST and Clopper–Pearson bounds: `src/stats.py`,
`experiments/stats_main.py`, `experiments/seed_level_main.py` → `results/res_main_statistics.json`,
`results/res_main_seed_level.json` (Section 6.7; Supplementary Section S19).

## Verification

`verify_*.sh` re-run a representative part of each experiment and compare with the archived results; the
verification log is at the end of `docs/SCRIPTS.md`. `SHA256SUMS.txt` lists the checksum of every file.
