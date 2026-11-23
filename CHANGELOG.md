# Changelog

## 1.10 - 2026-11-23

- Published the first-revision experiment package (E1–E19) in the versioned GitHub repository.
- Updated the public README, citation metadata and manuscript map; added a cross-platform checksum verifier.
- Preserved the experiment implementations, configurations and archived result files without changes.
- Retained historical logs and results for provenance; omitted manuscript appendix and draft-caption text from this code distribution.
- Included the final revision-15 Figure 2 as SVG, PDF and PNG assets.
- Restored the complete MIT licence notice from the original public v1.0.0 repository.

Historical source-package changelog follows.

## Unreleased (revision working copy) - 2026-10-27

### Changed
- Manuscript map (22 Nov 2026): sections and tables renumbered to the IET style of the revised manuscript (Sections 1–9, Tables 1–8); supplementary figures S1 and S2 are now Figs. S8.1 and S8.2.
- Fig. 1 (22 Nov 2026): the per-class capacity label and its dotted marker were removed from the RSU panels of `plotting/concept_figures.py` (C1); figures regenerated. Notation in the manuscript and Supplementary Material aligned (class probabilities, backlog, guard band and related symbols); no script changes behaviour.

### Added
- Figures (22 Nov 2026): `plotting/export_submission_tiffs.py` now maps the eight figures of the revised manuscript (Figs. 2 and 8, redrawn for the IET layout, are taken from the manuscript, which is passed as the first argument; Fig. 8 shows the persistent-queue data of E5b); `figures/submission/` holds `Figure_1.tif` … `Figure_8.tif` and `MANIFEST.txt`.
- E19 (22 Nov 2026): SLA-MA-PPO with completion-gated feedback. `src/sla_mappo.py` gains `GIPA_SLA_FEEDBACK=gated`: a latency (residual, fleet KPIs) is used only after the task completes and a deadline outcome (risk) only after completion or deadline expiry; the default (`ideal`) reproduces E16 exactly. `experiments/exp_sla.py --stage reeval` evaluates the archived E16 actors with gated feedback; `run_e19.sh` retrains all 80 runs with gated feedback (E16 tuned settings) into `results/sla_gated/`; `experiments/compile_sla_gated.py` compares idealised, gated and re-evaluated SLA-MA-PPO with fixed GIPA variants, with seed-level and two-level (training runs and seeds) bootstrap intervals (`results/sla_gated_summary.{json,md}`).
- E18 (22 Nov 2026): queue priority at the original targets (`experiments/exp_priority.py`, `run_e18.sh`; `results/res_priority.json`, `results/priority_summary.md`): backlog screening applied at the background offload share of the carry-aware planned prior, separating the screening rule from the re-optimised target of E17. No existing file changes behaviour.
- E17 (17 Nov 2026): backlog-aware screening of the planned prior with carried-over local work (`src/backlog.py`: exact stationary backlog chain, screened plan, simulation; `experiments/exp_backlog.py`, `run_e17.sh`; `results/res_backlog.json`, `results/backlog_summary.md`). No existing file changes behaviour.
- E16 (17 Nov 2026): a recent task-specific vehicular DRL baseline, SLA-MA-PPO of Georgiades et al. (IEEE TNSM 2026, manuscript ref. [24]), reimplemented in `src/sla_mappo.py` for the slotted simulators (identical tasks via `gipa_core.Env`, heterogeneous tasks via `hetero.HetEnv`, with or without carried-over local work). `experiments/exp_sla.py` tunes (learning rate × deadline-penalty weight on disjoint seeds), trains five runs per fleet size and setting with checkpoint selection on the tuning seeds, and merges into `results/res_sla.json`; `experiments/compile_sla.py` pairs it with the archived planned policies on the 30 evaluation seeds (`results/sla_summary.{json,md}`); `experiments/diag_sla.py` records load spreading (`results/sla_diagnostics.json`); `run_e16.sh` runs every stage and `verify_e16.sh` retrains one run and recompiles. `experiments/exp_sla.py --outdir` writes sensitivity runs (`results/sla_long/`). No existing file changes behaviour.
- E13 (workplan, 8 Nov 2026): the slotted experiments rerun with carried-over local work. `GIPA_LOCAL_CARRY=own|blind` in `src/gipa_core.py` makes every slotted simulator carry unfinished local work into the next slot (backlog W' = max(W + L 1{local} - 20, 0)); in `own` mode each vehicle also uses its own backlog (mask (8), local score, and a redraw among the plan's RSUs for planned policies). New `src/carry.py` (stationary local-queue chain, sampling share under redirects, carry-aware fixed-point planning with a redirect floor) wraps the affine, Eq. (5) and asymmetric planners; `run_carry.sh` reruns every slotted experiment except MAPPO into `results/carry_own/` and `results/carry_blind/`; `experiments/check_carry.py` checks the chain against direct simulation (`results/carry_check.json`); `experiments/compile_carry.py` writes `results/carry_summary.{json,md}`; `verify_e13.sh` reruns one mode and compares. With the variable unset every simulator reproduces the archived results exactly.
- `experiments/exp_prediction_obs.py`, `results/res_prediction_obs.json` and `verify_e1b.sh` (workplan E1b, 8 Nov 2026): EWMA observation models (`obs`: uncongested breakdown, realised latency, realised latency minus the prior's congestion), planned policies that read per-task predictions (`plan_mask`, `gipa_plan`), a margin-load mask (`mask_z`) and Stage 0 latency-model error (`R0`) in `src/prediction.py`. Defaults reproduce E1 exactly.
- `experiments/exp_async_local.py`, `results/res_async_local.json`, `verify_e5b.sh` and Fig. R5b (workplan E5b, 8 Nov 2026): asynchronous arrivals with a persistent on-board processor per vehicle (FIFO; processor sharing as sensitivity). `src/async_sim.py` gains `FIFOQueue`, the M/D/1 waiting-time distribution, a closed-form local admission model, own-queue-aware policies (`+own`) and local-queue-aware planner modes. The default `local='none'` reproduces E5 exactly.
- `docs/SCRIPTS.md`: a catalogue of every script, with inputs, outputs, seeds, runtimes, manuscript mapping, known issues, missing scripts and a verification log.
- `src/planner.py`: deadline-constrained system-optimal planned allocation (exploratory, for the first revision).
- `experiments/exp_planner.py` and `results/res_planner.json`: planner versus GIPA and all-local in both regimes and four densities, plus a fleet-size misspecification sweep.
- `src/capacity.py`, `experiments/exp_capacity.py` and `results/res_capacity.json` (workplan E3): RSU deadline capacities, selected versus admitted load, and one admission-control layer applied to every policy. Soft mode reproduces `gipa_core.run` exactly.
- `src/prediction.py`, `experiments/exp_prediction.py` and `results/res_prediction.json` (workplan E1): predicted instead of oracle latencies, with score, mask and prior error arms, a drifting-truth model with an EWMA predictor, and a guard band. Zero error reproduces `gipa_core.run` exactly.
- `src/calibration.py`, `experiments/exp_calibration.py` and `results/res_calibration.json` (workplan E2): physical calibration, the Eq. (5) processor-sharing model with class-dependent compute, a class-aware prior and an Eq. (5) planner. Class-independent compute reproduces the affine model to 3 × 10⁻¹⁴.
- `src/mappo.py`, `experiments/exp_mappo.py` and `train_mappo.sh` (workplan E7): MAPPO baseline in pure NumPy with hand-written gradients (checked against finite differences), staged bench, tune, train and merge, and two-process training for a 2-core laptop.
- `src/qlearning.py`, `experiments/exp_qlearning.py` and `results/res_qlearning.json` (workplan E4): Q-learning with 10 training runs per configuration, disjoint training environments, episode reset, optional discount, both latency models, and a two-level bootstrap. Reproduces the archived learner exactly in its original configuration.
- `src/learning_automata.py`, `experiments/exp_la.py` and `results/res_la.json` (workplan E6): per-vehicle learning automata (L_R-I, L_R-εP, pursuit) with binary and S-model responses, tuned on disjoint seeds, evaluated with warm-up 25 and 500 against GIPA, the planner and all-local, plus a fleet-size change test (N = 80 → 240 → 80).
- `experiments/exp_complexity.py` and `results/res_complexity.json` (workplan E8): asymptotic complexity, measured per-slot and per-vehicle decision time for every policy, a scalar GIPA checked against the operator, Stage 0 / planner / predictor cost, training cost and runtime communication payload.
- `src/hetero.py`, `experiments/exp_hetero.py`, `results/res_hetero.json`, `verify_e12.sh` and Fig. R9 (workplan E12, decision D5): heterogeneous tasks, uneven links and partial coverage; GIPA operating on the plan (Framing A) against the planner with per-task ranking (Framing B). The homogeneous case reproduces E2 exactly.
- `plotting/revision_figures.py`, `figures/revision/` (Figs. R1–R8), `results/res_revision_figures.json` and `docs/FIGURES_REVISION.md`: one summary figure per reviewer concern, with seed-level bootstrap CIs, a fixed CVD-validated colour per method, and draft captions.
- `experiments/seed_level_main.py`: rebuilt the missing instrumented principal run; it regenerates `results/res_main_seed_level.json` byte for byte.
- `src/stats.py`, `experiments/stats_main.py` and `verify_stats.sh`: rebuilt the missing statistics script. It regenerates every value of `results/res_main_statistics.json` (bootstrap CIs with the shared stream, paired comparisons, TOST, Clopper–Pearson bounds), and is expected to be byte-identical under the pinned NumPy/SciPy.
- `src/async_sim.py`, `experiments/exp_async.py`, `results/res_async.json` and `verify_e5.sh` (workplan E5): event-driven simulator with Poisson arrivals and persistent processor-sharing RSU queues (unfinished work carried forward), static, broadcast and feedback policies, an asynchronous planner, admission control and queue-stability metrics. It also rebuilds the missing Table III processor-sharing check (`check_table3`).
- `src/asymmetric.py`, `experiments/exp_asym.py`, `results/res_asym.json` and `verify_e9.sh` (workplan E9): asymmetric RSUs (one 30% faster) under both latency models, uniform and capacity-recalibrated priors, an asymmetric SLSQP planner and herding metrics (fast-RSU herding ratio, load CV, same-choice probability, load swing, overload). The symmetric case reproduces archived and E2 values exactly.
- `experiments/compare_results.py`: generic number-by-number comparison of two JSON result files (scale-aware tolerance).
- MAPPO deadline-only mask (`--mask deadline`), `train_mappo_highdensity.sh` and `experiments/compare_mappo.py`: a fair MAPPO comparison at N ≥ 240 and a cross-machine verification. The default mask reproduces earlier results exactly.
- `results/res_mappo.json` and `results/train_mappo_log.txt`: the full E7 run on the i5-4210M laptop (27 Oct).
- `src/planner.py`: optional `xi` argument for a misestimated congestion slope (default behaviour unchanged).
- Restored files present on GitHub but missing from the downloaded archive: `docs/statistical_analysis.md`, `docs/GIPA_Appendices_A_to_H_FINAL.md`, `figures/algorithm1_gipa_decision_operator.{tex,pdf}`, `figures/fig_1.png`, `figures/fig_2.png`, `.gitattributes`.

### Changed
- Figures (29 Oct): all figure wording (legend entries, panel titles, axis labels, notes) moved into an `EDIT HERE` block at the top of `plotting/revision_figures.py` and `plotting/gipa_framework.py`, with a `--titles full|letter|none` switch. Plotted values and images are unchanged except for wording: Fig. R5 footer and Fig. R6 legend now use the GIPA naming, and the Fig. R8 machine footer wraps onto two lines. Every figure is also written as a 600 dpi LZW TIFF; new `plotting/export_submission_tiffs.py` writes `figures/submission/Figure_1.tif` … `Figure_11.tif` in manuscript order with a size and width check (`MANIFEST.txt`).
- Figures (30 Oct): new `plotting/concept_figures.py` draws seven explanatory figures C1–C7 into `figures/concept/` (PNG 300 dpi, PDF, SVG, TIFF 600 dpi), with draft captions and suggested placement in `docs/FIGURES_CONCEPT.md`. C1 is a redraw of the system architecture (submitted Fig. 1) showing processor sharing.
- Figures (30 Oct, later): C2 redrawn to show the three variants of GIPA (heuristic prior, planned prior, planned prior with per-task ranking) step by step on one example task. New `figures/algorithm1_gipa_slow_timescale` (Stage 0 and Stage 2) and `figures/algorithm2_gipa_stage1` (Stage 1) cover every variant.
- E8 timings (29 Oct): `results/res_complexity.json` is now the median of five single-threaded runs on the reference machine (`run_complexity_repeats.sh`, `experiments/aggregate_complexity.py`), with the run-to-run range and the full computing environment (`src/sysinfo.py`). Fig. R8 shows the machine details. The earlier single run is superseded.
- Run logs renamed to `results/*_log.txt`, and the reference-machine MAPPO file to `results/res_mappo_dmask_reference.json`; scripts, logs and documentation updated to the new names.
- `src/learning_automata.py`: `feasible_set` reads intrinsic costs from `env.q`, so it supports per-RSU speeds (E9). Results unchanged.
- `src/gipa_core.py`, `experiments/exp.py`: full docstrings and comments (units, manuscript equation references, Q-learning specification, known issues). Behaviour unchanged; the code is AST-identical to v1.0.0.
- `plotting/generate_figures.py`: reformatted from one-line statements into readable code with docstrings. Behaviour unchanged (AST-identical); regenerated figures are pixel-identical.
- `run_all.sh`: also runs the planner, capacity, prediction, calibration, Q-learning, learning-automata, complexity, asymmetric-RSU, asynchronous-arrival and heterogeneous-task experiments.
- `README.md`: updated layout and pointers to the catalogue.
- `SHA256SUMS.txt`: regenerated for this working copy.

### Verified
- MAPPO at N = 240 and 320 (deadline-only mask): independent runs on the i5-4210M laptop and on the reference machine agree within 0.010 ms on group means (28 Oct); `results/res_mappo.json` replaced by the laptop's complete merged file.
- A full rerun of `experiments/exp.py` reproduces the archived `res_main.json` exactly (MD5 `b4b3c084c3b8c97562c6c58097b6b923`).

### Known gaps
- Several scripts behind reported results are not in the repository (Appendix G verification, live feedback, aggregate relaxation, Regime B split policy). See `docs/SCRIPTS.md`.

## 1.0.0 - 2026-08-05

- Consolidated the shared simulator and experiment driver.
- Archived 30 matched seed-level records and statistical summaries.
- Merged confidence-interval and spacious-layout plotting workflows.
- Added reproducibility metadata, checksums, and multi-format figures.


## Revision R1 (November 2026)

- Added E14 (independent DQN and VDN) and E15 (re-solved probabilistic plan) with run scripts `run_e14.sh` and `run_e15.sh`; E15 tuning follows the selection rule of Section VI-E.
- Added `docs/MANUSCRIPT_MAP.md`, mapping every figure and table of the revised paper to its script and result file.
