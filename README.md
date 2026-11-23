# GIPA Reproducibility Repository v1.10

Code and archived results supporting **Planned Priors and Per-Task Predictive Screening for Deadline-Constrained Vehicular Edge Offloading**, manuscript ITS-2026-08-0579 R1.

Repository: [mail2chuka/GIPA-Repository-v1.10](https://github.com/mail2chuka/GIPA-Repository-v1.10).

GIPA combines a planned class-level offloading distribution, per-task predictive screening and independent decisions. The implementation is classical. The package covers affine and processor-sharing models, prediction error, unequal RSUs, task variation and persistent queues, together with learning and probabilistic-planning comparisons.

## Contents

- `src/`: simulators, planning, probability refinement, queue-aware rules and baseline implementations.
- `experiments/` and the root-level shell scripts: experiment drivers and result comparisons.
- `results/`: archived outputs, per-seed records, training runs and statistical summaries. Logs, `pre_fix/` and `old_tune/` preserve provenance; use the current result paths identified in the manuscript map.
- `plotting/` and `figures/`: figure generators and archived images. `figures/manuscript/` contains frozen current manuscript diagram assets, including SVG where supplied.
- [`docs/MANUSCRIPT_MAP.md`](docs/MANUSCRIPT_MAP.md): links manuscript results to code and data. [`docs/SCRIPTS.md`](docs/SCRIPTS.md): detailed commands, seeds, historical notes and known gaps.

## Install

The shell commands below use Bash, such as Linux or WSL. Run them from the repository root.

```bash
git clone https://github.com/mail2chuka/GIPA-Repository-v1.10.git
cd GIPA-Repository-v1.10
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

`requirements.txt` pins NumPy, SciPy and Matplotlib. [`ENVIRONMENT.md`](ENVIRONMENT.md) records the original and revision environments; these are different environments. Timing results depend on the machine. Python commands also work from PowerShell after setting `$env:PYTHONPATH = (Resolve-Path .\src).Path`; the `.sh` runners require Bash.

## Quick verification

```bash
python scripts/verify_archive.py
python experiments/stats_main.py --check
# Optional: also rerun the principal seed-level simulation and its statistics.
bash verify_stats.sh
```

The first command checks the archive manifest. The second rebuilds statistical summaries from the archived seed-level data into `results/res_main_statistics_rebuilt.json` and compares them with the archived summary. `verify_stats.sh` additionally writes `results/res_main_seed_level_rebuilt.json`; it does not retrain the learned policies. Inspect the printed byte and numerical comparisons: different library versions can produce final-digit differences. These checks do not establish that the full experiment suite has been rerun.

To rerun a focused planning experiment while keeping its archived output:

```bash
python experiments/exp_planner.py --out results/reproduction/res_planner.json
python experiments/compare_results.py results/res_planner.json results/reproduction/res_planner.json
```

## Reproduce selected experiments

Experiment labels follow the script catalogue. Long-running commands below can overwrite generated outputs; use a working copy or supported `--out`/`--outdir` options when retaining the archive.

| Experiment | Entry point from the repository root |
| --- | --- |
| Main affine batches | `python experiments/exp.py` (writes `res_main.json` and `res_sc.json` in the current directory) |
| E1 / E1b: prediction and observed-latency models | `python experiments/exp_prediction.py`; `python experiments/exp_prediction_obs.py` |
| E2: class-dependent processor sharing | `python experiments/exp_calibration.py` |
| E3: capacity and admission | `python experiments/exp_capacity.py` |
| E4: tabular Q-learning | `python experiments/exp_qlearning.py` |
| E5 / E5b: asynchronous arrivals and local queues | `python experiments/exp_async.py`; `python experiments/exp_async_local.py` |
| E6: learning automata | `python experiments/exp_la.py` |
| E7: MAPPO | `bash train_mappo.sh`, then `bash train_mappo_highdensity.sh` |
| E8: repeated decision timings | `bash run_complexity_repeats.sh` |
| E9: unequal RSUs | `python experiments/exp_asym.py` |
| E12: heterogeneous tasks | `python experiments/exp_hetero.py` |
| E13: work carried between slots | `bash run_carry.sh own`; `bash run_carry.sh blind`; `python experiments/compile_carry.py` |
| E14: independent DQN and VDN | `bash run_e14.sh` |
| E15: plan recomputed each slot | `bash run_e15.sh` |
| E16: SLA-MA-PPO with idealised feedback, a sensitivity comparison | `GIPA_SLA_FEEDBACK=ideal bash run_e16.sh` |
| E17: backlog-priority screening | `bash run_e17.sh` |
| E18: queue priority with the original offload targets | `bash run_e18.sh` |
| E19: SLA-MA-PPO with completion-gated feedback, the main manuscript comparison | `bash run_e19.sh` |

`run_all.sh` runs the main batch and several early revision experiments and redraws figures. It does **not** run the whole package: MAPPO training and E13–E19 have separate entry points. The archive contains five independent training runs per fleet size and setting for the principal multi-agent learners and ten for tabular Q-learning; evaluations use 30 matched seeds. Training, tuning and feedback protocols are described in the catalogue and source.

E19 uses latency feedback after completion and deadline feedback after completion or deadline expiry. E16 gives outcomes at the next decision, including unfinished tasks. Existing SLA run files are skipped by the training runners. For an actual fresh, single-run gated comparison in a separate output directory:

```bash
GIPA_SLA_FEEDBACK=gated python experiments/exp_sla.py --stage train --setting homog --densities 160 --seeds 0 --outdir results/reproduction_sla_gated
```

This is one training run, not the complete five-run, four-setting, four-density comparison. The reported GIPA comparisons use one variant fixed for each setting; they do not imply that every GIPA variant is superior under every workload.

## Figures and attribution

`python plotting/revision_figures.py --figure 1` redraws the main latency comparison from archived results; other figure selections are documented in the manuscript map. Frozen diagram assets can be viewed without rerunning experiments. The legacy `plotting/export_submission_tiffs.py` utility requires a clean manuscript DOCX as its first argument and the optional `python-docx` package; it is separate from the experiment runners.

The SLA-MA-PPO implementation adapts the vehicular offloading method cited in `src/sla_mappo.py`. Its action space, observations, reward settings and feedback timing are adapted to this simulator; it is not the original authors' software. Baseline attributions are retained in the source files and documentation.

See [`CITATION.cff`](CITATION.cff) for software citation metadata and [`LICENSE`](LICENSE) for the supplied licence. Experiment code and archived results are unchanged from the source package. Historical records and known gaps are described in `docs/SCRIPTS.md`.
