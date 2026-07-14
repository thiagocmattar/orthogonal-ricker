# Code Map

Use this page to find the smallest safe edit for a run or feature. Read
`docs/methods.md` before changing scientific behavior and `docs/paper_map.md`
before changing a paper figure. During active experiments, do not rename,
rewrite, or clean configs, results, checkpoints, figures, or reports that may be
in use.

## Execution Path

The installed `paper-exp` command and `python -m paper_exp.cli` both enter
`src/paper_exp/cli.py`. The selected handler loads a YAML config when needed,
writes numbered runs through `src/paper_exp/run.py`, and records provenance
through `src/paper_exp/utils.py`.

```text
CLI -> config loader -> workflow module -> run/artifact writers
                         |
                         +-> saved results -> plots.py -> paper figures
```

The CLI command selects the workflow; `evaluation.metric` records the intended
measurement but does not dispatch the command.

## Package Index

| Module | Owns | Look here when changing |
| --- | --- | --- |
| `cli.py` | Argument parsing and command dispatch | Command names, flags, or routing |
| `config.py` | Shared filename and minimum-field validation; random-initialization invariant | Cross-workflow config rules |
| `integrity.py` | Read-only checks for configs, run envelopes, document references, paper outputs, and figure numbering | Preflight and open-source release checks |
| `run.py` | Experiment/run directory naming, smoke/calibration lifecycle, and the common config/metrics/manifest/predictions envelope | Launch snapshots, terminal status, and artifacts required of completed runs |
| `utils.py` | JSON/JSONL helpers and environment, Git, GPU, package, and run provenance | Manifest contents and serialization |
| `data.py` | Dataset loading, tokenization, cache metadata, and cache reuse checks | `data`, `tokenizer`, and `preprocessing` behavior |
| `calibration.py` | Calibration and pretraining loops, validation, checkpoints, training events, and naive/orthogonal update routing | Optimizer or training-loop behavior |
| `modeling.py` | Runtime model architecture modifications and checkpoint reconstruction | Post-LayerNorm ReLU behavior; treat as scientific code |
| `activations.py` | Named activation sites, hooks, captured tensors, and post-hoc clipping | Where a site is measured or modified |
| `activation_pressure.py` | Pressure config parsing, L1/Ricker losses, gradient diagnostics, and Adam-step orthogonal correction | Pressure mathematics and metrics |
| `clipping.py` | Checkpoint-based clipping sweeps and logical projection-skip proxies | Post-hoc clipping evaluation |
| `activation_histograms.py` | Activation-distribution diagnostics across selected checkpoints | `activation_histograms` runs |
| `weight_histograms.py` | Checkpoint parameter-distribution diagnostics | `weight_histograms` runs |
| `activation_propagation.py` | Exact-zero propagation and logical zero-product accounting | `activation_propagation` runs |
| `sweeps.py` | The fixed-step pressure matrix, generated configs, and sequential sweep runners | Existing fixed-step sweep composition |
| `plots.py` | Shared paper style, procedural figure dispatch, result selection/loaders, and family renderers | Figure dependencies, labels, layouts, and output names |
| `eval.py` | Tiny harness-only prediction metrics | Smoke-test metrics |

Focused tests live in `tests/test_config.py`, `tests/test_activation_pressure.py`,
`tests/test_modeling.py`, `tests/test_activation_propagation.py`, and
`tests/test_smoke.py`. Repository and plot-selection contracts are covered by
`tests/test_integrity.py` and `tests/test_plot_selection.py`; launch/terminal
transitions are covered by `tests/test_run_lifecycle.py` and
`tests/test_calibration_lifecycle.py`.

## Run Lifecycle: First Tranche

`run.start_run` snapshots `config.yaml` and a `status: running` manifest before
smoke or calibration/pretrain work begins. `run.complete_run` writes metrics
and predictions first, then atomically publishes the terminal manifest with
`status: completed` and `finished_at`. `run.run_lifecycle` records an escaping
exception as `status: failed` with `finished_at`, exception type, and message,
then re-raises it. A normal lifecycle exit without `complete_run` is also
rejected and recorded as failed. Terminal manifests are derived from the immutable launch
snapshot, so their Git commit and dirty state remain launch provenance even if
the working tree changes during a run.

This lifecycle currently applies only to `run.run_smoke` and
`calibration.run_calibration`, including CLI `calibrate` and `pretrain` calls.
Data preparation, activation and weight histograms, activation propagation,
and clipping still use `create_run_dir`, a late `build_manifest`, and
`write_run_artifacts`. They remain legacy workflows until the second migration;
do not describe their directories as having explicit `running`, `completed`,
or `failed` status. Statusless historical runs remain supported when their core
artifact envelope is coherent.

For calibration/pretrain, `predictions.jsonl` currently duplicates the train
and validation event history in `events.jsonl`; it is not generated-token
output. `calibration/wall_seconds_train` includes validation performed inside
the timed training interval. Separate validation timing metrics expose that
component, and `calibration/wall_seconds_total` additionally includes final
checkpoint work but not the final artifact writes.

## CLI and Artifact Index

Commands that accept `--config` first use `config.load_config`; `clip-sweep`
instead loads the config and manifest saved by its checkpoint run. Smoke and
calibration/pretrain use the explicit lifecycle above. The remaining
run-producing workflows still write `config.yaml`, `metrics.json`,
`manifest.json`, and `predictions.jsonl` together at successful completion via
the legacy `run.write_run_artifacts`.

| CLI command | Workflow entry point | Workflow-owned config sections | Additional or primary artifacts |
| --- | --- | --- | --- |
| `smoke` | `run.run_smoke` | Shared required fields | Lifecycle launch snapshot, then common completed envelope with canned smoke predictions |
| `baseline` | `run.run_baseline` | Shared required fields | Currently stops at the explicit budget `TODO`; no baseline run is implemented |
| `prepare-data` | `data.prepare_tokenized_data` | `data`, `tokenizer`, `preprocessing`, optional `validation` | `tokens.int32.bin` and `metadata.json` under the token cache; cache paths are recorded in the run |
| `calibrate` | `calibration.run_calibration` | `model`, `data`, `preprocessing`, `training`, `validation`, `checkpoint`, optional `activation_pressure` | Lifecycle launch snapshot; `events.jsonl`; optional `checkpoints/final/`; terminal manifest last |
| `pretrain` | `calibration.run_calibration(..., mode="pretrain")` | Same as `calibrate` | Same lifecycle as `calibrate`; `predictions.jsonl` currently contains event history |
| `clip-sweep` | `clipping.run_clipping_sweep` | Saved source-run config plus `activation_clipping`; thresholds/sites are normally CLI arguments | `clipping_frontier.jsonl` and the common envelope |
| `activation-histograms` | `activation_histograms.run_activation_histograms` | `activation_histograms`, `validation`, and cache/model fields | `activation_histograms.json` and the common envelope |
| `weight-histograms` | `weight_histograms.run_weight_histograms` | `weight_histograms` and source-run references | `weight_histograms.json` and the common envelope |
| `activation-propagation` | `activation_propagation.run_activation_propagation` | `activation_propagation`, `validation`, and cache/model fields | `activation_propagation.json` and the common envelope |
| `write-pressure-sweep-configs` | `sweeps.write_pressure_fixed_step_configs` | Specs defined in `sweeps.py` | Numbered YAML configs |
| `run-pressure-sweep` | `sweeps.run_pressure_fixed_step_sweep` | Generated training configs | Standard pretraining run artifacts for each selected config |
| `run-pressure-sweep-clipping` | `sweeps.run_pressure_fixed_step_clipping_sweeps` | Completed sweep configs plus CLI clipping arguments | Standard clipping-sweep artifacts |
| `plots` | `plots.generate_plots` | No live config; consumes saved result artifacts and the dispatch in `plots.py` | Numbered PDFs and optional PNGs |
| `check` | `integrity.check_repository` | No config; reads repository indexes and artifact envelopes | Findings only; never writes repository files |
| `plot-run` | `plots.generate_run_diagnostics` | No live config; consumes one run | Requested PDF and optional PNG |
| `plot-clipping-frontier` | `plots.generate_clipping_frontier` | No live config; consumes one clipping run | Requested PDF and optional PNG |

See `configs/README.md` for field ownership and starting examples.

## Surgical Change Recipes

### Add or change a pressure method

1. Write the mathematical and optimization semantics in `docs/methods.md`.
2. Add config parsing, the method identifier, and pressure calculation in
   `activation_pressure.py`. Keep naive loss augmentation and Adam-step
   orthogonal correction as different methods.
3. Touch `calibration.py` only where the training step must route or log the new
   behavior. Preserve task loss, pressure loss, and update diagnostics as
   distinct metrics.
4. Add focused numerical tests in `tests/test_activation_pressure.py`, then add
   a numbered config copied from the closest method/run shape.
5. Record the run and its interpretation boundary in the experiment log; add it
   to the paper map only if it supports a paper artifact.

### Add an activation site

1. Define the tensor precisely: module path, pre/post operation, shape, and
   downstream operator.
2. Add the alias and one registration method in `activations.py`; ensure hooks
   are removed by `ActivationCapture` and clipping replaces the intended tensor.
3. Add capture and clipping tests in `tests/test_activation_pressure.py` using a
   small Pythia-like module.
4. If the site changes zero-product accounting, update
   `activation_propagation.py` and `tests/test_activation_propagation.py`
   separately.
5. Use explicit `activation_pressure.sites` or diagnostic `sites` in the new
   config. Do not silently broaden older configs.

### Add a diagnostic

1. First decide whether the output belongs in an existing diagnostic JSON. If
   not, add one focused workflow module with a single `run_*` entry point.
2. Give it a workflow-owned config section and a numbered config; add CLI
   parsing/routing in `cli.py`.
3. Existing diagnostic workflows still use `create_run_dir`, `build_manifest`,
   and `write_run_artifacts`, then write specialized JSON/JSONL next to the
   common envelope. Do not migrate one diagnostic ad hoc: the second lifecycle
   tranche must also make source-run status checks and specialized-artifact
   ordering consistent.
4. Select only completed source runs and record their config/run identifiers in
   the payload or manifest. Add a CPU-sized test for the schema or calculation.
5. Document what is measured, its denominator/sample size, and its limits in
   `docs/methods.md` and `docs/experiment_log.md`.

### Add a paper figure

1. Add a row to `docs/paper_map.md` with purpose, exact configs, exact saved
   results, and the numbered output filename.
2. In `plots.py`, keep the family together: dependency constants, selection in
   `_generate_known_paper_figures`, an export wrapper, data loader, and renderer.
   Shared colors, typography, axes, and output behavior remain centralized;
   family-specific loaders/renderers may move to a focused module when that
   makes a surgical edit easier.
3. Read only saved artifacts. Do not place training or measurement logic inside
   a renderer, and do not silently substitute a different experiment.
4. Generate into a temporary comparison location first. Check inputs, series,
   labels, axes, sample size/uncertainty, layout, PDF, and optional PNG before
   replacing a paper artifact.
5. Treat Report 04 and figures `79` through `90` as the current visual baseline:
   `report/04-2026-07-11-post-layernorm-relu-ol1-comparison/`.

## Architecture Changes

Architecture edits are scientific interventions, not harness cleanup. Their
current path spans model config fields, `calibration._apply_model_architecture_overrides`,
`modeling.py`, activation hooks, checkpoint loading, tests, methods, and run
documentation. Scope these changes separately, preserve random initialization,
and verify that saved checkpoints reconstruct the same architecture. Never fold
an architecture change into a plotting or navigation refactor.
