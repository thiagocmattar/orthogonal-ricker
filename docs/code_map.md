# Code Map

Use this map to locate the smallest owning module. Read `methods.md` before
changing scientific behavior, `diagnostics.md` before changing measurements,
and `plotting.md` before changing figure behavior.

## Execution Path

The installed `paper-exp` command and `python -m paper_exp.cli` enter
`src/paper_exp/cli.py`.

```text
config -> CLI workflow -> model/data/method implementation -> run artifacts
                                                             |
saved pinned artifact -> loader -> pure reduction -> renderer -> PDF/PNG
```

The CLI command selects the workflow. A descriptive config field such as
`evaluation.metric` does not dispatch a command.

## Module Ownership

| Module | Owns |
| --- | --- |
| `cli.py` | Public command names, arguments, validation errors, and dispatch |
| `config.py` | Shared config validation, random-initialization invariant, architecture/gate contracts, and learned-threshold requirements |
| `data.py` | Dataset loading, tokenization, cache metadata, and compatibility checks |
| `reproducibility.py` | Deterministic training schedules and document-disjoint validation partitions/hashes |
| `calibration.py` | Calibration/pretraining loop, optimizer groups, validation, checkpoints, event logging, and method routing |
| `modeling.py` | Pythia architecture modifications, fixed and learned gates, threshold ownership, metrics, and checkpoint reconstruction |
| `activations.py` | Stable activation-site aliases, exact hook locations, tensor metadata, and clipping hooks |
| `activation_pressure.py` | Pressure parsing, Ricker/L1 scalars, raw-gradient diagnostics, and post-AdamW orthogonal correction |
| `activation_histograms.py` | Streamed activation distributions and exact/near-zero threshold counters |
| `weight_histograms.py` | Checkpoint parameter distributions |
| `activation_propagation.py` | Exact-zero propagation and actual-operand logical zero-product counting |
| `clipping.py` | Saved-checkpoint threshold, quantile, and RMS clipping sweeps |
| `run.py` | Run IDs, config snapshots, lifecycle transitions, and common artifact envelope |
| `runner.py` | Clean-tree, locked, fail-stop sequential execution, atomic state, logs, progress, and ETC |
| `integrity.py` | Read-only repository, config, run-envelope, and document-reference checks |
| `utils.py` | JSON/JSONL helpers and environment, Git, GPU, package, and run provenance |
| `plots.py` | Explicit single-run artifact loaders, reducers, renderers, and plot-kind dispatch |
| `plot_common.py` | Presentation-neutral numerical helpers shared by plot families |
| `plot_style.py` | Colorblind-safe palette and repository-wide presentation defaults |
| `plot_api.py` | Count-derived layouts, publication-profile checks, one-build PDF/PNG export, and atomic output promotion |

## CLI and Artifacts

| Command | Workflow | Primary outputs |
| --- | --- | --- |
| `smoke` | Tiny infrastructure check from `configs/00-smoke.yaml` | Common run envelope |
| `prepare-data` | Download/tokenize configured data | Token cache and metadata; preparation run record |
| `calibrate` | Short configured training/throughput run | Events, metrics, optional checkpoint, common envelope |
| `pretrain` | One random-initialized pretraining config | Events, metrics, predictions, checkpoint when configured, common envelope |
| `run-configs` | Ordered pretraining configs through one child at a time | Atomic runner state, child logs, verified child run envelopes |
| `run-status` | Read-only runner-state/artifact inspection | Human-readable progress and ETC on stdout |
| `clip-sweep` | Post-hoc clipping from one saved checkpoint | `clipping_frontier.jsonl` and common envelope |
| `activation-histograms` | Activation distribution diagnostic | `activation_histograms.json` and common envelope |
| `weight-histograms` | Parameter distribution diagnostic | `weight_histograms.json` and common envelope |
| `activation-propagation` | Exact-zero and logical-product diagnostic | `activation_propagation.json` and common envelope |
| `plot` | Render one explicitly named saved artifact | PDF and optional PNG |
| `check` | Read-only integrity scan | Findings on stdout |

See `configs/README.md` for config ownership and `results/README.md` for
lifecycle details.

## Change Routes

### Pressure Method

1. Specify the scalar, averaging, optimizer timing, and limitations in
   `methods.md`.
2. Add parsing and numerical behavior in `activation_pressure.py`.
3. Touch `calibration.py` only where routing or event logging changes.
4. Add focused numerical tests for signs, projection condition, caps, zero
   norms, and device/dtype behavior.
5. Add plan-authorized configs only after the definitive plan is reviewed.

Keep naive loss augmentation and post-AdamW orthogonal correction as different
method identifiers.

### Activation Site

1. Define module path, pre/post-operation location, shape, and downstream
   operator in `methods.md`.
2. Add one stable alias and hook path in `activations.py`.
3. Test capture, hook removal, and clipping replacement.
4. Update `activation_propagation.py` separately if actual-operand accounting
   changes.
5. Require new configs to opt into the site explicitly.

### Architecture or Gate

Architecture edits span config validation, `modeling.py`, construction in
`calibration.py`, activation hooks, checkpoint reconstruction, tests, and
methods documentation. Treat the change as a scientific intervention and
verify an exact checkpoint round trip. Do not hide it in plotting or cleanup.

### Diagnostic

1. Define the estimand, integer counters, denominator, coverage, and nonclaims
   in `diagnostics.md`.
2. Reuse an existing focused diagnostic artifact when its schema fits; add a
   new workflow only when the measurement is genuinely different.
3. Pin source config/run/checkpoint identities.
4. Write specialized artifacts before terminal completion.
5. Add CPU-sized schema and numerical tests.

### Plot

1. Name the exact source artifacts and output in the reviewed plan and
   `paper_map.md`.
2. Load only explicit paths; do not discover the latest run.
3. Put scientific reduction in a small pure function with numerical tests.
4. Keep figure-specific labels, axes, cohorts, and rendering together.
5. Reuse `plot_style.py`, `plot_common.py`, and `plot_api.py` for shared
   presentation and export mechanics.
6. Stage, render, inspect, and atomically publish the complete figure set with
   deterministic provenance.

## Testing Boundaries

Scientific tests cover pressure mathematics, gates, activation capture,
propagation counters, clipping, reproducibility, and checkpoint reconstruction.
Infrastructure tests cover config validation, lifecycle ordering, runner
locking/fail-stop/progress behavior, integrity checks, plotting mechanics, and
smoke execution.

Run the smallest focused tests while iterating. Before handoff, run:

```bash
make test
make check
```
