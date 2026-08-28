# Code Map

This is the navigation authority for the implementation. Find the smallest
owning module here before editing code. Scientific definitions live in
[`methods.md`](methods.md); measurement contracts live in
[`diagnostics.md`](diagnostics.md); launch procedure lives in
[`runbook.md`](runbook.md).

## Execution Boundaries

```text
scaffold run/runner.py -> runner.py -> sibling configs -> training.py -> scaffold raw/
CLI ------------------> workflow module -------------------------------> scaffold raw/
pinned scaffold raw/ -> plots/dispatch.py -> family renderer -> scaffold figs/
```

- `cli.py` exposes single-workflow commands; it is not a scientific queue.
- Every definitive training tranche enters through
  `experiments/NN-<phase>-<tranche>/run/runner.py`, even when the tranche
  contains one config.
- `runner.py` is the only parent runner. It validates the whole tranche and
  holds one lock. Definitive pretraining is serial by default. A1's dormant
  bounded-worker metadata covers only the completed `001`–`008` launch shape;
  it does not cover pending configs `009`–`011` or authorize worker slots for
  them. Calibration uses the same worker engine but remains a separate
  non-evidence workflow.
- Diagnostics consume exact saved run/checkpoint identities. Plots consume
  saved artifacts only.

## Core Modules

Paths below are relative to `src/paper_exp/`.

| Module | Owns | Change it when |
| --- | --- | --- |
| `__init__.py` | Package version only | The release version changes |
| `cli.py` | Public commands, arguments, lightweight parsing, and dispatch | A workflow gains or changes a user-facing command |
| `config.py` | YAML loading; common, training, and diagnostic schema validation; random-initialization plus `model.topology_id`/`model.site_gate` contracts | A reviewed config field or invariant changes |
| `design.py` | Reviewed-design SHA/blob validation, catalog counts and aliases, canonical condition fingerprints, duplicate detection, and exact group materialization guards | A case group becomes materializable or the reviewed identity/reuse contract changes |
| `topology.py` | Canonical transformer-site aliases, exact site metadata, supported topology IDs, active-port sets, and site-gate schema | Site nomenclature, a reviewed topology, or the gate-field contract changes |
| `launch.py` | Repository/scaffold/config resolution, reviewed-plan and clean-Git gates, owned `raw/` output roots, and the exclusive lock | Launch-wide preflight policy changes |
| `runner.py` | Serial-default parent runner, scaffold/config validation, attempt-state resume preflight, completed-config reuse, config-bound parallel authorization, isolated two-GPU coordination, and bounded calibration coordination | Case-runner or calibration-coordinator behavior changes |
| `parallel.py` | Deterministic bounded admission, slot assignment, failure draining, and unadmitted-work accounting | Shared concurrent-coordinator semantics change |
| `hardware_profile.py` | Pure non-evidence physical-batch request, result, selection, and artifact contracts | The microbatch profiling grid or operational selection rule changes |
| `hardware_profile_run.py` | Restart-safe fresh-process hardware-profile coordination and durable attempt/artifact publication | Profile orchestration, retry, or provenance changes |
| `hardware_profile_worker.py` | CUDA-only random-Pythia/OL1 profiling workload and synchronized operational measurements | The production-shaped profiling workload or timing boundary changes |
| `infrastructure_smoke.py` | Two-worker failure, drain, explicit recovery, completed reuse, GPU isolation, and evidence report | The combined concurrency/GPU smoke contract changes |
| `run.py` | Run IDs, immutable config snapshot, running/completed/failed manifests, and atomic artifact writes | The common run envelope or lifecycle changes |
| `training.py` | Calibration/pretraining orchestration, evaluation, event logging, and final checkpoint publication | The end-to-end training workflow changes |
| `optimization.py` | AdamW construction, minibatch sampling, LR warmup, naive L1 steps, OL1 post-Adam correction, and norm metrics | Optimizer-step mathematics or step metrics change |
| `modeling.py` | Random Pythia construction, realization of canonical topologies and site-gate operators, and exact checkpoint reconstruction | Model construction or a reviewed architecture intervention changes |
| `activation_pressure.py` | Pressure config parsing, L1 objective, near-zero metrics, gradient diagnostics, and OL1 projection/correction math | L1 or OL1 semantics change |
| `activations.py` | Capture and clipping hooks for canonical sites plus exact-zero counts | Capture/replacement behavior changes without changing site nomenclature |
| `data.py` | Dataset/tokenizer loading, document handling, token-cache construction, metadata, and compatibility checks | Data preparation or cache identity changes |
| `reproducibility.py` | Deterministic training schedules and document-disjoint validation partitions and hashes | Sampling or partition contracts change |
| `integrity.py` | Read-only checks for scaffold shape, runner/config numbering, current or indexed `raw/` envelopes, indexed `figs/`, document references, and preservation of exact indexed completed historical configs outside active reviewed scope | A durable repository invariant changes |
| `utils.py` | Small JSON/JSONL and environment/Git/GPU/package provenance helpers | A cross-workflow serialization or provenance primitive changes |

The scientific entry points are tested primarily in
`test_activation_pressure.py`, `test_modeling.py`,
`test_activation_propagation.py`, and `test_data_partition_contract.py`.
Launch and artifact behavior is covered by `test_config.py`, `test_launch.py`,
`test_runner.py`, `test_run_lifecycle.py`, `test_training_lifecycle.py`, and
`test_integrity.py`.

## Diagnostic Modules

All paths in this table are under `src/paper_exp/diagnostics/`.

| Module | Owns | Change it when |
| --- | --- | --- |
| `__init__.py` | Diagnostic package boundary; no workflow logic | A deliberately public diagnostic export is required |
| `sources.py` | Exact source-run selection, completed-checkpoint verification, portable source paths, and shared validation-cache identity | Any diagnostic source-pinning rule changes |
| `evaluation.py` | Shared evaluation starts, device/dtype/autocast selection, and peak-memory readings | Evaluation mechanics shared by diagnostics change |
| `activation_histograms.py` | Activation-distribution measurement across pinned runs | The activation histogram artifact or measurement changes |
| `weight_histograms.py` | Named checkpoint-parameter distribution measurement | Weight scopes or histogram artifacts change |
| `logical_products.py` | Integer logical-product opportunity counts for linear, QK, and PV products and block/model denominator reduction | Product-count mathematics changes |
| `propagation_capture.py` | Hooks and eager-attention capture for per-layer activation and actual-operand observations | A propagation observation point changes |
| `propagation_summary.py` | Stage vocabulary, architecture metadata, pooled denominators, and endpoint summaries | Propagation schema or reduction changes |
| `propagation.py` | Source orchestration and publication of `activation_propagation.json` | The propagation workflow changes, not its low-level counters |
| `clipping_evaluation.py` | One clipped validation evaluation and optional logical-product capture | Clipped-forward measurement changes |
| `clipping.py` | Generic single-checkpoint clipping sweep and its source envelope | The reusable single-source clipping workflow or artifact changes |
| `clipping_frontier.py` | Config-driven multi-checkpoint cohort lifecycle, exact source/reference validation, row arithmetic, and `clipping_frontier.jsonl` publication | The reviewed config-020 cohort or artifact contract changes |

Diagnostic contracts are exercised mainly by
`test_activation_propagation.py`, `test_activation_pressure.py`,
`test_auxiliary_lifecycle.py`, `test_clipping_frontier.py`, and
`test_config.py`. Define the estimand,
integer numerator, denominator, coverage, and nonclaims in
[`diagnostics.md`](diagnostics.md) before changing a diagnostic schema.

## Plot Modules

All paths in this table are under `src/paper_exp/plots/`.

| Module | Owns | Change it when |
| --- | --- | --- |
| `__init__.py` | Small public plotting API | A stable plotting export changes |
| `dispatch.py` | Plot-kind registry, explicit artifact loading, dispatch, hashes, and provenance sidecars | A plot kind or its input contract changes |
| `style.py` | Colorblind-safe series identities and repository-wide paper style | Shared visual presentation changes |
| `export.py` | Panel layouts, publication checks, one-build PDF/PNG export, staging, and atomic publication | Mechanical layout/export behavior changes |
| `histograms.py` | Presentation-neutral histogram validation, pooling, and density reductions | Histogram mathematics shared by plot families changes |
| `a1_lr_screen.py` | Exact eleven-run A1 cohort, fixed-horizon LR selection, training-progress panels, and deterministic suite provenance | The reviewed A1 evidence pins, selection presentation, progress rendering, or appendix output changes |
| `a2_spillover.py` | Exact six-run A2 cohort plus diagnostic `018`, count-first site/attention reductions, exact layerwise and within-site pooled densities, and deterministic three-figure provenance | The accepted A2 evidence pins, spillover plane, distribution atlases, density reduction, or claim boundary changes |
| `a2_clipping.py` | Exact diagnostic-020 A2 cohort, quality–`R_model` reductions, two-panel frontier, complete table, and deterministic provenance | The clipping-evidence pins, frontier reduction, presentation, or claim boundary changes |
| `run_diagnostics.py` | Training-run summary figure | That figure's cohorts, labels, axes, or rendering change |
| `activation_histograms.py` | Activation histogram figure | That figure family changes |
| `weight_histograms.py` | Weight histogram figure | That figure family changes |
| `propagation.py` | Activation-propagation figure | That figure family changes |
| `clipping.py` | Clipping-frontier figure | That figure family changes |

Plot tests live in `test_plot_api.py`, `test_plots.py`, and
`test_a2_clipping.py`. Keep loading and
pinning, pure numerical reduction, rendering, and export as separate
boundaries. See [`plotting.md`](plotting.md) before changing figure behavior.

## Scientific Method Catalog

These are the only implemented pressure methods and architecture gates. Names
in this table are exact config/API identifiers where shown.

| Method | Definition and boundary | Implementation owner |
| --- | --- | --- |
| `none` | Monitor configured activation sites without adding a pressure objective | `activation_pressure.py`, routed by `optimization.py` |
| `l1_naive` (L1) | Add the mean absolute captured activation objective directly to task loss | `activation_pressure.py`, executed by `optimization.py` |
| `orthogonal_l1` (OL1) | Take a task-only AdamW step, remove only a conflicting component from the preconditioned L1 direction, cap it with `step_budget`, then apply the correction | `activation_pressure.py`, executed by `optimization.py` |
| Canonical topology | Select one of the 11 active gate-port sets from `A0` through `A6-POST`; the ID does not select the operator, optimizer, or pressure | `topology.py`, realized by `modeling.py` |
| ReLU | Stock GELU remains at `h` for `A0`; `site_gate.operator: relu` explicitly applies ReLU at every active port of a non-`A0` topology | `topology.py`, `modeling.py` |
| Fixed one-sided gate `G+` | Preserve `x >= kappa`; replace smaller values with exact zero | `modeling.py`; captured/replaced through `activations.py` |
| Fixed symmetric gate `G±` | Preserve `abs(x) >= kappa`; replace smaller magnitudes with exact zero | `modeling.py`; captured/replaced through `activations.py` |
| Post-hoc clipping | At evaluation only, zero selected activations by absolute threshold, quantile, or RMS multiplier | `activations.py`, `diagnostics/clipping*.py` |

L1 and OL1 are different optimizer interventions and must remain different
identifiers in configs, metrics, plots, and prose. Fixed gates are architecture
interventions, not pressure methods. Clipping is an inference-time diagnostic,
not training. Exact formulas and interpretation limits are authoritative in
[`methods.md`](methods.md).

Retired pressure and adaptive-gate families are not part of this workflow. Do
not reintroduce them as compatibility paths. A future method requires an
explicitly reviewed plan change, a methods-contract update, focused numerical
tests, and new immutable configs.

## Change Routes

| Intended change | Read first | Primary owner | Also verify |
| --- | --- | --- | --- |
| L1/OL1 math or timing | `methods.md` | `activation_pressure.py`, `optimization.py` | Config parsing, metrics, focused numerical tests |
| Transformer site or topology nomenclature | `methods.md`, `diagnostics.md` | `topology.py` | Config validation, model realization, capture, diagnostics, tests |
| Capture or clip a canonical site | `methods.md`, `diagnostics.md` | `activations.py` | Model path, diagnostic capture, clipping, tests |
| Model or fixed gate | `methods.md` | `modeling.py` | Config validation, training construction, checkpoint round trip, diagnostics |
| Data or partitioning | `experiment_plan.md`, `methods.md` | `data.py`, `reproducibility.py` | Cache identity, manifests, plan-authorized configs |
| Diagnostic | `diagnostics.md` | One focused `diagnostics/` workflow | Source pinning, schema, lifecycle, plot consumer |
| Figure | `experiment_plan.md`, `paper_map.md`, `plotting.md` | One focused `plots/` renderer | Pure reduction tests, style, provenance, both output formats |
| Run artifact lifecycle | `../experiments/README.md`, `runbook.md` | `run.py` | Training/diagnostic callers, integrity checks |
| Launch sequencing | `experiment_plan.md`, `runbook.md` | Thin `experiments/NN-*/run/runner.py`; shared behavior in `runner.py` | Sibling configs, scaffold ownership, numeric order, ETC procedure |

Use the smallest focused tests while editing. Before handoff, run `make test`
and `make check`; run `make smoke` when lifecycle or release plumbing changes.
