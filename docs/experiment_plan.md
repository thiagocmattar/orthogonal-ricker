# Definitive Experiment Plan

Plan status: placeholder

> **Launch gate:** scientific experiment launches are blocked.

`TODO:` Replace this placeholder with the user-provided definitive experiment
plan, then review it against `docs/methods.md`, `docs/diagnostics.md`, and
`docs/runbook.md` before creating scientific configs.

After review, change the raw status line to exactly:

```text
Plan status: reviewed
```

That opens the executable launch gate.

Until that review is complete:

- do not infer or reconstruct the plan from Git history;
- do not reuse historical campaign configs or results;
- do not invent models, datasets, budgets, seeds, methods, thresholds,
  comparisons, promotion rules, diagnostics, or paper claims;
- do not allocate definitive config prefixes beginning at `01`;
- do not launch calibration, pretraining, diagnostics, or paper plotting as
  scientific evidence.

The infrastructure-only `configs/00-smoke.yaml` may be used to test the harness.
Its settings are not part of the future experiment plan.

## Required Plan Content

The supplied plan should make the following items explicit. Missing information
must remain `TODO:` rather than being guessed.

### Research Questions and Estimands

- primary and secondary questions;
- planned comparisons and matched controls;
- quantities to estimate and interpretation boundaries;
- claim language that the evidence may and may not support.

### Models and Data

- architecture source, revision, and random-initialization requirement;
- tokenizer and dataset identities, revisions, splits, and licenses;
- preprocessing, sequence length, cache identity, and validation partitions;
- model sizes and scaling stages, if any.

### Training Design

- optimizer and schedule;
- token or step budgets;
- batch shape and gradient accumulation;
- precision and checkpoint policy;
- model-initialization and data-order seeds;
- method, architecture, gate, and pressure factors;
- stopping, failure, retry, and exclusion rules.

### Diagnostics and Promotion

- diagnostic required for every run;
- diagnostics restricted to selected representatives;
- selection and confirmation partitions;
- promotion and collapse rules defined before observing promoted evidence;
- uncertainty and seed requirements.

### Execution

- ordered config tranches;
- dependencies between controls and interventions;
- calibration basis for ETC estimates;
- storage, hardware, and completion gates.

Every multi-run pretraining tranche must use one `paper-exp run-configs`
invocation and run sequentially. Until another workflow has a reviewed
sequential-run contract, launch its configs one at a time rather than in
parallel. The pretraining launch handoff must include current-run and full-queue
ETCs.

### Paper Outputs

- planned tables and figures;
- exact source artifact kinds;
- uncertainty presentation;
- final confirmation and release gates.
