# Definitive Experiment Plan

Plan status: placeholder

> **Launch gate:** scientific experiment launches are blocked.

[`../exp-plan-v0.md`](../exp-plan-v0.md) is a non-final structural preview. It
shows phased searches and review-dependent promotions, but it is not the
definitive plan. Replace this placeholder only when the owner supplies the
final plan, then review it against `docs/methods.md`, `docs/diagnostics.md`, and
`docs/runbook.md` before creating scientific configs or case runners.

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
- do not allocate definitive runner or config numbers;
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

Every definitive training tranche gets one numeric case runner and matching
config folder, including a tranche with only one config. Screening,
review-dependent selection, and promotion are separate launches. The shared
parent executes each tranche sequentially under one lock. Until another
workflow has a reviewed sequential contract, launch its configs one at a time.
The launch handoff must include current-run and full-tranche ETCs.

### Paper Outputs

- planned tables and figures;
- exact source artifact kinds;
- uncertainty presentation;
- final confirmation and release gates.
