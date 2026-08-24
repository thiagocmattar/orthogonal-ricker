# Experimental Plan v0.1 — Review Draft

> Status: non-authoritative structural preview.
>
> The reviewed authority is docs/experiment_plan.md. Its status is still
> placeholder. This draft proposes decisions for review; it does not approve
> them. Do not allocate immutable config numbers, create scientific configs,
> calibrate, or launch from this file. Those actions remain blocked until the
> definitive plan incorporates reviewed choices and changes its status.

## 1. Decision notation

This draft uses three labels:

- **Proposed**: a concrete choice recommended for the definitive plan.
- **Dependency-selected**: a value fixed by an earlier reviewed tranche and
  then reused without retuning.
- **TODO**: a decision or identity that is genuinely unresolved. A TODO is a
  launch blocker for every dependent tranche.

All numerical results in this document are budgets, design choices, or
identities. They are not experimental findings.

## 2. Research questions, hypotheses, and nonclaims

The study asks three ordered questions:

1. When pressure is applied only at the MLP hidden site h, how do exact-zero
   and near-zero distributions change at h and at untargeted canonical sites?
2. How much logical zero-product opportunity is visible block-wide and
   model-wide, rather than only at the targeted MLP site?
3. Can explicit site gates, alone or with activation pressure, improve the
   validation-loss versus logical-opportunity frontier?

The motivating hypotheses are:

- higher configured activation_pressure.weight within one method increases
  near-zero mass at h;
- changes can spill over to untargeted attention ports;
- topology and gate form affect the location and amount of exact zeros;
- orthogonal_l1 may preserve task quality better than l1_naive at comparable
  achieved n_h(0.01) or R_model. Equal configured weights are not assumed to
  produce equal applied corrections across methods.

These are hypotheses, not required outcomes. Finite null, adverse,
non-monotone, and method-divergent results remain valid evidence and must be
reported. Nonfinite scientific failures are preserved and reported but are
ineligible for selection. The plan does not assume that every layer behaves
alike.

This study does not infer any of the following without a dedicated
measurement:

- exact zeros from differentiable L1 pressure;
- functional compensation from an activation correlation;
- sparse-kernel acceleration from an exact-zero count;
- wall-clock, energy, or FLOP/s gains from R_block or R_model;
- behavior at an unmeasured residual or attention-output site.

## 3. Estimands and matched comparisons

The primary quality estimand is final held-out causal-language-modeling loss on
a named validation partition at a fixed input-token budget.

The activation estimands at site s are:

    z_s = count(x == 0) / count(x)

and, for each declared threshold epsilon,

    n_s(epsilon) = count(|x| <= epsilon) / count(x)

The primary near-zero threshold is epsilon = 0.01. Epsilon = 0.1 is a
predeclared secondary diagnostic. The comparison is inclusive at equality.
Every named checkpoint-diagnostic artifact stores the integer numerator and
denominator; displayed fractions are derived from those counts. Count-first
pooling across all evaluated tokens and layers is primary. Layerwise values
are descriptive. Last-microbatch training monitoring is the explicitly
labeled exception and stores fractions rather than pooled integer evidence.

R_block and R_model use the integer logical-product accounting in
docs/diagnostics.md. They are potential logical zero-product opportunities,
not realized kernel skips or measured speedups.

Each treatment is compared with a same-seed control that matches:

- model and tokenizer identity and revision;
- random initialization and initial-parameter hash;
- train token cache and realized data-order schedule hash;
- validation partition and token-cache hashes;
- context, microbatch, accumulation, optimizer, schedule, precision, and token
  budget;
- validation implementation and diagnostic checkpoint rule.

A shared nominal seed is not enough to establish matching. Same-budget
comparisons require the complete realized schedule hash. The E1 unequal-budget
robustness comparison instead requires the explicitly verified 1.7B schedule
prefix defined in Section 12.

## 4. Shared protocol

### 4.1 Model, data, and cache identity

All main experiments pretrain Pythia-family causal language models from random
initialization. Released checkpoint weights are not loaded in Phases A–E1.

The following local cache is a **proposed** Phase A input because it already
has verifiable metadata. It still requires explicit ownership and license
review before promotion into the definitive plan.

| Item | Proposed identity |
| --- | --- |
| Dataset | JeanKaddour/minipile |
| Dataset revision | 18ad1b0c701eaa0de03d3cecfdd769cbc70ffbd0 |
| Training scope | 1,000,000 source documents; 1,491,711,416 cached tokens |
| Model architecture | EleutherAI/pythia-14m-deduped |
| Model revision | 7386d9a4ae45aef494a6e704910394def3037fc5 |
| Tokenizer | EleutherAI/pythia-14m-deduped |
| Tokenizer revision | 7386d9a4ae45aef494a6e704910394def3037fc5 |
| Tokenization | Append EOS; store int32 token IDs |
| Cache ID | 03-pythia-14m-minipile-random-full-10min |
| Training-token SHA-256 | da82a2ea2e0080c7fd681c7a93b07d3d9ff3d5357a8640895a82d536a1eaf97c |
| Validation source scope | First 500 validation source documents |
| Partition scheme | shuffled_source_documents_half_v1 |
| Partition seed | 20260718 |

Proposed selection partition:

- 250 source documents;
- 311,739 cached tokens;
- 152 complete 2,048-token evaluation sequences = 311,296 input tokens;
- 443 excluded tail tokens;
- ordered-index SHA-256
  ffc857a6f0771929dd75c93bc17729de98a692f3a175ac5742cc9d101ff4ea47;
- token-cache SHA-256
  22bb7c27864f0e5941548c572d6c75b1b5ba6a4c13e4cd26f40f4de546c5cc19.

Proposed confirmation partition:

- 250 source documents;
- 381,929 cached tokens;
- 186 complete 2,048-token evaluation sequences = 380,928 input tokens;
- 1,001 excluded tail tokens;
- ordered-index SHA-256
  8953a93f85c80a48d25fcacb7a0fbf44f6d9fd5b54037f92e01c5250f045ad99;
- token-cache SHA-256
  ee777ebdb8672b676ecfc05b2e7024c2f9446f8a9e46ac22b78e8a6c36f0890b.

TODO: record the dataset license, model license, tokenizer license, source split
names, text column, and the independent review accepting these identities.
Large-model identities, revisions, caches, and hashes remain TODO in Phase D.

### 4.2 Proposed Pythia-14M training recipe

| Config field or derived quantity | Proposed value |
| --- | --- |
| preprocessing.block_size | 2,048 input tokens |
| training.device | cuda |
| training.precision | bfloat16: CUDA BF16 autocast; FP32 parameters and AdamW state |
| training.micro_batch_size | 4 sequences |
| training.gradient_accumulation_steps | 8 microbatches |
| Effective batch | 32 sequences = 65,536 input tokens per optimizer update |
| run.training_schedule_scheme | random_contiguous_blocks_with_replacement_v1 |
| training.optimizer | adamw over all trainable parameters |
| training.adamw_betas | 0.9, 0.95 |
| training.adamw_eps | 1e-8 |
| training.weight_decay | 0.1 |
| Gradient clipping | none |
| training.warmup_steps and schedule | 260-update linear warmup, then constant learning rate |
| Warmup boundary | update 1 uses 1/260 of configured LR; update 260 reaches full LR |
| training.log_every | 50; also log update 1 and final update |
| training.max_wall_seconds | null for every definitive run |
| checkpoint.save_final | true |
| checkpoint.save_optimizer | false |
| Intermediate checkpoints | none |

The 260-update warmup covers 17,039,360 input tokens, approximately 1.002% of
a 1.7B-token run. It applies to the 400M, 1.7B, 3.4B, and 300M recipes.

The batch is expressed both in sequences and input tokens. Budget matching uses
input tokens, not predicted positions. Any model-size-specific reduction in
microbatch must preserve the reviewed effective token batch through
accumulation or be declared as a scientific mismatch.

### 4.3 Exact token budgets

Runs stop only at the listed optimizer update. The small overage is the
deterministic consequence of the fixed effective batch.

| Purpose | Nominal budget | Updates | Exact input tokens |
| --- | ---: | ---: | ---: |
| Screening | 400M | 6,104 | 400,031,744 |
| Main comparison | 1.7B | 25,940 | 1,700,003,840 |
| Long-training robustness | 3.4B | 51,880 | 3,400,007,680 |
| Pretrained adaptation | 300M | 4,578 | 300,023,808 |

A 400M result is screening evidence only. It cannot become a headline result
without its predeclared 1.7B promotion.

The infrastructure-only scaffold 00 smoke remains separate, keeps its own
settings, and may test the harness before plan review. It is not a 10M
scientific recipe, does not calibrate ETC, and cannot select or support a paper
condition.

### 4.4 Seeds and data order

The proposed scientific seed set is {0, 1, 2}. Within a seed:

    run.seed = model_initialization_seed = data_order_seed

The screen seed is 0. Promotions add seeds 1 and 2 while reusing the eligible
seed-0 result; they do not rerun or replace seed 0. Comparisons require the
same realized schedule hash and initial-parameter hash within seed.

No missing, failed, or unfavorable seed may be silently replaced. If a
predeclared three-seed comparison does not have three valid runs per condition,
the comparison remains incomplete and cannot use unequal sample sizes.

### 4.5 Validation policy

Selection and confirmation partitions are document-disjoint.

Selection validation is:

- deterministic and evaluated in fixed cached order;
- batch size 4;
- eval_batches null, meaning every complete 2,048-token block is evaluated
  once;
- validation.eval_every_steps = 763 for every Phase A–C 400M or 1.7B
  training config, with additional evaluations at update 1 and the final
  update and no duplicate when final is a cadence update;
- 763 updates = 50,003,968 input tokens;
- 9 evaluations at 400M: update 1 and the 8 cadence updates through final
  update 6,104;
- 35 evaluations at 1.7B: update 1, 33 cadence evaluations, and the final
  update;
- summarized with evaluated sequence count, evaluated token count, and
  excluded tail tokens.

Implementation dependency: extend the validation result schema to save the
cached partition-token total and excluded_tail_tokens, derived as cached
partition tokens minus evaluated input tokens, and test the full-cache and
partial-final-batch cases. Do not repair this provenance manually after a run.

Final selection-validation loss is the only Phase A1 ranking metric.
Intermediate validation, best-so-far validation, training loss, activation
monitoring, and checkpoint diagnostics cannot change the A1 ranking.

Confirmation cache metadata, counts, and hashes may be prepared and inspected
before tuning. Confirmation loss and every other model-derived confirmation
metric remain unevaluated and uninspected during tuning. The one confirmation
partition is sealed study-wide; it is not opened phase-by-phase. Its single
release occurs only after all planned A–D headline cohorts, eligibility
decisions, analysis code, exclusions, labels, and scientific statements are
frozen. If reviewed compute limits omit a planned headline cohort, freeze and
record the reduced final scope before opening confirmation.

At release, evaluate every frozen headline checkpoint once over the complete
confirmation partition using a new generic saved-checkpoint validation command
that reuses the training loss evaluator and writes checkpoint_validation.json.
Do not substitute a clipping-sweep zero row, because clipping evaluation may
change execution mechanics. Then run the predeclared confirmation activation
diagnostics required by each figure.

Implementation dependency: add and test that generic confirmation-validation
workflow, exact source-run selection, partition identity checks, complete-block
coverage, excluded-tail serialization, and output provenance before any
scientific launch. A disagreement with selection evidence is reported and can
downgrade or invalidate the frozen claim; it cannot rerank conditions, change
the cohort, or trigger another tuning round.

### 4.6 Throughput calibration and ETC

Calibration is operational plumbing, not a scientific run and not paper
evidence. The proposed representative case is the Phase A1 seed-0, LR = 1e-4
configuration with the production model, batch, accumulation, precision,
optimizer, schedule, logging, and h monitoring.

Calibration procedure:

1. Define setup time from command entry after the launch guard through the
   instant immediately before the first production-shaped optimizer update.
2. Use a calibration-only 600-second training window beginning at that first
   update. Do not write 600 seconds into the definitive scientific wall-time
   field.
3. End the cold window at the completion of the first optimizer update whose
   completion timestamp is at or beyond 120 training seconds. Record its
   actual duration T_cold and completed-update count N_cold; retain only later
   updates for steady-state estimation.
4. Require at least 10 completed optimizer updates after the cold window.
5. Estimate steady-state throughput from retained training-only time divided
   by retained updates. Scheduled validation and checkpoint work are excluded
   from this interval.
6. Time one complete selection-validation pass separately.
7. Time final-model serialization separately and record serialized bytes.
8. Record hardware, driver, CUDA and framework versions, GPU memory, setup
   time, every retained device-complete step time, retained step count, mean
   and sample standard deviation of step time, validation time, serialization
   time, and checkpoint size.

For a 1.7B A1 run:

    ETC = setup time
        + T_cold
        + (25,940 - N_cold) × retained mean seconds per update
        + 35 × full-selection validation time
        + final-model serialization time

Before launch, report first-run ETC, full-tranche ETC, projected local
completion time, evidence, workload and hardware assumptions, at least ±15%
engineering uncertainty per run, and at least ±20% for the full tranche.
Estimate the later four-run A1 promotion tranche separately after screening.

Calibration is unstable if retained step-time sample SD divided by its mean is
greater than 0.15, or if the absolute difference between first-half and
second-half retained means divided by the full retained mean is greater than
0.10. For an odd retained count, omit the central observation from the
half-to-half comparison. Calibration is unusable if it has fewer than 10
retained updates, meets either instability condition, changes production
overhead outside the separately timed validation/checkpoint operations, or
has incomplete validation/checkpoint timings. Recalibrate and do not request
launch approval.

Implementation dependency: the current calibration path needs a
calibration-only duration override; timing regions that exclude scheduled
validation and checkpoint work; low-overhead CUDA-event or equivalent timing
with device synchronization before reading results; every-step timing rather
than log-cadence samples; cold-window accounting; and separate setup,
validation, checkpoint-time, and checkpoint-size fields. The current
wall_seconds_train includes validation, and current logged step times are not a
sufficient calibration sample. A non-null scientific wall cap can also publish
a truncated pretraining run. Implement and verify the calibration-only path
before any calibration. Definitive configs retain a null wall cap and must
pass exact postflight step/token checks.

Use one BF16-capable CUDA GPU for both calibration and A1. Do not silently fall
back to another precision or combine unlike GPU models. Before launch, project
raw attempts, final checkpoints, diagnostics, and figures from measured
calibration bytes and require enough free storage for the complete tranche
plus 20% headroom.

The dense A1 calibration is valid only for the A1 dense workload. Before every
later tranche, obtain a same-hardware estimate for each materially new workload
shape: l1_naive, orthogonal_l1, gated training, each larger model/batch shape,
and each required diagnostic kind. Reuse prior timing only when model shape,
method overhead, batch, precision, logging, validation, and checkpoint policy
match. Otherwise run a plan-approved representative single-config calibration
or one timed diagnostic operation. Full-tranche ETC sums the appropriate
method-specific training ETCs and all required diagnostic times; calibration
never selects a scientific setting.

### 4.7 Completion, failure, invalid evidence, and retry

A training attempt is complete only when all of the following are durable:

- terminal manifest status is completed;
- exact required update and input-token budgets were reached;
- final task and required validation metrics are finite;
- config, Git, cache, partition, architecture, schedule, and seed provenance
  agree;
- config.yaml, manifest.json, metrics.json, predictions.jsonl, events.jsonl,
  and the required final model checkpoint pass integrity checks.

An escaping exception, caught OOM, nonfinite loss, detected incomplete budget,
or artifact-publication failure leaves the attempt failed. The serial tranche
stops at the first escaping failure and waits for review.

An abrupt process kill, host loss, or power loss may leave a stale running
manifest because no exception handler executes. After read-only inspection,
give that artifact evidence status invalid for the intended comparison, with
reason stale/incomplete, without rewriting its saved provenance or pretending
it completed. It is ineligible for selection.

A completed manifest that later fails provenance or integrity verification is
invalid evidence. Preserve its terminal record and document the integrity
finding; do not relabel it into a successful result.

A finite but poor, adverse, or non-monotone outcome is valid evidence. Preserve
all attempts. Do not exclude a result because it weakens a hypothesis or
frontier.

Retry an unchanged config only after an evidenced infrastructure failure. A
successful infrastructure retry may become the valid attempt for that same
immutable config and seed, while every failed attempt remains preserved. Never
retry away a scientific failure, change a config under the same number, or
substitute another seed. Recovery begins with read-only inspection. Any
changed scientific setting receives a new reviewed config.

The current parent runner restarts the complete ordered config list and can
therefore create new attempts for earlier cases when recovering a later case.
Before scientific launch, implement and test reviewed resume-at-case behavior:
validate the unchanged full runner/config list, preserve all prior attempts,
require an explicit reviewed start config, skip only earlier cases with a
recorded terminal classification, and begin at the first authorized
unattempted case under the normal lock. This is required to finish a screen
after a reviewed scientific failure without retrying it or losing later cases.

## 5. Measurement and artifact plan

### 5.1 Online training telemetry

At each training-log event, record:

- optimizer update and exact input tokens seen;
- task loss, learning rate, and update wall time;
- task-gradient norm and the two weight norms defined below;
- h monitoring values for Phase A1.

Global weight norm is the FP32 L2 norm over every model parameter, including
bias and normalization parameters. MLP weight norm is the FP32 L2 norm over
named parameters containing .mlp. and ending in .weight; it excludes biases
and parameters outside MLP modules. Both aggregate by summing squared
per-parameter L2 norms and taking one square root.

Record cumulative elapsed time, aggregate throughput, and peak allocated and
reserved GPU memory in the terminal summary. The current implementation does
not provide per-event cumulative elapsed time, throughput, or memory.

For both l1_naive and orthogonal_l1, also record pressure_loss,
pressure_weight, weighted_pressure_loss, raw pressure-gradient norm,
pressure/task gradient-norm ratio, task-pressure dot product and cosine, and
the conflict flag. Record augmented_loss for l1_naive; if it is emitted for
orthogonal_l1, label it monitoring-only because AdamW moments remain task-only.

For orthogonal_l1 only, additionally record the Adam-step diagnostics in
docs/diagnostics.md: task-direction norm, raw preconditioned
pressure-direction norm, pre-projection dot and cosine, projection flag,
post-projection dot and cosine, weighted raw direction ratio, trust scale,
final direction ratio, and both eligible and skipped parameter counts.

### 5.2 Phase A1 monitoring configuration

Use activation pressure only as a monitoring hook:

    enabled: true
    method: none
    sites: [h]
    weight: 0.0
    step_budget: null
    eps: 1e-12
    log_thresholds: [0.01, 0.1]

This applies no pressure. Logged h fractions describe only the last microbatch
at the training-log update. They are operational monitoring, not pooled
validation diagnostics and not primary sparsity evidence.

### 5.3 Named checkpoint diagnostics

Primary activation evidence comes from a named final checkpoint, exact run
identity, and the full frozen selection or confirmation coverage required by
the phase.

Use only the canonical sites:

    a, m, h, q_pre, k_pre, q_post, k_post, v

Do not introduce residual-stream or attention-output aliases. For every
requested site and layer, save exact-zero counts, near-zero threshold hits,
element counts, and coverage. Streamed histograms save bin edges, counts,
underflow, overflow, total elements, threshold-hit counters, layer/site and
source identity, and validation coverage. Separate the exact-zero atom from
the density bin containing zero and normalize the remaining density by the
nonzero total.

For every phase that reports R_block or R_model, including A2 and all topology
studies, produce activation_propagation.json with actual operand counts and
declared denominators. PRE-RoPE gate outputs are not substitutes for the
POST-RoPE operands consumed by QK.

Weight histograms are optional mechanism diagnostics only. Before using one,
freeze exact parameter names or patterns, bias and normalization-parameter
inclusion rules, bin edges, saved bin counts, out-of-range accounting, total
element count, source run, and checkpoint. They cannot alone establish why an
activation changed. Online global and MLP weight norms remain the explicitly
named training summaries.

### 5.4 Artifact identity

Every result and diagnostic must resolve to an exact immutable config, attempt,
Git commit, cache, run seed, checkpoint, validation partition, and diagnostic
command. Generated tables and figures load only pinned completed evidence.

The minimum result table reports condition, seed, final validation loss,
z_s, n_s(0.01), n_s(0.1), R_block, R_model, counts/coverage, and evidence
status where applicable. Use N/A for an absent or unevaluated site. Report 0%
only when a compatible counter evaluated the site and observed a zero
numerator.

## 6. Selection, uncertainty, and analysis freeze

Selection rules are applied only to eligible runs and only to the named
selection partition. Eligibility is assigned in two stages.

A run is training-eligible as a diagnostic source only if:

- its training evidence record is valid for diagnostic-source use;
- its manifest is completed at the exact step and input-token budget;
- its required training and selection-validation metrics are finite;
- config, source, cache, partition, schedule, seed, Git, and checkpoint
  provenance pass integrity checks.

A run becomes promotion-eligible only after it is training-eligible and every
promotion-required named diagnostic is durable and valid for the intended
comparison. A1 has no promotion diagnostic, so its training-eligible cases are
also its promotion-eligible cases.

Every planned case still appears in the report denominator with its completed,
failed, invalid, or unattempted classification. Ineligibility prevents ranking;
it does not erase the case or its artifacts.

Do not rank any screen until every planned case has a reviewed terminal
classification and no case remains unattempted. Resume-at-case recovery may
finish later cases after review, but scientific failures remain ineligible.
Each phase's predeclared collapse rule determines whether the remaining
promotion-eligible cohort is sufficient; never improvise a smaller favorable
subset after observing results.

For three-seed conditions, report all seed values, arithmetic mean, sample
standard deviation, and n = 3. Primary treatment comparisons use paired
same-seed deltas. Do not report standard error as if it were seed-to-seed
variation.

Pareto comparisons minimize validation loss and maximize R_model. A point is
dominated when another eligible point is no worse on both axes and strictly
better on at least one. Plot all eligible points, including dominated and
adverse ones; promotion status is an annotation, not an exclusion.

Before first confirmation evaluation, freeze:

- the complete condition and seed cohort;
- all eligibility and exclusion decisions;
- checkpoint and partition identities;
- reduction code, metrics, plot labels, and uncertainty summaries;
- the scientific statement the confirmation set is intended to assess.

## 7. Ordered execution graph and preflight

docs/runbook.md is the normative launch and recovery contract. Every training
row below is a separate serial tranche with its own chronological scaffold. A
promotion tranche is not configured until its upstream result and required
diagnostics are reviewed and each dependency-selected value is written into
the definitive plan.

| Order | Serial training tranche | Depends on |
| ---: | --- | --- |
| 1 | A1 learning-rate screen | reviewed plan, committed recipe, verified cache, calibrated ETC, explicit launch approval |
| 2 | A1 learning-rate promotion | reviewed complete five-case A1 screen |
| 3 | A2 pressure screen | frozen 14M LR and reviewed OL1 step budget |
| 4 | A2 pressure promotion | reviewed A2 screen diagnostics and frozen weights |
| 5 | B1 symmetric-gate topology screen | frozen Phase A protocol |
| 6 | B1 topology promotion | reviewed B1 screen propagation diagnostic |
| 7 | B2 gate-form screen | reviewed B1 promotion |
| 8 | B2 gate-form promotion | reviewed B2 screen diagnostics |
| 9 | C final-method screen | frozen Phase A and B settings |
| 10 | C final-method promotion | reviewed C screen diagnostics |
| 11 | D1 per-size LR screens, ordered by size then LR | all Phase D identities and frozen 14M LR |
| 12 | D2 cross-scale spillover cohort | reviewed per-size LR selections and frozen pressure cohort |
| 13 | D3 cross-scale intervention cohort | frozen Phase C intervention and per-size LRs |
| 14 | E1 fresh 3.4B robustness cohort | frozen E1 models, conditions, seeds, and schedule-prefix rule |
| 15 | E2 pretrained-adaptation cohort | complete reviewed exception protocol and implementation |

A3 is an analysis of A2 artifacts, not a training tranche. Data preparation
and throughput calibration are also single-config operations, not case-runner
tranches.

Promotion diagnostics are explicit dependency stages:

| After training stage | Required selection-partition operation before review |
| --- | --- |
| A2 screen | activation_histograms.json and activation_propagation.json for every training-eligible condition and matched dense control |
| A2 promotion | the same artifacts for the promoted multi-seed cohort; then A3 analysis |
| B1 screen | activation_propagation.json for every training-eligible topology/kappa condition and A0 control |
| B1 promotion | activation_propagation.json and predeclared activation histograms for the promoted cohort |
| B2 screen/promotion | propagation diagnostics for every condition entering a gate-form comparison |
| C screen/promotion | propagation diagnostics and pinned clipping_frontier.jsonl artifacts required by the frozen frontier |
| D2/D3/E1 | the phase-declared histogram and propagation artifacts for every headline condition |
| Study-wide confirmation release | checkpoint_validation.json plus each figure's predeclared confirmation diagnostics |

Each diagnostic config names exact tranche/config/run/checkpoint identities and
uses the complete selection or confirmation cache declared for that stage. Run
diagnostic configs one at a time under the direct launch guard; do not route
them through the training parent. Before allocating diagnostic config numbers,
freeze a reviewed ownership/scaffold contract consistent with the runbook, or
implement and test a sequential diagnostic runner. This is an implementation
and launch blocker, not permission to scatter configs.

For A1, the operational sequence is exactly:

1. Set the authority marker to Plan status: reviewed after owner review.
2. Create the A1 scaffold, runner, and five immutable configs; verify and
   commit the recipe on a clean checkout.
3. Prepare or verify the named token cache with one explicit config operation.
4. Run the one-config 600-second calibration and compute first-run and
   five-run ETCs.
5. Report ETCs, projected local completion, evidence, assumptions,
   uncertainty, hardware, memory, and storage headroom here.
6. Obtain explicit user approval to launch.
7. Run the A1 case runner serially.

Every preflight verifies reviewed dependencies; no unresolved tranche TODOs;
tracked sibling configs and exact output ownership; model, data, tokenizer,
optimizer, schedule, budget, validation, and random initialization; clean Git
status and recorded SHA; no active lock or launch; cache integrity; BF16 GPU,
memory, and disk capacity; focused tests plus make test, make check, and make
smoke; and first-run/full-tranche ETC with projected local completion. The
smoke is mandatory after the planned calibration, confirmation-validation, or
resume-at-case lifecycle changes. Never run case runners in parallel. Every
training tranche uses one parent runner, one lock, immutable numbered configs,
complete-list validation, stop-on-first-failure, and the reviewed
resume-at-case recovery path.

## 8. Phase A — Pythia-14M discovery

### A1. Learning-rate selection

Purpose: select the optimizer learning rate used by subsequent Pythia-14M
experiments. A1 does not test sparsity, spillover, or efficiency hypotheses.

Fixed condition:

- Pythia-14M random initialization;
- topology A0 and site_gate null;
- method none, h monitoring only;
- exact 1,700,003,840-token budget and the shared recipe above.

Seed-0 screen grid:

    1e-5, 3e-5, 1e-4, 3e-4, 1e-3

Run all five learning rates in one serial tranche. Rank eligible runs by final
selection-validation loss only after all five planned cases have a reviewed
terminal classification and no case remains unattempted. A failed or invalid
case remains reported and ineligible; it does not authorize ranking an
incomplete grid. Promote the two lowest-loss eligible learning rates. Break an
exact numerical tie in favor of the lower learning rate.

The promotion tranche adds seeds 1 and 2 for both promoted learning rates and
reuses seed 0, producing exactly three seeds per candidate. Freeze the
candidate with the lower arithmetic mean final selection-validation loss.
Break an exact mean tie in favor of the lower learning rate.

If fewer than two screen runs are eligible, any screen case is unattempted, or
either promoted candidate lacks three valid finite runs, do not rank unequal
or incomplete cohorts; stop for plan review.

A1 selection requires training metrics and the final model checkpoint, but no
post-hoc activation diagnostic. After the LR is frozen and before the selected
dense cohort is reused by A2, run the A2-declared histogram and propagation
diagnostics against those exact final checkpoints.

A1 output:

- appendix log-scale LR plot and table;
- every screen and promotion run with evidence status;
- individual seed losses, mean, sample SD, and n;
- the predeclared frozen LR;
- no sparsity or spillover claim.

### A2. h-pressure and spillover

Purpose: estimate how pressure applied only at h changes targeted and
untargeted activation distributions.

Dependency-selected condition: frozen A1 learning rate.

Dense controls reuse the eligible A1 selected-LR runs for seeds {0, 1, 2}
after verifying that all non-LR protocol fields match.

Proposed seed-0 pressure grid:

| Method | h-pressure weights |
| --- | --- |
| l1_naive | 0.1, 0.5, 1, 2, 5 |
| orthogonal_l1 | 0.1, 0.5, 1, 2, 5 |

All conditions use topology A0, site_gate null, sites [h], and the exact
1.7B-token budget. Pressure loss and task loss remain separate.

Before A2 promotion, run one pinned activation-histograms diagnostic and one
pinned activation-propagation diagnostic over the full selection partition for
every training-eligible seed-0 condition and its matched dense control. The histogram
config uses sites [a, m, h, q_post, k_post, v] and thresholds
[0.0, 0.01, 0.1]; its
selected_runs entries name exact source attempts. Promotion reads pooled
n_h(0.01) threshold hits from activation_histograms.json and never uses a
last-microbatch monitoring value. R_block and R_model come only from the
matching activation_propagation.json.

TODO: freeze activation-histogram bin count and finite display range before A2
diagnostic configuration. Underflow and overflow remain counted, so this
choice cannot alter threshold-hit or exact-zero estimands.

TODO: freeze orthogonal_l1 step_budget before the A2 screen. This is a
scientific parameter and cannot be inferred from method name or smoke tests.

Proposed promotion rule:

1. Require a complete reviewed ten-case screen: five positive weights for each
   of the two methods.
2. Form the common set of weights whose seed-0 conditions are
   promotion-eligible under both methods.
3. Within each method, form the Pareto set that minimizes final selection loss
   and maximizes final pooled n_h(0.01), restricted to that common set.
4. Take the union of weights represented on either restricted method frontier.
5. If that union has more than three weights, promote the numerically smallest,
   median, and largest union weights. For an even-sized union, use the lower
   median. If it has three or fewer, promote all.
6. Add seeds 1 and 2 for both methods at each promoted weight. Retain and plot
   every seed-0 screen point.

If the common promotion-eligible set or the restricted frontier union is
empty, A2 promotion is blocked and the screen is reported as inconclusive. Do
not create an asymmetric method cohort or substitute a failed seed-0 peer.

Primary matched spillover summaries compare each treatment with its same-seed
dense control at h and at a, m, q_post, k_post, and v. q_pre and k_pre may be
added only as separately named ports; PRE and POST results are never collapsed.

The primary effect vector contains paired changes in:

- n_h(0.01) and z_h at the targeted site;
- n_s(0.01) and z_s at each untargeted named site;
- final selection loss;
- R_block and R_model when a compatible propagation diagnostic and denominator
  are defined. A completed diagnostic with a zero numerator reports 0.

Near-zero change is not exact sparsity. L1 pressure may move a distribution
without producing exact zeros.

### A3. Orthogonal-pressure mechanism analysis

A3 adds no duplicate training grid. It reuses the completed A2 matched cohort
to compare l1_naive and orthogonal_l1 at common weights and seeds.

Report:

- task loss versus configured activation_pressure.weight within method;
- validation loss versus n_h(0.01);
- validation loss versus R_model when a compatible propagation diagnostic and
  denominator are defined;
- raw-gradient conflict frequency and cosine;
- projection frequency and pre/post-projection cosine;
- trust-scale and final pressure/task direction ratio;
- paired method deltas at common seed and weight.

Conflict frequency is conflict-true boundary count divided by the count of
accumulated optimizer boundaries with both raw task and pressure gradients.
Projection frequency is projection-fired boundary count divided by the count
of orthogonal_l1 boundaries eligible for the Adam-step diagnostic. Save the
integer numerators, denominators, and first/last covered updates over all
optimizer boundaries, not only log-cadence events. Implement and test these
aggregate counters before A2 if the current event stream does not preserve
all-boundary coverage.

Any quality advantage, lack of advantage, or instability is reported. The
orthogonal method is not described as unconditionally orthogonal; it removes
only a conflicting component under the implemented trust budget.

Proposed Figure 1 combines:

- layer-by-weight heatmaps for targeted h and separately named untargeted sites;
- paired Delta n_h(0.01) versus Delta n_s(0.01) for each named
  s in {a, q_post, k_post, v}, with m shown separately as the MLP-branch input;
- validation loss versus configured activation_pressure.weight, faceted by
  method, plus cross-method views against achieved n_h(0.01) or R_model;
- one representative h histogram panel comparing the seed-0 dense control,
  l1_naive, and orthogonal_l1 at the median promoted configured weight and the
  deepest transformer layer. For an even number of promoted weights, use the
  lower median. Include the panel only if all three seed-0 source artifacts are
  valid; otherwise report its predeclared absence instead of substituting a
  different condition.

## 9. Phase B — Pythia-14M architecture study

### B1. Symmetric-gate topology screen

Purpose: determine how explicit gate placement changes exact zeros and logical
opportunity at fixed training protocol.

Use the complete normative topology registry:

    A0
    A1-H
    A2
    A3
    A4-Q
    A4-K
    A4-V
    A5-QK-PRE
    A5-QK-POST
    A6-PRE
    A6-POST

A0 is one dense control with site_gate null. Each of the other ten topologies
uses symmetric_threshold with proposed kappa values:

    0.03, 0.10, 0.30

The screen therefore has 31 seed-0, 400M-token configurations: one A0 control
and 30 gated conditions. A topology ID specifies active ports only; the
operator and kappa remain explicit factors.

Rank using final selection-validation loss and R_model, with exact
activation-propagation counts from actual operands. Keep PRE- and POST-RoPE
topologies distinct. There is no inferred A2 reach ceiling.

Proposed promotion and collapse rule, frozen before the B1 screen:

1. After all 31 cases are classified and required diagnostics are valid, form
   the global promotion-eligible Pareto set over A0 and all gated points.
2. A gated topology family is a candidate when at least one of its points is
   on that global Pareto set. Rank candidate families by descending count of
   Pareto points, breaking ties by their order in the normative topology
   registry in docs/methods.md. Keep at most the first three families.
3. Within each kept family, promote the Pareto point with lowest validation
   loss and the Pareto point with highest R_model. If these are the same point,
   promote it once. Break a lowest-loss tie by higher R_model then lower kappa;
   break a highest-R_model tie by lower loss then lower kappa.
4. If no gated family contributes a promotion-eligible Pareto point, report the
   null screen and stop B1 promotion, B2, and the gated branch of C. Do not
   substitute a dominated family.

Promoted configurations run for 1.7B tokens with seeds {0, 1, 2}. Seed-0 400M
screen results appear only in the appendix and cannot headline Figure 2.
Figure 2 uses promoted 1.7B results and shows all promotion-eligible points,
uncertainty, topology labels, validation loss, and R_model.

When its fields match, the 1.7B A0 control is the completed A1 selected-LR
cohort plus pinned topology diagnostics; do not retrain an equivalent dense
control merely because it appears in a later phase.

### B2. Gate-form comparison

For dependency-selected topology families, compare:

    one_sided_threshold: x survives when x >= kappa
    symmetric_threshold: x survives when |x| >= kappa

Equality survives for both operators. Use the same proposed kappa grid
{0.03, 0.10, 0.30} at 400M with seed 0, reusing promotion-eligible B1 symmetric results
rather than duplicating them.

The B2 topology cohort is exactly the families promoted by B1. The following
rule is frozen before the B2 screen:

1. For each family, retain only kappa values whose one-sided and symmetric
   seed-0 conditions are both promotion-eligible.
2. Over both gate forms at those paired kappa values, form the within-family
   Pareto set. Select the kappa of its lowest-loss point and the kappa of its
   highest-R_model point; deduplicate if equal. Use the same loss/R/kappa tie
   rules as B1.
3. Promote both gate forms at every selected kappa, so the 1.7B comparison is
   paired by topology, kappa, and seed. Run seeds {0, 1, 2}.
4. Drop a family if it has no paired promotion-eligible kappa. If all families
   drop, report B2 as inconclusive and block a gate-form claim; C may proceed
   only after a reviewed amendment explicitly freezes the B1 symmetric gate.

No directional expectation is treated as a required result.

## 10. Phase C — frozen Pythia-14M frontier

Purpose: compare the final training families after Phase A and B settings are
frozen.

Across the dependency-selected architectures, the training families are
exactly:

1. A0 + none;
2. A0 + l1_naive at h;
3. A0 + orthogonal_l1 at h;
4. selected gate + none;
5. selected gate + l1_naive at h;
6. selected gate + orthogonal_l1 at h.

The three A0 rows are shared controls and are run or reused once per matched
seed, not duplicated once for every selected gated architecture.

Post-hoc clipping is a separate saved-checkpoint diagnostic, not a seventh
training family. Its sweep includes its own zero-threshold reference and
reports delta validation loss relative to that reference.

Proposed execution:

- 400M seed-0 screen of the frozen combinations;
- after every planned case is classified and required diagnostics are valid,
  form the global promotion-eligible validation-loss/R_model Pareto set;
- promote every noncontrol condition on that set to 1.7B seeds {0, 1, 2}, plus
  its required matched controls; exact metric ties retain every tied condition;
- if no noncontrol condition is promotion-eligible on the Pareto set, report a
  null C screen and do not manufacture a final intervention frontier;
- mark the frozen C cohort and analysis ready for the later single study-wide
  confirmation release; do not open confirmation at Phase C.

The architecture/gate cohort and pressure weights are the dependency-selected
values frozen by B2 and A2 before the C screen. TODO: freeze the exact post-hoc
activation-clipping modes and threshold grids before launching C; the zero-row
reference and confirmation decision rule are already fixed by Sections 4 and
13.

Figure 3 plots final validation loss against R_model, with redundant method
encodings, individual seeds, uncertainty, and separate notation for
post-hoc activation clipping. It must call R_model a logical opportunity, not
compute, acceleration, or measured speedup.

## 11. Phase D — scaling

Candidate random-initialized model sizes are Pythia 31M, 70M, 160M, and 410M.
Do not repeat the full Pythia-14M search.

Global blockers for Phase D:

- TODO: immutable architecture and tokenizer identities and revisions per size;
- TODO: immutable train/validation caches and hashes per size;
- TODO: per-size microbatch and accumulation preserving or declaring effective
  token-batch differences;
- TODO: per-size warmup rule and exact validation cadence;
- TODO: compute feasibility and calibrated ETCs.

### D1. Per-size LR screen

For each size, screen at 400M tokens with seed 0:

    frozen_14M_LR / 3, frozen_14M_LR, 3 × frozen_14M_LR

Select the finite eligible point with lowest final selection-validation loss;
break an exact tie in favor of the lower LR. Freeze that LR for every
same-size method comparison. Rank only after all three cases for that size have
a reviewed terminal classification. If none is promotion-eligible, remove that
size from D2/D3 and report the failed calibration; do not borrow another size's
LR. One or more eligible cases are sufficient to freeze the best among them.
A 400M LR screen is tuning evidence only.

### D2. Spillover replication

At each size, use the dense control plus up to three dependency-selected
low-, middle-, and high-weight h-pressure settings from Phase A. These labels
refer to the ordered weights, not an assumed effect size. The headline budget
is 1.7B tokens.

TODO: freeze whether D2 includes l1_naive, orthogonal_l1, or both, and freeze
the exact multi-seed cohort. At minimum, the dense control and every condition
used in a cross-scale claim require the same reviewed seed set.

Figure 4 reports paired changes by exact canonical site, rather than an
ambiguous aggregate attention sparsity. A cross-scale hypothesis is supported,
not required, if its predeclared direction persists; every exception remains
visible.

### D3. Frozen intervention across scale

Carry the dependency-selected Phase C topology, gate, kappa, pressure method,
pressure weight, and OL1 step budget to each size without topology retuning.

The proposed comparison contains:

- dense control;
- h pressure only;
- gate only;
- combined quality-oriented setting;
- combined aggressive setting.

TODO: identify each exact frozen configuration and seed cohort after Phase C.
All headline runs use 1.7B tokens.

The scale table has one row per model × noncontrol condition, so every delta
has one unambiguous owner:

| Model | Condition | Dense control loss | Condition loss | Paired Delta loss | Paired activation delta vector | Absolute R_model | Paired Delta R_model |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| size | pressure-only, gate-only, or named combined setting |  |  |  |  |  |  |

The activation delta vector is explicitly ordered as Delta n_s(0.01) for
s = [h, a, m, q_pre, k_pre, q_post, k_post, v]; absent sites are N/A and PRE
and POST ports are never collapsed. Absolute R_model is the condition's
logical opportunity; paired Delta R_model subtracts its same-seed dense
control. Report individual seed rows and the reviewed mean/sample-SD summary.

Do not label a near-zero metric as sparsity and do not convert R_model into an
actual throughput claim.

## 12. Phase E — robustness and scope

### E1. Longer training

Preferred design: train fresh, uninterrupted 3.4B-token runs with the final
frozen protocol, rather than attempting to resume final-model-only 1.7B
checkpoints.

TODO: freeze model sizes, exact conditions, and seed cohort. The proposed
minimum scientific contrast is dense, high-weight h pressure, and the selected
combined method for Pythia-14M and one larger size.

Match each fresh 3.4B run to its 1.7B counterpart by seed and
initial-parameter hash. Their full realized schedule hashes must differ because
their lengths differ. Instead, save and verify a schedule-prefix hash proving
that schedule[:25_940, :, :] from the 3.4B run exactly equals the complete
1.7B schedule for the same seed and batch shape. With accumulation 8 and
microbatch 4, that prefix contains 830,080 sampled block starts. Prefix-hash
metadata must describe the sliced shape and must not include the unequal full
max_steps values. Implement and test prefix-hash serialization before E1 if
the current schedule artifact does not expose this comparison.

A true 1.7B-to-3.4B continuation would require durable optimizer state,
scheduler state, data-order cursor, RNG states, and equivalence tests. The
shared final-model-only checkpoint policy does not provide those states, so
such continuation is not currently authorized.

### E2. Pretrained-model adaptation

E2 is an explicit exception to random initialization and remains supportive,
not primary. It may proceed only after a separate reviewed protocol freezes:

- released checkpoint identity, revision, and license;
- model size and checkpoint step;
- tokenizer and dataset relationship;
- control, l1_naive, and combined conditions;
- pressure weights, topology, gate, kappa, and OL1 step budget;
- optimizer-state policy and whether optimization restarts;
- data partition, 300M-token schedule, seeds, validation, diagnostics, and
  contamination considerations.

Until every item is resolved, E2 remains blocked and produces no scientific
config. It also requires implementation: the current definitive training
schema enforces random initialization and the model path constructs from
configuration rather than loading released weights. Add a reviewed
nonrandom-initialization exception, exact released-checkpoint reconstruction,
source-checkpoint provenance, optimizer restart/state handling, integrity
checks, and focused round-trip/lifecycle tests before E2 configuration.

## 13. Paper outputs and evidence gates

The Results section follows the scientific argument, not launch chronology.
Selection-partition renderings are provisional review artifacts. Released main
figures and tables use the one-time confirmation measurements for every frozen
headline metric; selection results remain available in the appendix and are
reported beside confirmation when the two materially differ.

| Output | Selection-stage source artifacts | Release source artifacts | Minimum evidence |
| --- | --- | --- | --- |
| Figure 1 | metrics.json, events.jsonl, activation_histograms.json, activation_propagation.json | checkpoint_validation.json and confirmation activation_histograms.json/activation_propagation.json | A2/A3 1.7B matched multi-seed cohort |
| Figure 2 | metrics.json and activation_propagation.json | checkpoint_validation.json and confirmation activation_propagation.json | B1 promoted 1.7B multi-seed cohort |
| Figure 3 | metrics.json, activation_propagation.json, clipping_frontier.jsonl | checkpoint_validation.json plus confirmation propagation and clipping-frontier artifacts | C promoted 1.7B multi-seed cohort |
| Figure 4 | metrics.json, activation_histograms.json, activation_propagation.json | checkpoint_validation.json plus confirmation histogram and propagation artifacts | D2 reviewed cross-scale multi-seed cohort |
| Figure 5 and scale table | metrics.json, activation_histograms.json, activation_propagation.json | checkpoint_validation.json plus confirmation histogram and propagation artifacts | D3 reviewed 1.7B multi-seed cohort |
| Appendix | config.yaml, manifest.json, metrics.json, events.jsonl, named diagnostic artifacts, failure records | Same pinned artifacts with selection/confirmation role labeled | Every planned case and attempt |

Every figure is generated from pinned saved artifacts. Use colorblind-safe
redundant encodings, untruncated axes unless a justified view is paired with
the full range, individual seeds, sample size, and uncertainty.

Paper release requires all frozen headline source artifacts to be valid, the
study-wide confirmation release to be complete, selection/confirmation
disagreements to be disclosed without retuning, exact source pins and analysis
code to be frozen, and every final table/figure to regenerate from those pins.
If confirmation invalidates a planned statement, revise or remove the claim;
do not search the confirmation partition for a replacement condition.

Each intended scientific artifact use has an evidence record whose status is
exactly one of:

- valid: terminal artifacts and required diagnostics satisfy the reviewed
  plan and integrity checks;
- provisional: a precisely stated limitation permits only the named use;
- invalid: the artifact cannot support the intended comparison.

The same durable artifact may have different statuses for different named
uses, each with its reason preserved. Separately, each valid or provisional
use has an analysis-role field such as tuning, screening, promoted main, or
robustness. Operational calibration is not scientific evidence and receives
neither a scientific evidence status nor an analysis role.

## 14. Compute priority

If compute is constrained, preserve evidential completeness before breadth:

1. A1 LR screen and promotion;
2. A2 h-pressure screen and promotion;
3. A3 mechanism diagnostics from A2;
4. B1 topology screen and promotion;
5. B2 gate-form screen and promotion;
6. C frozen 14M frontier;
7. D1 LR selection at the smallest feasible set of reviewed scales;
8. D2 replication at those scales;
9. D3 frozen intervention at the largest feasible reviewed scale;
10. E1 long training;
11. E2 pretrained adaptation.

Do not trade required seeds, matching, validation separation, or artifact
integrity for more conditions.

## 15. Remaining review ledger

The definitive plan may open Phase A1 only after these A1-scoped blockers are
resolved and incorporated:

- owner/license acceptance plus exact split and text-column identities for the
  proposed Phase A dataset, model, tokenizer, and caches;
- the validation excluded-tail schema fields and generic
  checkpoint_validation.json confirmation workflow;
- the calibration-only implementation and its verification;
- the reviewed resume-at-case implementation and tests;
- owner review of the shared protocol, A1 grid, selection, failure, and ETC
  rules in this draft.

The following decisions are intentionally downstream-scoped. They do not block
A1, but each must be resolved by a reviewed definitive-plan amendment before
its dependent screen or operation is configured:

- before A2: orthogonal_l1 step_budget, activation-histogram bins/range,
  all-boundary conflict/projection counters, and diagnostic-config ownership or
  a reviewed sequential diagnostic runner;
- before C: exact dependency-selected architecture/gate and pressure cohort,
  plus the post-hoc activation-clipping sweep;
- before D1–D3: all model/cache/batch/schedule identities, seed cohorts, and
  exact tranche composition;
- before E1: models, conditions, seeds, exact tranche composition, validation
  cadence, and schedule-prefix artifact;
- before E2: the complete exception protocol, exact tranche composition, and
  nonrandom-loading implementation.

Dependency-selected values cannot exist before their upstream evidence is
reviewed. Keep their deterministic decision rules in the reviewed plan, then
write each realized value through a reviewed amendment before allocating the
dependent configs. A downstream TODO blocks only its named dependent tranche;
it does not close an already reviewed upstream tranche.

Once the A1-scoped blockers are resolved in docs/experiment_plan.md and its raw
marker is exactly Plan status: reviewed, A1 immutable config allocation and
launch preparation may begin. Calibration follows the committed recipe, and
the case runner still waits for the explicit approval in Section 7. Until that
marker changes, all scientific configuration, calibration, and launch gates
remain closed.
