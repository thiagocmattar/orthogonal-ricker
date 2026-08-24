# Experimental Plan — Executive Summary for Advisor Review

> **Status:** proposal for scientific review, not launch authority. The immediate
> request is approval of A1 plus conditional endorsement of A2 and the staged
> design. Before launch, calibration will use a 600-second production-shaped
> training window plus separate setup, full-validation, and checkpoint timings
> to provide ETCs for approval. See the
> [full technical specification](exp-plan-v0.md).

## Aim

Test whether pressure applied only to the MLP hidden activation h changes
targeted and untargeted activation structure, and whether explicit activation
gates improve the tradeoff between selection-partition language-modeling loss and
model-wide exact-zero opportunity.

The study compares ordinary L1 with conflict-aware orthogonal L1 (OL1).
Hypotheses are that pressure may increase near-zero mass at h, untargeted sites
may also change, and gates or OL1 may improve the loss–activation frontier.
Null, adverse, and non-monotone results remain valid.

## Staged design

Stages A–E1 pretrain randomly initialized Pythia-family architectures.
Pythia-14M is the discovery model; larger models are tested only after the 14M
choices are frozen. E2 is the explicit pretrained-adaptation exception.

| Stage | Design | Budget and decision |
| --- | --- | --- |
| **A1 — Learning rate** | Dense baseline at 1e-5, 3e-5, 1e-4, 3e-4, and 1e-3 | Five seed-0 runs at 1.7B tokens; add seeds 1 and 2 for the best two; freeze the LR with lowest mean final selection loss |
| **A2–A3 — Pressure and spillover** | L1 and OL1 applied only at h, with weights 0.1, 0.5, 1, 2, and 5; A3 adds no training | Ten seed-0 runs at 1.7B; promote up to three common weights to three seeds; compare methods at achieved n_h(0.01) or R_model, not nominal weight alone |
| **B1–B2 — Explicit gates** | Ten gate-placement families at κ = 0.03, 0.10, and 0.30; then symmetric versus one-sided gates | B1: 31 seed-0 runs at 400M plus capped 1.7B promotions; B2 reuses matches and freezes a capped cohort that may retain both forms and several κ values |
| **C — Final 14M frontier** | Six families: dense none/L1/OL1 and gated none/L1/OL1 | Screen at 400M; current rule promotes every eligible Pareto condition to 1.7B × three seeds; approval waits for a numeric cap and gross-token bound or explicit acceptance of the uncapped rule |
| **D — Cross-size replication** | Repeat only frozen comparisons on candidate 31M, 70M, 160M, and 410M models | Three LRs per size at 400M; selected comparisons at 1.7B; fixed thresholds test literal transport and may reflect activation-scale changes, not a scaling law |
| **E — Conditional robustness** | Fresh 3.4B-token runs; pretrained adaptation is a separate supportive exception | Conditions and seeds remain contingent and capped later |

A Pareto condition minimizes selection loss while maximizing the relevant
activation outcome. Promotion means follow-up evaluation, not demonstrated
benefit. A 400M result is tuning evidence only. Main A–C comparisons require
the predeclared 1.7B follow-up with matched seeds {0, 1, 2}; D/E seed cohorts
remain unresolved. Every stage has a separate scientific and compute stop/go
decision.

## Fixed Pythia-14M protocol

| Item | Proposed choice |
| --- | --- |
| Data | Pinned MiniPile cache containing 1.49B tokens; training samples cached blocks with replacement |
| Context and batch | 2,048 tokens; microbatch 4; accumulation 8; effective batch 32 sequences or 65,536 input tokens/update |
| Precision | BF16 computation; FP32 parameters and AdamW state |
| Optimizer | AdamW; betas (0.9, 0.95); epsilon 1e-8; weight decay 0.1; no clipping |
| Schedule | 260-update linear warmup, then constant LR |
| Main budget | 1.7B processed input tokens |
| Validation | Full deterministic selection partition; A1 uses final loss, while later stages use phase-specific Pareto criteria |

Because the cache has 1.49B tokens, 1.7B and 3.4B runs revisit cached data. The
fixed warmup is about 4.3% of a 400M screen but 1.0% of a 1.7B run; promotion
therefore assumes that 400M behavior transports to the longer schedule.

## Evidence and interpretation

| Quantity | Definition and use |
| --- | --- |
| **Validation LM loss** | Final causal-language-modeling loss on the named selection or confirmation partition |
| **Near-zero mass** | n_s(0.01): fraction with abs(x) ≤ 0.01; threshold 0.1 is secondary |
| **Exact-zero rate** | z_s: fraction with x = 0; never conflated with near-zero mass |
| **Logical-opportunity proxy** | R_model: fraction of declared model scalar products with an exactly-zero activation operand |

Measurements use h and separately named MLP and attention ports. PRE- and
POST-RoPE ports are not collapsed. R_model is not a measured reduction in
FLOPs, latency, memory, energy, or cost. R_block is retained as a secondary
block-level logical-opportunity diagnostic.

The 500 validation documents are split document-disjoint:

- **Selection:** 250 documents; 311,296 evaluated input tokens; used for tuning.
- **Confirmation:** 250 documents; 380,928 evaluated input tokens; untouched
  for model evaluation until all A–D headline conditions and analyses are
  frozen.

Confirmation is an untouched **in-domain** check, not external validation. It
may support, weaken, or invalidate a statement, but cannot rerank conditions.
Headline reports show every seed, mean, sample standard deviation, n = 3, and
paired same-seed deltas. Three seeds describe limited run-to-run variation, not
broad robustness. Failed and unfavorable cases remain visible.

## Compute exposure

These are upper planning bounds, not approved launches. Tokens are processed
input tokens, not unique data, GPU-hours, or financial cost.

| Stage | Maximum current workload | Gross tokens |
| --- | ---: | ---: |
| A1 | 9 × 1.7B runs | 15.3B |
| A2 | 10 screen + up to 12 promotion runs at 1.7B | 37.4B |
| B1 | 31 × 400M + up to 18 × 1.7B | 43.0B |
| B2 | Up to 9 new 400M + 36 × 1.7B before reuse | 64.8B conservative gross bound |
| C | Dependency-selected; no hard cap yet | **TBD** |
| D–E | Scope and seed cohorts unresolved | **TBD** |

A1+A2 can reach **52.7B processed tokens**. The gross upper bound through B2 is
**160.5B before reuse**. This conservative sum double-counts reusable B1/B2
matches; net work can be lower. Approval of A does not authorize B–E. Each
materially different workload, especially OL1, gates, diagnostics, and larger
models, requires its own same-hardware timing.

## Decisions requested

1. Approve or revise the scientific questions, nonclaims, and four outcomes.
2. Accept Pythia-14M and MiniPile as the discovery setting, subject to identity
   and license review.
3. Approve or reduce the full-budget A1 and A2 screens; unlike B/C, both use
   1.7B-token screens.
4. Approve the screen–promote logic and either amend C with a hard cap or
   explicitly accept its current uncapped Pareto-promotion rule; set an overall
   compute ceiling.
5. Accept the validation-set size and the fixed absolute thresholds, recognizing
   that Phase D is a cross-size transport test.
6. Choose all four larger sizes or a smaller subset for D.
7. Decide whether E1 and supportive E2 remain planned or move outside the core
   study.
8. Before A2, approve the OL1 step budget and final diagnostic settings.
