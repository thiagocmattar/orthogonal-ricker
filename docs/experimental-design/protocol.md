# Shared Experimental Protocol

> Review status and exact case-group scope live in
> [`../experiment_plan.md`](../experiment_plan.md).

This file owns campaign-wide settings. Stage-specific factors and seeds live in
[`cases.yaml`](cases.yaml); method mathematics and canonical site definitions
remain in [`../methods.md`](../methods.md).

## Models and Intervention Scope

- Pretrain Pythia-14M, Pythia-70M, and Pythia-410M from random
  initialization. Never load released checkpoint weights.
- Activation pressure targets only `h` in every transformer block.
- Stage A pressure runs use topology `A1-H` with explicit ReLU. At `h`, ReLU
  or a hard threshold replaces GELU as `mlp.act`; it is not applied after GELU.
- The threshold topology grid is owned by `cases.yaml`. Its Q/K placements are
  POST-RoPE, immediately before the downstream `QK^T` multiplication.
- L1 is `l1_naive`; OL1 is `orthogonal_l1`. Both target `h` and remain distinct
  methods in configs, artifacts, plots, and prose.

The proposed B1 design uses one global `kappa`: `m/h` are one-sided while
active attention sites use the case's one-sided or symmetric form. This mixed
per-group realization is not supported by the current single-form
`model.site_gate` contract and remains blocked in
[`workboard.md`](workboard.md).

## A1 Reproducibility Pins

The A1 packet fixes realized immutable revisions rather than the moving
`main` references originally used to create the local cache. The group was
reviewed at design commit `54be534f383001b4af3d3b43597e135d4ca6653d` and
completed at execution commit `276da7cd8e9142da48b95e12b46a99d61367ca8f`;
these pins are now a reproducibility record, not launch authority.

| Item | A1 identity |
| --- | --- |
| Dataset | `JeanKaddour/minipile` at revision `18ad1b0c701eaa0de03d3cecfdd769cbc70ffbd0` |
| Dataset fields | Loader default configuration; `train` split; `text` column |
| Training-cache source | All 1,000,000 training documents; 1,491,711,416 cached tokens |
| Training horizon | `lr-400m`: 1,526 updates; 400,031,744 input tokens |
| 14M architecture | `EleutherAI/pythia-14m-deduped` at revision `7386d9a4ae45aef494a6e704910394def3037fc5` |
| 14M tokenizer | `EleutherAI/pythia-14m-deduped` at the same revision |
| Training-implementation identity | `a1_pretraining_v1` |
| Tokenization | Append EOS; store `int32` token IDs |
| Training cache | `03-pythia-14m-minipile-random-full-10min`; SHA-256 `da82a2ea2e0080c7fd681c7a93b07d3d9ff3d5357a8640895a82d536a1eaf97c` |
| Validation source | First 500 validation documents; `shuffled_source_documents_half_v1`; partition seed `20260718` |
| Selection partition | 152 complete sequences; ordered-index SHA-256 `ffc857a6f0771929dd75c93bc17729de98a692f3a175ac5742cc9d101ff4ea47`; token SHA-256 `22bb7c27864f0e5941548c572d6c75b1b5ba6a4c13e4cd26f40f4de546c5cc19` |
| Confirmation partition | 186 complete sequences; ordered-index SHA-256 `8953a93f85c80a48d25fcacb7a0fbf44f6d9fd5b54037f92e01c5250f045ad99`; token SHA-256 `ee777ebdb8672b676ecfc05b2e7024c2f9446f8a9e46ac22b78e8a6c36f0890b` |

Integrity was recomputed locally on 2026-08-25 for the training, selection,
and confirmation `tokens.int32.bin` files; all three digests match the table.
The cache metadata also fixes `append_eos: true`, block size 2,048, `int32`
storage, the validation split and `text` column, 250 disjoint source documents
per validation partition, and partition seed `20260718`.

License review on 2026-08-25 found that the
[MiniPile dataset card](https://huggingface.co/datasets/JeanKaddour/minipile)
states that the Pile-derived subset is MIT-licensed, although its Hub metadata
uses the label `other`; retain the MiniPile and Pile citations and do not treat
that dataset license as a license grant over every third-party source
document. The
[Pythia-14M repository](https://huggingface.co/EleutherAI/pythia-14m-deduped)
declares Apache-2.0. These experiments consume its architecture configuration
and tokenizer only; model parameters are initialized randomly and released
checkpoint weights are not loaded.

The training-implementation identity is part of every condition fingerprint
and immutable config. It names the behaviorally frozen active training
contract, not the Git revision: manifests retain exact Git provenance
separately. Increment it whenever a change alters model construction,
sampling, optimization/schedule, an exercised threshold or pressure path,
data partitioning, or required training-time validation/checkpoint semantics
for an existing config. Adding a dormant, separately configured method path
does not by itself invalidate A1. Any later use of an A1 checkpoint across a
code revision requires explicit evidence that A1's active A0/no-pressure path
is unchanged; otherwise bump the identity and block reuse for review. Post-hoc
diagnostic schema versions remain separate.

## Batch and Learning-Rate Grids

| Model | Global batch | Peak-LR grid |
| --- | ---: | ---: |
| Pythia-14M | 262,144 input tokens/update | `{5e-4, 1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2, 3.2e-2, 6.4e-2, 1.28e-1, 2.56e-1, 5.12e-1}` |
| Pythia-70M | 262,144 input tokens/update | `{5e-4, 1e-3, 2e-3}` |
| Pythia-410M | 262,144 input tokens/update | `{1.5e-4, 3e-4, 6e-4}` |

At sequence length 2,048, each update contains 128 sequences. In the current
single-device harness:

```text
micro_batch_size * gradient_accumulation_steps = 128
```

| Model | Microbatch | Accumulation | Median core tokens/s | Peak reserved VRAM | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| Pythia-14M | 16 | 8 | 141,871 | 64.5% | Frozen for A1 on A40 48GB |
| Pythia-70M | 16 | 8 | 71,513 | 71.0% | Operational proposal; not reviewed |
| Pythia-410M | 4 | 32 | 10,853 | 61.5% | Operational proposal; not reviewed |

These measured values came from the 2026-08-25 idle-GPU profile on one Secure
Cloud NVIDIA A40 48GB, with two repeats over every listed microbatch and the
profile-only A1-H/ReLU workload with OL1 on `h` and AdamW. Profile report
SHA-256 values are respectively
`c2664914c956b834b5e85a9f03833ae5c928db0a815112d8475c9f2c8c7060bd`,
`3ee61118fc0baba3759bf74926a79c71eee7b044d1e1545e83798dda7b6ea091`, and
`97eed475385e834d25490c94245ee682a1435e81b77a8acbfcd3a8fef801dc43`.
The locally retrieved pack is at
`experiments/00-infrastructure-smoke/raw/runpod-20260825-b62f03b/`
(ignored by Git). Archive
`runpod-artifacts-b62f03b-20260825T1104Z.tar.gz` has SHA-256
`65ac472668b347ff74aab7886160cbc4674a8bd533cc037d92e171635e81623c`.

The Pythia-14M decomposition was explicitly approved on 2026-08-25 and closes
`OPS-04` for A1. The scale-up rows remain proposals under `OPS-07`. All three
measurements are operational, not scientific evidence or end-to-end ETCs; they
preserve 128 sequences per update and matched data grouping. Post-review ETC
calibration verifies a frozen decomposition and measures production
throughput; it never tunes it. Until the applicable case group is formally
reviewed, do not copy any row into scientific configs.

## Fixed Optimization Recipe

| Setting | Value |
| --- | --- |
| Optimizer | PyTorch AdamW; one parameter group over all trainable parameters |
| Betas | `(0.9, 0.95)` |
| Epsilon | `1e-8` |
| Weight decay | `0.1` |
| Gradient clipping | Global norm `1.0` |
| Hidden/attention dropout | `0 / 0` |
| Sequence length | `2,048` |
| Precision | BF16 CUDA autocast; FP32 parameters and AdamW state |
| Tuned optimizer field | Peak learning rate only |
| Training-event logging | Update 1, every 10 updates, and final update |

The released Pythia recipe uses Adam and FP16 dynamic loss scaling. This plan
records its deliberate implementation choices—PyTorch AdamW and BF16—rather
than treating them as exact Pythia replication.

Global clipping applies to the gradient consumed by AdamW: task-only for
`none` and OL1, task plus pressure for L1. OL1's separate post-AdamW pressure
direction is limited by its reviewed `step_budget`.

## Learning-Rate Schedule

For every run, linearly warm from zero to `eta_max` over the first 1% of
optimizer updates, then cosine-decay over the remaining 99% to
`0.1 * eta_max`. Only `eta_max` changes across an LR grid.

```text
warmup_steps = ceil(0.01 * max_steps)
```

The 400M-token schedule has 16 warmup updates; the complete-block-pass schedule
has 57. The final scheduled update reaches `0.1 * eta_max`.

## Training Budgets and Data Order

The cache contains 728,374 complete 2,048-token blocks and a 1,464-token tail.

| Budget ID | Updates | Input tokens | Rule |
| --- | ---: | ---: | --- |
| `lr-400m` | 1,526 | 400,031,744 | First 1,526 updates (195,328 complete blocks) of the seed's permutation; no wrap |
| `full-pass-wrap` | 5,691 | 1,491,861,504 | Visit every complete block once, repeat the permutation's first 74 blocks to fill the last global batch, and exclude the tail |

`full-pass-wrap` is one complete-block pass plus a 0.010% deterministic wrap.
Within a seed, all matched conditions share the complete-block permutation and
schedule hash. A nominal seed without the realized schedule hash is not a
matched data order.

For A1 seed 0 with the verified 14M cache and physical batch 16 × 8, the
realized `lr-400m` schedule SHA-256 is
`5feffe55fe37c764e86c6709500f1b0afad85be652de127f5fc7c958a7eb481c`.
Every A1 config must pin that value, the training-cache digest above, and the
selection-cache token digest above.

A1 consumes 195,328 complete blocks (26.817% of the complete-block cache),
leaves 533,046 complete blocks unused, and excludes the tail. Its LR decision
is therefore specific to the 400,031,744-token horizon; it is not a full-pass,
convergence, or horizon-independent optimum claim. This A1 reduction does not
change the budget assigned to any downstream group in `cases.yaml`.

Definitive pretraining stops at the exact optimizer-step budget, not a wall
clock limit. The runbook's calibration accumulates 600 seconds of completed
optimizer-step time, with other phases reported separately; that operational
timer must not alter this training condition.

## Validation and Checkpoints

- Evaluate every complete block of the selection partition in fixed order with
  evaluation batch size 4.
- Evaluate at update 1, every 191 updates, and the final update: 9 evaluations
  for `lr-400m` and 31 for `full-pass-wrap`.
- Intermediate validation is monitoring only. Stage decisions use the final
  selection measurement under the phase file's rule.
- Save one final model checkpoint. Do not save intermediate or optimizer-state
  checkpoints; all scientific training starts fresh.
- After all planned choices and analyses are frozen, evaluate each named
  headline checkpoint once on every complete confirmation block. Confirmation
  cannot rerank conditions or trigger another tuning round.

## Matching Contract

Within a seed, a treatment and control must match model/tokenizer revisions,
initial-parameter hash, train cache and realized schedule hash, global and
physical batch, optimizer/schedule, precision, budget, validation partition,
and final-checkpoint rule. Any mismatch is explicit and cannot be presented as
a controlled effect.

## Case Eligibility and Failure Classification

`experiment_log.md` assigns every scientific attempt exactly one reviewed case
classification: `eligible`, `scientific_failure`, `infrastructure_failure`, or
`unresolved`.

- `eligible` requires an accepted completed pretraining attempt at the exact
  budget, finite final selection loss, every stage-required metric with full
  declared coverage, and the required final checkpoint/artifact envelope.
- `scientific_failure` is an interpretable outcome of the frozen recipe, such
  as optimization divergence or nonfinite task values. It is resolved,
  ineligible for selection, retained as a dominated grid cell, and never
  retried or replaced.
- `infrastructure_failure` is an interruption without an interpretable
  scientific outcome. It remains unresolved until the recorded recovery is
  retried under the unchanged config or the tranche is explicitly stopped.
- `unresolved` covers running, statusless, inconsistent, ambiguous, or
  unreviewed state and blocks every decision that requires the case.

Only `eligible` cases enter phase decision rules. A complete grid may contain
`eligible` and `scientific_failure` cells, but no
`infrastructure_failure` or `unresolved` cell. Missing or unfavorable seeds are
never replaced. When an infrastructure retry exists, the physical case is
eligible only through one exact accepted eligible attempt; all prior attempts
remain logged.
