# A2 L1 Spillover — Formal Review Packet

> **Proposal; no launch authority.** Approval must name the exact Git commit
> containing this packet. The definitive plan remains a placeholder until that
> approval is recorded.

This packet owns the proposed next review scope only:
`[A2-relu-control, A2-l1-screen]`. It changes grids and run scope without
changing the model, L1 method, optimizer, training loop, diagnostics schema,
data, validation, or failure semantics.

## Question and Evidence Level

As h-only L1 pressure increases over a 50× range, how do the targeted FFN
hidden activation `h` and untargeted sites `a`, `m`, `q_post`, `k_post`, and
`v` change relative to a matched ReLU-only control?

A2 is a seed-0 discovery screen. It supports a directional response and no
seed-uncertainty or replication claim. Full-grid added seeds are not planned.
If the final B2 winner cohort is later reviewed and completed, its matching
ReLU-only and selected-L1 components at seeds 1 and 2 may provide replication
of the selected-lambda contrast without duplicating a physical condition. They
do not replicate the `{0.1, 1, 5}` response.

## Exact Four-Run Cohort

| Physical cell | Seed | Topology and activation | Pressure |
| --- | ---: | --- | --- |
| ReLU control | 0 | `A1-H`; ReLU replaces GELU at `h` | `none` |
| L1 low | 0 | `A1-H`; ReLU replaces GELU at `h` | `l1_naive`, `sites: [h]`, lambda `0.1` |
| L1 central | 0 | `A1-H`; ReLU replaces GELU at `h` | `l1_naive`, `sites: [h]`, lambda `1` |
| L1 high | 0 | `A1-H`; ReLU replaces GELU at `h` | `l1_naive`, `sites: [h]`, lambda `5` |

The control is the lambda-zero curve anchor but is encoded as `pressure:
none`, never as an L1 condition with zero weight. No A1 training run or
checkpoint is reusable: A1 used topology `A0` and `lr-400m`, while A2 uses
`A1-H` and `full-pass-wrap`. A1 contributes only the frozen `lr_14m = 6.4e-2`
decision and shared immutable input pins.

## Fixed Training Contract

| Item | A2 value |
| --- | --- |
| Model | Randomly initialized Pythia-14M; released weights are not loaded |
| Seeds | `run.seed: 0`; `model_initialization_seed: 0`; `data_order_seed: 0` |
| Initialization check | All four cells must report the same initial-parameter SHA-256 before their first update |
| Peak LR | `6.4e-2` for every cell |
| Budget | `full-pass-wrap`: 5,691 updates; 1,491,861,504 input tokens |
| Seed-0 schedule | SHA-256 `35da3f6aa891a2248407344715e4c75e99cb518b17119a8e66004466a823a21c` |
| Global/physical batch | 262,144 tokens/update; 16 sequences × accumulation 8 on A40 48GB |
| Optimizer | Existing AdamW recipe, global gradient clipping `1.0` |
| Precision | BF16 autocast; FP32 parameters and optimizer state |
| LR schedule | 57-step linear warmup, then cosine decay to `0.1 × peak` |
| Validation | Complete selection partition at update 1, every 191 updates, and final: 31 evaluations |
| Checkpoint | Final model only; no optimizer state or intermediate checkpoints |

The current `l1_naive` implementation is frozen. It computes task and L1
gradients separately for gradient norm, dot-product, cosine/alignment, and
conflict diagnostics, and applies the existing augmented-loss update. This
packet authorizes no method or implementation optimization.

Before formal review, `OPS-08` code work is limited to config-admission logic
in `src/paper_exp/design.py` and its tests; no other `src/paper_exp` file may
change. After review, every tracked `src/paper_exp` blob must remain identical
through execution. Any difference, or any other change that can affect the
exercised A1-H/L1 path, requires a new training identity and scientific
re-review.

## Spillover Measurement

Run the existing activation-histogram diagnostic after all accepted final
checkpoints have exact config/run identities. One four-source recipe must use:

- sites `[h, a, m, q_post, k_post, v]`;
- thresholds `[0, 0.01, 0.1]`;
- one common, explicitly recorded histogram bin count and range; and
- the complete selection-validation partition.

No diagnostic-schema change is required. Existing layer rows already store
integer threshold hits and totals, finite counts, RMS, histogram counts,
underflow, overflow, source identity, and validation coverage. Pool threshold
fractions by summing hits and totals. If a compact site-level RMS is reported,
derive it as:

```text
sqrt(sum_l finite_l * rms_l^2 / sum_l finite_l)
```

RMS is `sqrt(mean(x^2))`, an activation-scale summary. It is not an
unnormalized activation norm and does not define spillover. The primary A2
report contains final validation loss and the seed-0 change from the ReLU
control in `n_s(0.01)` at every named site. Exact-zero mass, `n_s(0.1)`, the
already saved RMS, and layer distributions are supporting evidence.

## Completion and Claims

All four cells require reviewed terminal classifications. Eligible and
scientific-failure cells resolve the grid; infrastructure-failure or unresolved
cells block completion. Failed or unfavorable cells are never replaced.

Permitted claims are limited to a seed-0 dose response and within-seed
targeted/untargeted changes. A2 cannot claim seed robustness, population
uncertainty, functional compensation, compute reduction, or measured speedup.
A null result does not stop B1, but B1 is then interpreted as a general
multi-site threshold study.

## Remaining Readiness Work

- `DIAG-02` requires no new diagnostic machinery under the existing artifact
  audit and reduction contract above.
  Exact histogram geometry is an analysis parameter pinned once in the later
  diagnostic config, before diagnostic execution and before inspecting its
  outputs; it is not a pretraining or plan-review blocker.
- `OPS-08` remains the sole pre-review blocker: config validation must admit
  exactly one seed-0
  control and exactly the three reviewed L1 weights, reject every other A2
  cell, verify fingerprints, and prevent duplicates. This is launch/config
  plumbing, not a model, optimizer, pressure-method, or diagnostic change.
- After formal review and materialization, configs would receive the next four global IDs
  (`012`–`015`) in one A2 screen scaffold. No number is reserved by this
  proposal.

## Preliminary Operations

The control projection is approximately 2.45–2.55 hours from accepted A1 A40
timings. The current L1 path has no direct production A40 timing; its provisional
range is 4.0–5.5 hours per run. This implies:

- one-A40 serial: about 14.5–19.1 hours;
- one coordinator with two A40 workers: about 8–11 hours; and
- historical-price lifecycle planning only: roughly `$7.5–$10.4` at
  `$0.44/A40-hour`.

These are not launch-quality ETC or live-price estimates. After reviewed
materialization, calibrate the exact ReLU control and lambda-1 L1 cells for 600
completed optimizer-step seconds each on the intended homogeneous A40 setup.
Then return with measured first-run/full-cohort ETC, current price, cost cap,
deadline, and a separately reviewed two-worker authorization.

## Review Sequence and Effect

This design-only proposal is not yet eligible for formal approval because
`OPS-08` fails closed for non-A1 groups. First implement and verify only the
exact four-cell config-admission contract, while preserving the frozen
scientific-path blobs above. Then commit the complete review packet and return
one exact SHA for formal approval.

Approval of that later exact SHA authorizes recording the plan as reviewed for
`[A2-relu-control, A2-l1-screen]` and materializing those four configs. It does
not authorize RunPod provisioning, spending, calibration, definitive
training, retry, replacement, or teardown. Those remain separate explicit
approvals.
