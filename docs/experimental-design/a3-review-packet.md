# A3 OL1 Screen — Review Candidate

> **Status:** awaiting explicit review at one committed Git SHA. This packet
> proposes the A3 scientific contract only. It authorizes no config
> materialization, calibration, RunPod spending, or scientific launch.

## Question

At the five pressure weights already used in A2, does orthogonal L1 (OL1)
improve the final-validation-loss versus achieved-`n_h(0.01)` frontier relative
to naive L1, and how often do OL1 conflict removal and its trust budget act?

Here `h` is the FFN hidden activation after the first MLP projection and ReLU,
immediately before the `W2` down-projection matrix multiplication. A3 applies
pressure only at `h`; it does not change activation placement or gating.
`A3` here is the experimental stage label; every new cell uses topology
`A1-H`, not topology `A3`.

A3 is a seed-0 directional screen (`n = 1` per condition). It does not estimate
seed uncertainty or establish robustness across initializations.

## Exact Scientific Cohort

Five new Pythia-14M cells use topology `A1-H`, ReLU at `h`, and
`orthogonal_l1` at `sites: [h]`:

| Cell | Seed | `weight` | `step_budget` |
| --- | ---: | ---: | ---: |
| OL1 low | 0 | 0.1 | 0.1 |
| OL1 lower-middle | 0 | 0.5 | 0.1 |
| OL1 central | 0 | 1 | 0.1 |
| OL1 upper-middle | 0 | 2 | 0.1 |
| OL1 high | 0 | 5 | 0.1 |

There is no OL1 lambda-zero run. The curve anchor reuses completed A2 control
config `012`; matched naive-L1 evidence reuses completed configs `013`-`017`.
No accepted A1 or A2 training condition is rerun. After approval, the five new
physical configs receive the next global numbers `019`-`023` in ascending
weight order.

| Reused role | Exact accepted evidence |
| --- | --- |
| ReLU anchor | Config `012-a2-relu-control`, run `001-20260827-150809-2eb832f6` |
| Matched L1 curve | Configs `013`-`017`, runs `001-20260827-150808-8117d1fe`, `001-20260827-173546-360c077f`, `001-20260827-193752-3fbbd6c0`, `001-20260827-220532-79995961`, and `001-20260828-000829-0959f855` |
| Accepted A2 activation evidence | Config `018-a2-activation-histograms`, run `001-20260828-082044-a031175f` |

## Frozen OL1 Trust Budget

The proposed decision is `ol1_step_budget = 0.1`, fixed before calibration and
used unchanged for all five cells. At each optimizer boundary, it caps the
weighted, conflict-safe pressure-direction norm at 10% of the task Adam
direction norm.

This is a conservative, dimensionless intervention bound, not a tuned
hyperparameter, sparsity target, or convergence guarantee. It prevents the
post-AdamW pressure correction from dominating the task direction while
leaving `weight` as the only A3 grid axis. A3 does not sweep this value and
does not select it from smoke timing, A2 outcomes, or A3 calibration.

## Fixed Training Contract

Everything except pressure method, pressure weight, and OL1's fixed trust
budget is inherited unchanged from accepted A2:

| Item | A3 value |
| --- | --- |
| Model | Randomly initialized Pythia-14M; released weights are not loaded |
| Topology | `A1-H`; ReLU replaces GELU at `h` |
| Seeds | Run, initialization, and data-order seed `0` |
| Peak LR | Frozen `lr_14m = 6.4e-2` |
| Budget | `full-pass-wrap`: 5,691 updates; 1,491,861,504 input tokens |
| Global/physical batch | 262,144 tokens/update; 16 sequences × accumulation 8 on A40 48GB |
| Optimizer | Existing AdamW recipe; task-only global gradient clipping at `1.0` |
| Precision | BF16 autocast; FP32 parameters and optimizer state |
| LR schedule | 57-step linear warmup, then cosine decay to `0.1 × peak` |
| Validation | Complete selection partition at update 1, every 191 updates, and final |
| Checkpoint | Final model only; no optimizer state or intermediate checkpoint |

The existing OL1 mathematics is unchanged: AdamW moments use task gradients
only; the current pressure gradient is preconditioned by the task second
moment; only a conflicting component is removed; then the bounded correction
is applied after AdamW.

## Required Evidence

Each run retains the existing sampled train events at update 1, every 10
updates, and final. In addition, terminal integer counters aggregate every
completed OL1 optimizer boundary:

- total OL1 boundaries;
- raw task/L1-gradient conflicts;
- preconditioned pressure projections;
- trust-budget-limited corrections; and
- summed eligible and skipped parameter-tensor counts.

These counters reuse values already computed by OL1. They add no gradients,
hooks, config fields, per-boundary event rows, or post-hoc mechanism pass.
Report each event count with total-boundary coverage; eligible/skipped counts
are tensor-boundary totals, not unique tensors.

The exact terminal metric suffixes are
`ol1/optimizer_boundary_count`,
`ol1/raw_gradient_conflict_boundary_count`,
`ol1/preconditioned_projection_boundary_count`,
`ol1/trust_budget_limited_boundary_count`,
`ol1/eligible_parameter_tensor_count_sum`, and
`ol1/skipped_parameter_tensor_count_sum`, under the `training/` or
`calibration/` prefix. A completed definitive cell requires the common
denominator to equal all 5,691 optimizer boundaries.

After all five checkpoints are accepted, one common activation-histogram
diagnostic will evaluate the A2 control, five A2 L1 cells, and five A3 OL1
cells over the complete selection partition. It needs only site `[h]` for the
matched `n_h(0.01)` curves and uses one common stored histogram geometry. It
does not rerun training.

## Analysis and Decision

Report complete matched curves for final validation loss and count-first
`n_h(0.01)`, including every adverse, dominated, failed, or invalid cell.
Mechanism summaries report conflict, projection, and budget-limited counts
divided by total completed OL1 boundaries, plus eligibility coverage.

For `lambda_B2`, consider only nonzero weights with valid matched A2-L1 and
A3-OL1 evidence. Form the A3 OL1 Pareto set that minimizes final selection
loss and maximizes `n_h(0.01)`. Select lowest loss, then higher
`n_h(0.01)`, then lower weight. If no matched nonzero point is valid, do not
define `lambda_B2`; stop B2 and C3.

Permitted claims are limited to this matched seed-0 response and the measured
OL1 mechanism frequencies. A3 cannot claim seed robustness, causal functional
compensation, compute reduction, or measured speedup.

## Completion and Failure Rules

All five new cells require a reviewed terminal classification. Scientific
failures remain visible grid outcomes and are never replaced. Infrastructure
failure or unresolved status blocks the complete response; retry requires a
separate unchanged-run authorization.

## Post-Review Operations

Approval of this packet authorizes freezing `ol1_step_budget = 0.1`, activating
only group `[A3-ol1-screen]`, and materializing exactly five configs. It does
not authorize calibration or cloud use.

After materialization, calibrate only the endpoint cells (`weight = 0.1` and
`5`) concurrently for exactly 600 completed optimizer-step seconds, one
process per distinct A40, to replace the provisional ETC. Calibration may
validate timing, memory, worker isolation, and all-boundary counter coverage;
it may not change the grid or trust budget. Return with calibrated ETC, cost,
deadline, and a separate definitive-launch approval request.
