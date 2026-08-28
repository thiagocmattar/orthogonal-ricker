# A2 Post-hoc Clipping Frontier — Review Candidate

> **Status:** proposed; not reviewed. This packet authorizes no config
> materialization, diagnostic execution, figure generation, retraining, cloud
> provisioning, spending, or retry.

This packet adds one post-hoc diagnostic over the six already accepted A2
checkpoints. It does not change training, the L1 implementation, checkpoints,
data, or the clipping operator.

## Questions

1. For each A2 checkpoint, how does validation loss trade off against observed
   model-wide logical opportunity when small activations are set exactly to
   zero after training?
2. Does increasing h-only L1 pressure change the absolute or incremental
   quality–`R_model` frontier under the same tested clipping cutoffs?
3. Which of the tested checkpoint/cutoff pairs are nondominated when lower
   validation loss and higher `R_model` are preferred?

This is a descriptive seed-0 comparison. Because all six sites are clipped
jointly and L1 can change weights and activations throughout the model, it does
not causally attribute a frontier change specifically to untargeted spillover.
An h-only clipping control would be required for that stronger attribution and
is not part of this packet.

## Exact Source Cohort

| Label | Config | Accepted run |
| --- | --- | --- |
| ReLU control | `012-a2-relu-control` | `001-20260827-150809-2eb832f6` |
| L1 lambda `0.1` | `013-a2-l1-1e-1` | `001-20260827-150808-8117d1fe` |
| L1 lambda `0.5` | `014-a2-l1-5e-1` | `001-20260827-173546-360c077f` |
| L1 lambda `1` | `015-a2-l1-1` | `001-20260827-193752-3fbbd6c0` |
| L1 lambda `2` | `016-a2-l1-2` | `001-20260827-220532-79995961` |
| L1 lambda `5` | `017-a2-l1-5` | `001-20260828-000829-0959f855` |

Each source resolves only to its accepted final checkpoint. No source is
discovered by scanning for a latest run, and no checkpoint is retrained.

## Frozen Clipping Design

| Item | Value |
| --- | --- |
| Mode | Absolute post-hoc magnitude clipping |
| Rule | Set `x` to exactly zero when `abs(x) <= t` |
| Sites | Jointly `[a, m, h, q_post, k_post, v]` in every layer |
| Cutoffs | `t = [0, 0.01, 0.03, 0.10, 0.30]` |
| Cutoff sharing | One common `t` across all selected sites and layers |
| Q/K placement | Post-RoPE only; PRE-RoPE ports are not clipped |
| Source order | Configs `012` through `017` in ascending order |
| Point order | Cutoffs in ascending order within each source |
| Total points | `6 checkpoints × 5 cutoffs = 30` |

The grid is fixed before looking at clipping outcomes. `t = 0` is the required
same-sweep reference; `0.01` and `0.10` reuse the A2 near-zero measurements;
`0.03`, `0.10`, and `0.30` align with the already proposed Phase-B numeric
cutoffs. Quantile and RMS-relative clipping are excluded.

This intervention is not a trained topology and must not be called `A6-POST`.
It happens to name the same six POST-RoPE operand ports, but it uses the
existing inclusive symmetric clipping rule after training. It is also not the
Phase-B threshold gate: clipping removes equality, while the trained gate
preserves equality and has different FFN threshold semantics.

## Evaluation and Saved Evidence

Every point uses the frozen complete A2 selection partition:

- 152 sequences and 311,296 input tokens;
- sequence length 2,048 and validation batch size 4;
- the exact validation-cache identity and hash already recorded by each source;
- seed 0 and the source precision/device policy; and
- eager attention required by actual-operand logical-product measurement.

One numbered multi-source diagnostic recipe, proposed as
`020-a2-posthoc-clipping-frontier.yaml`, must run all 30 points sequentially
under one lifecycle and experiment lock. It must save one cohort artifact with
one row per source/cutoff pair. Each row requires:

- exact source config, run, checkpoint-content, and validation-cache identity;
- cutoff, sites, validation batches/sequences/tokens, and validation loss;
- per-site exact-zero integer hits and element counts;
- per-operation zero-product integer numerators and denominators;
- exact `R_block` and `R_model` numerators, denominators, and fractions; and
- elapsed evaluation time.

For every checkpoint, derive only against its own `t = 0` row:

```text
delta_validation_loss(t) = validation_loss(t) - validation_loss(0)
delta_R_model(t) = R_model(t) - R_model(0)
```

Do not splice the ordinary training loss or diagnostic `019` into these
baselines because the clipping diagnostic forces eager attention.

## Figure and Report Contract

Generate one double-column, two-panel figure at
`experiments/02-a2-l1-screen/figs/04-a2-posthoc-clipping-frontier`:

- **Panel A:** x = observed `R_model` (%), y = validation loss;
- **Panel B:** x = within-checkpoint change in `R_model` (percentage points),
  y = within-checkpoint change in validation loss;
- one ordered curve per A2 checkpoint, with colorblind-safe color plus
  redundant line/marker encoding;
- a common threshold-marker encoding and one shared legend below the panels;
- visibly distinct `t = 0` anchors, but no selected/winner marker; and
- all 30 valid points, including dominated or adverse outcomes.

The companion Markdown must define the clipping rule and sites, explain both
panels, and tabulate all 30 points with absolute and paired metrics. The
provenance sidecar must pin the recipe, accepted diagnostic run, six source
identities, checkpoint-content hashes, validation-cache identity, artifact
hash, reduction, and output hashes.

## Interpretation Limits

- All six sources share one Pythia-14M architecture; “every A2 model” here
  means the six trained A2 conditions, not six model sizes.
- The evidence uses seed 0 and the selection partition only; it carries no
  seed uncertainty or confirmation claim.
- The finite cutoff grid maps tested points, not a continuous or global
  optimum.
- Joint clipping measures checkpoint-level model-wide thresholdability. It
  does not isolate the marginal effect of clipping any one site.
- `R_block` and `R_model` are potentially avoidable logical products, not
  removed FLOPs, latency, energy, throughput, or measured speedup.

## Implementation and Approval Sequence

Reuse the existing clipping operator, activation hooks, validation evaluator,
and logical-product accounting. Add only the smallest config-driven
multi-source orchestration and fixed-cohort plotting path needed to enforce the
contract above. The implementation must fail closed on a missing/duplicate
point, mismatched grid/site/source identity, absent `t = 0`, incomplete integer
counts, or inconsistent `R_model` arithmetic.

After this design is reviewed at an exact Git SHA:

1. restore the manifest to `Plan status: reviewed` for
   `[A2-relu-control, A2-l1-screen]` at that SHA;
2. implement and commit the numbered recipe, cohort diagnostic, renderer, and
   focused tests without changing clipping mathematics;
3. run a non-evidence production-shaped local timing calibration;
4. report first-source and full 30-point ETC, projected local completion time,
   evidence, assumptions, and uncertainty; and
5. wait for explicit diagnostic launch approval before producing evidence.

Diagnostic `019` completed six unmodified full-partition logical-opportunity
passes in 23.39 seconds on the local RTX 5070 Ti. Because clipping also computes
LM logits/loss, the current uncalibrated projection is approximately 3–6
minutes for 30 points, with no cloud cost. This estimate is planning evidence
only and must be replaced by the required calibration before launch approval.
