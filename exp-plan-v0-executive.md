# Experimental Plan — Executive Brief

> **Status:** lean proposal for advisor review. It is not launch authority.
> Scientific configuration and calibration remain blocked until
> [the definitive plan](docs/experiment_plan.md) is reviewed.

## Goal

The plan tests three hypotheses motivating the working paper:

1. L1 pressure at the FFN hidden activation `h` may be accompanied by
   **sparsity spillover** at untargeted FFN and attention sites.
2. Threshold placement may change the model-wide share of logical products
   with an exactly-zero activation operand—a potential logical skip.
3. Selected multi-site thresholding plus OL1 may improve the
   validation-loss/logical-opportunity frontier, and the result may persist
   from Pythia-14M to 70M and 410M.

These are questions, not assumed findings. `R_model` is a logical
zero-product opportunity, not measured runtime speedup.

## Fixed training recipe

All runs pretrain randomly initialized Pythia models on MiniPile. Released
checkpoint weights are not loaded.

| Model | Global batch | Peak-LR sweep |
| --- | ---: | ---: |
| Pythia-14M | **262,144 input tokens/update** | `{5e-4, 1e-3, 2e-3}` |
| Pythia-70M | **262,144 input tokens/update** | `{5e-4, 1e-3, 2e-3}` |
| Pythia-410M | **262,144 input tokens/update** | `{1.5e-4, 3e-4, 6e-4}` |

At context length 2,048, each update contains 128 sequences. Microbatch and
accumulation may change with hardware, but their product must remain 128.

| Setting | Fixed value |
| --- | --- |
| Optimizer | AdamW |
| Adam parameters | `beta = (0.9, 0.95)`, `epsilon = 1e-8` |
| Weight decay | 0.1 |
| Gradient clipping | Global norm 1.0 |
| Dropout | Hidden 0; attention 0 |
| Precision | BF16 computation; FP32 parameters and optimizer state |
| LR schedule | 1% linear warmup, then cosine decay to 10% of peak |
| Tuned optimizer field | Peak learning rate only |

The 14M discovery runs and the C2/C3 replications use one complete-block
MiniPile pass plus a fixed-batch wrap. The exact realization is 5,691 updates:
every complete 2,048-token block once, followed by the first 74 blocks of the
same seeded permutation to fill the last fixed-size update. C1 LR sweeps use
1,526 updates = 400,031,744 tokens.

## Canonical sites and threshold policy

Pressure always targets only `h`, the FFN activation between `W1` and `W2`.
Stage A2 and Stage A3 use topology `A1-H` with explicit ReLU, matching the
paper. `A0` is stock Pythia with GELU and no pressure.

| Alias | Exact operand |
| --- | --- |
| `a` | Attention-branch LayerNorm output entering the QKV projection |
| `m` | FFN-branch LayerNorm output entering `W1` |
| `h` | FFN activation-function output entering `W2` |
| `q_post`, `k_post` | Post-RoPE query/key operands entering `QK^T` |
| `v` | Value operand entering the attention-probability-by-value product |

Each alias means the post-intervention operand passed downstream. At `h`,
ReLU or a hard threshold replaces GELU as `mlp.act`; it is not applied after
GELU. Stage labels name experiments, whereas topology IDs name active site
sets; always qualify the latter as, for example, topology `A3`.

B1 studies these repository site variants:

| Topology | Active sites |
| --- | --- |
| `A1-H` | `h` |
| `A2` | `m, h` |
| `A3` | `a, m, h` |
| `A4-Q / A4-K / A4-V` | `a, m, h` plus `q_post / k_post / v` |
| `A5-QK-POST` | `a, m, h, q_post, k_post` |
| `A6-POST` | `a, m, h, q_post, k_post, v` |

Only POST-RoPE Q/K sites are used because they are the operands immediately
preceding `QK^T`.

One global `kappa in {0, 0.03, 0.10, 0.30}` applies to all active sites.
FFN sites `m, h` are always one-sided. Attention sites
`a, q_post, k_post, v` are either one-sided or symmetric. One-sided keeps
`x >= kappa`; symmetric keeps `abs(x) >= kappa`; all other values become zero.
This proposed mixed-form extension requires review of the repository's current
single-threshold-form contract. The matrix has 56 conceptual cells, but six
symmetric `kappa = 0` cells are functional duplicates of topology `A2` at
`kappa = 0`; reusing that
anchor leaves 50 unique training runs plus the shared `A0` control. Family
selection counts only positive-kappa Pareto points, so the shared anchor cannot
favor multiple symmetric families.

## Lean experiment matrix

| Stage | Experiment | Question answered |
| --- | --- | --- |
| **Stage A1 — 14M LR** | `A0`; three peak LRs; one complete-block pass plus wrap; seed 0 | Which peak LR should every 14M comparison use? |
| **Stage A2 — L1 spillover** | Topology `A1-H` + ReLU; h-only L1; seed-0 `lambda = {0, 0.1, 0.5, 1, 2, 5}`; three seeds at `lambda = {0, 1}` | How do targeted `h` and untargeted `a, m, q_post, k_post, v` distributions change with pressure? |
| **Stage A3 — OL1 robustness** | Same seed-0 lambda grid and sites as Stage A2, replacing L1 with OL1 | Does conflict-aware pressure reduce quality sensitivity, and which optimizer diagnostics accompany the difference? |
| **Stage B1 — threshold ablation** | Eight site variants × four kappa values × applicable attention forms; no pressure | How do placement, form, and threshold value change the validation-loss/`R_model` frontier, and which family is selected? |
| **Stage B2 — combined winner** | Apply OL1 at one Stage A3-selected lambda to all kappa values of the selected Stage B1 family | Does threshold + OL1 improve the frontier over `A0`, pressure-only, and threshold-only controls? |
| **Stage C1 — scale LR** | 70M and 410M LR sweeps at 400M tokens | Which peak LR should each larger model use? |
| **Stage C2 — spillover replication** | Repeat the Stage A2 L1 grid at 70M and 410M, using each selected LR and one complete-block pass plus wrap | Does the spillover response persist with model scale? |
| **Stage C3 — frontier replication** | Transport the frozen Stage B2 topology, form, kappa cohort, OL1 lambda, and step budget to 70M and 410M | Does the selected frontier persist without retuning the intervention? |

All complete grids use seed 0 and are exploratory. Independently of B1/B2, the
central spillover contrast—ReLU-only (`lambda = 0`) versus L1 at `lambda = 1`—
uses seeds 0, 1, and 2 at every model size. After B2 selects a winner, its
six-condition component cohort (`A0`, ReLU-only, L1-only, OL1-only,
threshold-only, threshold+OL1) also uses three seeds at every size. Full curves
are labeled `n = 1`; only these selected contrasts are `n = 3`.

No selection is made from an unresolved grid. If a model has no eligible LR,
its dependent stages stop; a null B1 or no valid matched nonzero A2/A3 lambda
stops B2/C3. Scientific collapses and adverse results remain reported.

## Measurements and outputs

Spillover is measured by paired changes at every named site in:

- exact-zero fraction `z_s`;
- near-zero mass `n_s(0.01)`, with `n_s(0.1)` as sensitivity;
- pooled RMS and layerwise activation distributions;
- validation loss.

`A0` and every ReLU or threshold condition used in a frontier/table report
observed `R_model`; each topology also reports its validated ceiling
`R_model^max`. PRE/POST sites are never collapsed. Both quantities are logical
opportunities, not measured compute savings.

The minimum paper package is:

1. **Figure 1 — sparsity spillover:** Stage A2/C2 lambda responses across 14M,
   70M, and 410M. The fixed distribution panel uses 14M, the deepest layer,
   all six named sites, `lambda = 0` versus `lambda = 1`, and all three seeds.
2. **Figure 2 — logical opportunity:** `R_model` accounting and
   `R_model^max` by topology and model.
3. **Figure 3 — intervention and mechanism:** `A0`, ReLU-only, L1-only and
   OL1-only at `lambda_B2`, threshold-only, and threshold+OL1; Stage B1/C3
   frontiers; paired one-sided-versus-symmetric attention points at matched
   topology/kappa; loss versus achieved `n_h(0.01)`; and OL1
   conflict/projection frequencies. Because threshold+L1 is not run, the
   figure cannot claim OL1 beats L1 inside the selected threshold topology.
4. **Main results table:** complete final recipes, validation loss, spillover
   vector, `R_model`, paired changes, seeds, and uncertainty.

LR sweeps, complete lambda/kappa grids, extended OL1 mechanism diagnostics,
adverse or failed conditions, and selection/confirmation comparisons go to the
appendix.

The complete selection partition is evaluated at update 1, every 191 updates,
and the final update. One final checkpoint is saved. After all choices and
analyses are frozen, the confirmation partition is evaluated once from each
named saved checkpoint; it cannot trigger reselection.

## Decisions and implementation required before launch

1. Freeze the OL1 `step_budget`.
2. Approve the complete-block-pass wrap and headline seed policy.
3. Add and test the cosine scheduler and global gradient clipping.
4. Add per-group threshold forms: one-sided FFN plus independently
   one-sided/symmetric attention under one kappa, including the methods-contract
   amendment.
5. Implement and validate the `R_model^max` artifact contract for each
   model/topology/workload.
6. Verify/pin all model, tokenizer, dataset, cache, partition, batch, and
   license identities, including the proposed 14M inputs.
7. Add the exact validation cadence, generic saved-checkpoint confirmation,
   pooled RMS/histograms, and all-boundary OL1 counters.

Stage B1 is the dominant screen: 56 conceptual cells represented by 50 unique
training runs, approximately 74.59B processed tokens at seed 0. Before any
launch, each model/workload receives a 600-second production-shaped calibration
and separate setup, validation,
diagnostic, and checkpoint timing. The calibrated ETC and uncertainty return
for explicit approval.

Claim scope is deliberately narrow: full lambda/kappa curves are single-seed
exploration. A winner with a nonidentity attention threshold may be described
as a selected multi-site recipe containing attention thresholding, but the
improvement cannot be attributed specifically to attention without an
FFN-threshold+OL1 control. The introduction's 12B/`T = 50,000` ceiling examples
require separate validation or removal. B1 is spillover-motivated, not selected
from the spillover results. Directional statements about lambda sensitivity or
one-sided-versus-symmetric behavior remain exploratory unless their complete
curves receive additional seeds.

The [detailed review draft](exp-plan-v0.md) contains the selection rules,
controls, evidence policy, and implementation blockers.
