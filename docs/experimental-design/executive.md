# Experimental Plan — Executive Brief

> **Status:** proposal for advisor review, not launch authority.

## Aim

The study tests three hypotheses:

1. L1 pressure at the FFN hidden activation `h` is accompanied by a nonlocal
   activation response—**sparsity spillover**—at untargeted FFN and attention
   sites.
2. Threshold placement changes the share of model-wide logical products with
   an exactly-zero activation operand.
3. A selected multi-site threshold+OL1 recipe improves the
   validation-loss/logical-opportunity frontier and transports from
   Pythia-14M to Pythia-70M and Pythia-410M.

These are questions, not assumed findings. `R_model` is a logical opportunity,
not measured compute reduction or runtime speedup.

## Shared Recipe

All runs pretrain randomly initialized Pythia models on MiniPile. Released
checkpoint weights are not loaded.

| Model | Global batch | Peak-LR grid |
| --- | ---: | ---: |
| Pythia-14M | 262,144 tokens/update | `{5e-4, 1e-3, 2e-3, 4e-3}` |
| Pythia-70M | 262,144 tokens/update | `{5e-4, 1e-3, 2e-3}` |
| Pythia-410M | 262,144 tokens/update | `{1.5e-4, 3e-4, 6e-4}` |

Fixed settings: AdamW, betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay
`0.1`, global gradient clipping `1.0`, zero dropout, sequence length `2,048`,
and BF16 computation with FP32 parameters/state. The LR warms linearly for 1%
of updates, then cosine-decays to 10% of peak. Only peak LR is tuned.

A1 and C1 learning-rate screens use 1,526 updates (400,031,744 input tokens).
A2/A3, B1/B2, and C2/C3 retain one complete-block MiniPile pass plus a
74-block fixed-batch wrap: 5,691 updates and 1,491,861,504 input tokens. The A1
decision is specific to its 400M-token horizon.

## Sites and Thresholds

| Alias | Downstream operand |
| --- | --- |
| `a` | Attention-branch normalized input to QKV |
| `m` | FFN-branch normalized input to `W1` |
| `h` | FFN activation output to `W2` |
| `q_post`, `k_post` | Post-RoPE operands of `QK^T` |
| `v` | Value operand of the attention probability-value product |

Each alias denotes the post-intervention operand. At `h`, ReLU or a threshold
replaces GELU; it is not applied after GELU. Pressure always targets only `h`.
Stage labels name experiments; topology IDs name active site sets.

B1 compares topology `A1-H`; topology `A2`; topology `A3`; the three `A4-Q/K/V`
variants; `A5-QK-POST`; and `A6-POST`. Q/K thresholds are POST-RoPE. One global
`kappa` is used: FFN sites are one-sided, while attention sites are one-sided
or symmetric. The conceptual matrix has 56 cells but only 50 physical runs,
because six symmetric-attention `kappa = 0` cells reuse the same functional
anchor.

## Experiments

| Stage | Experiment | Question/decision | New physical runs* |
| --- | --- | --- | ---: |
| A1 | 14M `A0`, four peak LRs, seed 0 | Freeze the lowest final selection-loss LR; `4e-3` is a single upper-boundary extension. | 4 |
| A2 | ReLU `A1-H`, h-only L1 lambda grid | Map targeted and untargeted responses; confirm the fixed ReLU-only versus L1 contrast with three seeds. | 10 |
| A3 | Same grid with OL1 | Compare loss versus achieved near-zero mass and projection behavior; freeze one matched nonzero lambda for B2. | 5 |
| B1 | Placement × kappa × attention-threshold form, no pressure | Map and select the validation-loss/`R_model` family frontier. | 50 |
| B2 | Apply OL1 at the frozen lambda to the selected B1 family | Select the combined frontier and one quality-oriented winner. | 12 or 14 |
| C1 | 70M/410M LR screens at 400M tokens | Freeze one peak LR independently per size. | 6 |
| C2 | Repeat the L1 response at 70M/410M | Test the same fixed spillover contrast without retuning lambda. | 22 |
| C3 | Transport the B2 family, kappa cohort, lambda, and OL1 budget | Test frontier persistence without intervention retuning. | `18 + 4K` or `22 + 4K` |

*Counts exclude reused cases, calibration, and infrastructure retries. The
50-run B1 count assumes the predeclared kappa-zero equivalence passes its
required numerical/diagnostic review. The lower B2/C3 count applies when
`lambda_B2 = 1`; `K` is the number of frozen B2 frontier kappas. If all stages
proceed, the catalog contains 131–149 new pretraining runs (`K` from 1 to 4).

Full grids are exploratory seed-0 evidence. The fixed spillover contrast and
the selected six-component winner comparison use seeds 0, 1, and 2. A later
stage reuses any matching earlier condition/seed; it never creates a second
config for the same physical case.

## Reported Evidence

At every named site, report exact-zero mass, near-zero mass at `0.01`
(`0.1` sensitivity), pooled RMS, layerwise distributions, and paired
validation loss. Frontier conditions additionally report validated `R_model`
and topology-specific `R_model^max` at `T = 2,048`.

The minimum paper package is:

1. **Sparsity spillover:** lambda responses, the three-seed fixed contrast,
   site/layer distributions, and scale comparison.
2. **Logical opportunity:** operation-level `R_model` and model/topology
   ceilings.
3. **Intervention and mechanism:** six matched components, B1/C3 frontiers,
   paired threshold-form comparisons, and OL1 conflict/projection summaries.
4. **Results table:** complete recipes, paired changes, seeds, uncertainty,
   and evidence status.

The plan does not include threshold+L1 or a matched FFN-threshold+OL1 control.
It therefore cannot claim OL1 beats L1 inside the threshold topology or
attribute a combined improvement specifically to attention. Complete
lambda/threshold-form curves remain single-seed directional evidence.

The modular design, exact cases, reuse rules, and open blockers are indexed in
[`README.md`](README.md).
