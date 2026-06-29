# Fixed-step Activation-pressure Screen

Date: 2026-06-28

Purpose: run a fixed-token Pythia-14M MiniPile screen to understand whether activation-pressure methods change early pretraining loss, MLP hidden near-zero mass, and post-hoc clipping frontiers enough to justify a larger ablation.

This is not a full paper ablation. It is a planning screen: one seed, one model scale, one dataset cache, one fixed step budget, and short validation passes.

## Setup

- Architecture: Pythia-14M via `EleutherAI/pythia-14m-deduped`.
- Initialization: random. No released Pythia checkpoint weights are loaded.
- Dataset: local tokenized `JeanKaddour/minipile` cache.
- Train cache: 1,491,711,416 tokens.
- Validation cache: 693,668 tokens.
- Fixed budget: 2,048 optimizer steps.
- Tokens per step: 65,536.
- Tokens per run: 134,217,728.
- Estimated pass over MiniPile token cache: 0.08998 epochs.
- Optimizer: AdamW.
- Precision: FP32 parameters, bf16 autocast.
- Validation frequency: every 250 steps plus step 1 and the final step.
- Validation size: 8 batches per evaluation, 65,536 validation tokens.
- Activation site: Pythia MLP hidden activations, hooked at `gpt_neox.layers.N.mlp.act`.

## What Was Measured

- Train loss: causal language-modeling loss on sampled training token blocks, logged every 50 steps.
- Validation loss: causal language-modeling loss on validation token blocks, logged every 250 steps plus step 1 and final step.
- Near-zero mass: fraction of captured MLP hidden activations with absolute value below thresholds, especially `0.01` and `0.03`.
- Pressure loss: unweighted auxiliary activation-pressure objective for Ricker and L1 methods.
- Pressure semantics: L1 and Ricker pressures are applied to scalar activation elements `A_l[b,t,j]` and then averaged into one loss. They are not structured channel pressures and do not directly force the same MLP hidden dimensions to be inactive for every token.
- Gradient metrics: task gradient norm, pressure gradient norm, pressure/task ratio, task-pressure cosine, and conflict flag for pressure methods.
- Orthogonal metrics: post-Adam pressure update projection and trust-budget metrics for orthogonal methods.
- Runtime metrics: tokens/sec, wall seconds, peak GPU memory, weight norm, gradient norm, and final checkpoint size.
- Post-hoc clipping frontier: validation loss versus achieved exact-zero activation sparsity after clipping activations at thresholds `[0, 0.001, 0.003, 0.01, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3]`.
- Exact-zero sparsity semantics: reported clipping sparsity is also elementwise over captured activation entries. For example, `80%` exact-zero activation sparsity does not imply that a fixed `80%` of the 512 MLP hidden channels are zero for all batch items and token positions.

## Covered Parameter Grid

Source of truth: `src/paper_exp/sweeps.py`, sweep `pressure_fixed_step_v1`.

Shared settings:

- Seed: `0`.
- Activation site: `mlp_hiddens`.
- Pressure log thresholds: `[0.0, 0.001, 0.003, 0.01, 0.03]`.
- Orthogonal methods use `step_budget: 0.5` and `eps: 1e-12`.

Complete Ricker grid:

| Regime | Params | RN config | OR config |
| ------ | ------ | --------- | --------- |
| Initial screen | `w=0.03, c=0.05, s=0.05` | `13` | `20` |
| Initial screen | `w=0.1, c=0.05, s=0.05` | `14` | `21` |
| Initial screen and high-pressure grid overlap | `w=0.3, c=0.05, s=0.05` | `15` | `22` |
| Initial screen | `w=0.1, c=0.02, s=0.02` | `16` | `23` |
| Initial screen | `w=0.1, c=0.1, s=0.1` | `17` | `24` |
| Initial screen | `w=0.1, c=0.05, s=0.025` | `18` | `25` |
| Initial screen | `w=0.1, c=0.05, s=0.1` | `19` | `26` |
| High-pressure/wide-Ricker expansion | `w=0.3, c=0.1, s=0.1` | `35` | `40` |
| High-pressure/wide-Ricker expansion | `w=0.3, c=0.5, s=0.5` | `36` | `41` |
| High-pressure/wide-Ricker expansion | `w=1.0, c=0.05, s=0.05` | `37` | `42` |
| High-pressure/wide-Ricker expansion | `w=1.0, c=0.1, s=0.1` | `38` | `43` |
| High-pressure/wide-Ricker expansion | `w=1.0, c=0.5, s=0.5` | `39` | `44` |

Complete L1 grid:

| Regime | Params | L1N config | OL1 config |
| ------ | ------ | ---------- | ---------- |
| Initial screen | `w=0.05` | `27` | `31` |
| Initial screen | `w=0.15` | `28` | `32` |
| Initial screen | `w=0.5` | `29` | `33` |
| Initial screen | `w=1.0` | `30` | `34` |
| High-pressure L1 expansion | `w=2.0` | `45` | `47` |
| High-pressure L1 expansion | `w=5.0` | `46` | `48` |

## Result Table

The `Val@0.4`, `Val@0.6`, and `Val@0.8` columns come from the latest matched post-hoc clipping sweeps. Each cell is `validation loss @ achieved exact-zero activation sparsity`, using the nearest achieved sparsity point to the target and no interpolation.

| Config | Role | Params | Final val | Val@0.4 | Val@0.6 | Val@0.8 | Final train | Near-zero <=0.01 | Near-zero <=0.03 | Tokens/s | Run |
| ------ | ---- | ------ | --------- | ------- | ------- | ------- | ----------- | ---------------- | ---------------- | -------- | --- |
| `12` | adamw | monitor only | 7.0200 | 7.1118 @ 0.468 | 7.2111 @ 0.604 | 7.5016 @ 0.813 | 7.1511 | 6.47% | 19.37% | 138567 | `002-20260628-102105-5f5bf3b7` |
| `13` | ricker_naive | w=0.03, c=0.05, s=0.05 | 7.0353 | 7.0727 @ 0.345 | 7.1417 @ 0.599 | 7.5206 @ 0.813 | 7.1647 | 11.68% | 31.78% | 71899 | `001-20260627-200725-8fb49833` |
| `14` | ricker_naive | w=0.1, c=0.05, s=0.05 | 7.0848 | 7.1141 @ 0.379 | 7.1400 @ 0.597 | 7.5449 @ 0.800 | 7.2157 | 34.05% | 58.08% | 71980 | `001-20260627-203833-3ba1089c` |
| `15` | ricker_naive | w=0.3, c=0.05, s=0.05 | 7.1900 | 7.2171 @ 0.274 | 7.2272 @ 0.615 | 7.4089 @ 0.798 | 7.3224 | 58.10% | 71.28% | 71987 | `001-20260627-210938-6c452fee` |
| `16` | ricker_naive | w=0.1, c=0.02, s=0.02 | 7.0852 | 7.1257 @ 0.429 | 7.2801 @ 0.650 | 7.5267 @ 0.833 | 7.2129 | 20.94% | 35.41% | 71952 | `001-20260627-214043-cc7ab328` |
| `17` | ricker_naive | w=0.1, c=0.1, s=0.1 | 7.0593 | 7.0910 @ 0.320 | 7.2065 @ 0.698 | 7.3865 @ 0.808 | 7.1889 | 29.17% | 65.51% | 71148 | `001-20260627-221149-48004d43` |
| `18` | ricker_naive | w=0.1, c=0.05, s=0.025 | 7.0845 | 7.1156 @ 0.482 | 7.1433 @ 0.670 | 7.2387 @ 0.776 | 7.2146 | 44.37% | 65.92% | 71499 | `001-20260627-224317-7eea761e` |
| `19` | ricker_naive | w=0.1, c=0.05, s=0.1 | 7.1015 | 7.1443 @ 0.531 | 7.1443 @ 0.531 | 7.2977 @ 0.794 | 7.2294 | 48.60% | 77.51% | 72701 | `001-20260627-231435-41bbbd2a` |
| `20` | orthogonal_ricker | w=0.03, c=0.05, s=0.05 | 7.0321 | 7.0700 @ 0.348 | 7.1383 @ 0.595 | 7.5188 @ 0.800 | 7.1613 | 11.94% | 31.94% | 104580 | `001-20260627-234522-278d047a` |
| `21` | orthogonal_ricker | w=0.1, c=0.05, s=0.05 | 7.0480 | 7.0813 @ 0.332 | 7.1093 @ 0.569 | 7.5271 @ 0.799 | 7.1771 | 29.83% | 54.96% | 104520 | `001-20260628-000646-982f2a63` |
| `22` | orthogonal_ricker | w=0.3, c=0.05, s=0.05 | 7.0656 | 7.1079 @ 0.507 | 7.1538 @ 0.661 | 7.5621 @ 0.820 | 7.1944 | 47.37% | 65.01% | 104379 | `001-20260628-002811-8bd1eb51` |
| `23` | orthogonal_ricker | w=0.1, c=0.02, s=0.02 | 7.0457 | 7.0891 @ 0.395 | 7.2495 @ 0.619 | 7.5190 @ 0.812 | 7.1755 | 16.10% | 30.65% | 103408 | `001-20260628-004937-53101a2f` |
| `24` | orthogonal_ricker | w=0.1, c=0.1, s=0.1 | 7.0399 | 7.0736 @ 0.305 | 7.1654 @ 0.669 | 7.3196 @ 0.787 | 7.1668 | 27.74% | 63.12% | 102917 | `001-20260628-011116-01691995` |
| `25` | orthogonal_ricker | w=0.1, c=0.05, s=0.025 | 7.0521 | 7.0863 @ 0.426 | 7.1142 @ 0.638 | 7.3668 @ 0.820 | 7.1792 | 39.05% | 62.26% | 102956 | `001-20260628-013301-a51b2cc5` |
| `26` | orthogonal_ricker | w=0.1, c=0.05, s=0.1 | 7.0539 | 7.0927 @ 0.453 | 7.2057 @ 0.734 | 7.2920 @ 0.797 | 7.1802 | 41.32% | 71.36% | 102961 | `001-20260628-015445-d63a624d` |
| `27` | l1_naive | w=0.05 | 7.0151 | 7.0655 @ 0.359 | 7.2435 @ 0.664 | 7.5725 @ 0.863 | 7.1466 | 7.25% | 21.58% | 77798 | `001-20260628-021630-9f9b1431` |
| `28` | l1_naive | w=0.15 | 7.0104 | 7.0770 @ 0.449 | 7.1927 @ 0.633 | 7.4258 @ 0.776 | 7.1409 | 9.05% | 26.65% | 77874 | `001-20260628-024516-e0118629` |
| `29` | l1_naive | w=0.5 | 7.0330 | 7.0659 @ 0.228 | 7.1156 @ 0.577 | 7.3289 @ 0.766 | 7.1622 | 20.91% | 53.68% | 77871 | `001-20260628-031400-644f63a4` |
| `30` | l1_naive | w=1 | 7.0631 | 7.1008 @ 0.455 | 7.1008 @ 0.455 | 7.3707 @ 0.805 | 7.1930 | 41.22% | 76.83% | 77895 | `001-20260628-034244-7abc40ca` |
| `31` | orthogonal_l1 | w=0.05 | 7.0137 | 7.0667 @ 0.381 | 7.1314 @ 0.548 | 7.6115 @ 0.880 | 7.1450 | 7.69% | 22.83% | 109025 | `001-20260628-041128-45348673` |
| `32` | orthogonal_l1 | w=0.15 | 7.0100 | 7.0901 @ 0.490 | 7.2435 @ 0.676 | 7.5070 @ 0.811 | 7.1405 | 10.05% | 29.20% | 109291 | `001-20260628-043159-ebc81b5e` |
| `33` | orthogonal_l1 | w=0.5 | 7.0213 | 7.0553 @ 0.239 | 7.1178 @ 0.593 | 7.3518 @ 0.776 | 7.1512 | 21.90% | 55.25% | 109368 | `001-20260628-045228-dc0c9c81` |
| `34` | orthogonal_l1 | w=1 | 7.0348 | 7.0724 @ 0.386 | 7.2993 @ 0.751 | 7.2993 @ 0.751 | 7.1608 | 34.89% | 71.41% | 109217 | `001-20260628-051256-dc0b654d` |
| `35` | ricker_naive | w=0.3, c=0.1, s=0.1 | 7.0973 | 7.1520 @ 0.565 | 7.1520 @ 0.565 | 7.3886 @ 0.844 | 7.2238 | 51.40% | 82.39% | 70480 | `001-20260628-170230-11e52ca4` |
| `36` | ricker_naive | w=0.3, c=0.5, s=0.5 | 7.0116 | 7.0572 @ 0.337 | 7.1368 @ 0.548 | 7.4566 @ 0.762 | 7.1408 | 10.96% | 32.07% | 71855 | `001-20260628-173415-090fa785` |
| `37` | ricker_naive | w=1, c=0.05, s=0.05 | 7.6130 | 7.6379 @ 0.498 | 7.6379 @ 0.498 | 7.6405 @ 0.799 | 7.7287 | 72.60% | 76.62% | 71913 | `001-20260628-180524-782de820` |
| `38` | ricker_naive | w=1, c=0.1, s=0.1 | 7.2595 | 7.2901 @ 0.349 | 7.3561 @ 0.796 | 7.3561 @ 0.796 | 7.3870 | 75.79% | 91.80% | 71910 | `001-20260628-183631-3151faa7` |
| `39` | ricker_naive | w=1, c=0.5, s=0.5 | 7.0406 | 7.0727 @ 0.229 | 7.1679 @ 0.618 | 7.5479 @ 0.832 | 7.1707 | 21.36% | 56.61% | 71866 | `001-20260628-190738-61461b4a` |
| `40` | orthogonal_ricker | w=0.3, c=0.1, s=0.1 | 7.0538 | 7.0987 @ 0.476 | 7.0987 @ 0.476 | 7.3162 @ 0.785 | 7.1803 | 43.64% | 76.39% | 103309 | `001-20260628-193846-9e8571a8` |
| `41` | orthogonal_ricker | w=0.3, c=0.5, s=0.5 | 7.0090 | 7.0566 @ 0.365 | 7.1561 @ 0.585 | 7.5095 @ 0.790 | 7.1378 | 11.93% | 34.67% | 99675 | `001-20260628-200026-65c544f1` |
| `42` | orthogonal_ricker | w=1, c=0.05, s=0.05 | 7.0750 | 7.1050 @ 0.286 | 7.1299 @ 0.585 | 7.4082 @ 0.791 | 7.2073 | 55.64% | 70.07% | 101275 | `001-20260628-202254-e5fdabb6` |
| `43` | orthogonal_ricker | w=1, c=0.1, s=0.1 | 7.0655 | 7.0947 @ 0.239 | 7.1505 @ 0.620 | 7.3967 @ 0.834 | 7.1962 | 57.65% | 82.05% | 102834 | `001-20260628-204500-3bf30cb8` |
| `44` | orthogonal_ricker | w=1, c=0.5, s=0.5 | 7.0240 | 7.0566 @ 0.230 | 7.1673 @ 0.613 | 7.5407 @ 0.821 | 7.1517 | 21.19% | 56.11% | 103365 | `001-20260628-210645-99560f7d` |
| `45` | l1_naive | w=2 | 7.0978 | 7.1267 @ 0.286 | 7.1788 @ 0.684 | 7.4848 @ 0.888 | 7.2259 | 62.84% | 87.10% | 77468 | `001-20260628-212825-0b3a1f9c` |
| `46` | l1_naive | w=5 | 7.4313 | 7.4603 @ 0.552 | 7.4603 @ 0.552 | 7.4812 @ 0.876 | 7.5610 | 86.02% | 94.31% | 77462 | `001-20260628-215718-0bd5b002` |
| `47` | orthogonal_l1 | w=2 | 7.0458 | 7.1030 @ 0.532 | 7.1030 @ 0.532 | 7.4247 @ 0.820 | 7.1730 | 48.74% | 79.93% | 109941 | `001-20260628-222611-c8c001f6` |
| `48` | orthogonal_l1 | w=5 | 7.0554 | 7.0851 @ 0.298 | 7.1710 @ 0.674 | 7.4713 @ 0.864 | 7.1845 | 63.15% | 85.07% | 109794 | `001-20260628-224633-5ef1a335` |

## Main Observations

1. Loss curves are similar across selected methods, but not identical. The early language-modeling optimization works in all cases, while pressure strength controls the final loss penalty.
2. AdamW monitor-only baseline has final validation loss `7.0200`, near-zero mass `6.47%` at threshold `0.01`, and `19.37%` at threshold `0.03`.
3. Ricker pressure strongly increases near-zero mass. Naive Ricker is costly at moderate or high pressure. Orthogonal Ricker consistently reduces that validation-loss cost at matched nominal settings.
4. At `w=0.1, c=0.05, s=0.05`, naive Ricker ends at validation loss `7.0848`; orthogonal Ricker ends at `7.0480`.
5. At `w=0.3, c=0.05, s=0.05`, naive Ricker ends at validation loss `7.1900`; orthogonal Ricker ends at `7.0656`.
6. L1 pressure is the strongest early candidate in this screen. `l1_naive w=0.15` and `orthogonal_l1 w=0.15` slightly improve final validation loss relative to AdamW while increasing near-zero mass.
7. Orthogonal methods are faster than the corresponding naive pressure methods in this implementation because they avoid the same naive augmented-loss backward path, but runtime was not the primary controlled variable in this screen.
8. The high-pressure/wide-Ricker expansion keeps the same 2,048-step budget. The requested `w=0.3, c=s=0.05` setting was already covered by configs `15` and `22`; configs `35`-`48` add the missing high-pressure and wide-Ricker points.
9. Wider Ricker pressure at `c=s=0.5` has lower early validation-loss cost than narrow high-pressure Ricker in this one-seed screen. Examples: RN `w=0.3, c=s=0.5` ends at validation loss `7.0116`; OR `w=0.3, c=s=0.5` ends at `7.0090`.
10. High-pressure narrow Ricker and high-pressure L1 can force much larger near-zero activation mass but with a clear loss penalty. Examples: RN `w=1, c=s=0.05` ends at validation loss `7.6130`; L1N `w=5` ends at `7.4313`.
11. Orthogonal high-pressure variants reduce the loss penalty at matched nominal settings in this screen. Examples: RN `w=1, c=s=0.1` ends at `7.2595`, while OR `w=1, c=s=0.1` ends at `7.0655`; L1N `w=5` ends at `7.4313`, while OL1 `w=5` ends at `7.0554`.

## Figures

- `figures/05-pythia-14m-pressure-fixed-2048-summary.pdf`: all-run loss versus near-zero activation mass tradeoff.
- `figures/06-pythia-14m-pressure-fixed-2048-learning-curves.pdf`: representative learning curves for AdamW plus the best validation-loss run per pressure family.
- `figures/07-pythia-14m-pressure-fixed-2048-clipping-frontiers.pdf`: post-hoc clipping frontiers for the representative runs, including the 80-90% exact-zero sparsity region.
- `figures/17-pythia-14m-pressure-fixed-2048-high-pressure-rn-learning-curves.pdf`: learning curves for AdamW plus RN configs `35`-`39`.
- `figures/18-pythia-14m-pressure-fixed-2048-high-pressure-or-learning-curves.pdf`: learning curves for AdamW plus OR configs `40`-`44`.
- `figures/19-pythia-14m-pressure-fixed-2048-high-pressure-l1-learning-curves.pdf`: learning curves for AdamW plus L1N/OL1 configs `45`-`48`.
- `figures/20-pythia-14m-pressure-fixed-2048-high-pressure-clipping-frontiers.pdf`: post-hoc clipping frontiers for AdamW plus configs `35`-`48`.

## Interpretation Boundary

These results are useful for planning the full ablation, but they are not sufficient for a top-tier paper claim by themselves.

Required next evidence:

- Repeat key candidates over multiple seeds.
- Extend the token budget beyond 134M tokens.
- Use a larger validation sample or full deterministic validation pass for final reported metrics.
- Add a scale ladder only after the 14M setting is stable.
- Separate training-induced near-zero mass from post-hoc clipping-induced exact sparsity in every table and figure.
