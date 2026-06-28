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
- Gradient metrics: task gradient norm, pressure gradient norm, pressure/task ratio, task-pressure cosine, and conflict flag for pressure methods.
- Orthogonal metrics: post-Adam pressure update projection and trust-budget metrics for orthogonal methods.
- Runtime metrics: tokens/sec, wall seconds, peak GPU memory, weight norm, gradient norm, and final checkpoint size.
- Post-hoc clipping frontier: validation loss versus achieved exact-zero activation sparsity after clipping activations at thresholds `[0, 0.001, 0.003, 0.01, 0.03, 0.05]`.

## Result Table

| Config | Role | Params | Final val | Final train | Near-zero <=0.01 | Near-zero <=0.03 | Tokens/s | Run |
| ------ | ---- | ------ | --------- | ----------- | ---------------- | ---------------- | -------- | --- |
| `12` | adamw | monitor only | 7.0200 | 7.1511 | 6.47% | 19.37% | 138567 | `002-20260628-102105-5f5bf3b7` |
| `13` | ricker_naive | w=0.03, c=0.05, s=0.05 | 7.0353 | 7.1647 | 11.68% | 31.78% | 71899 | `001-20260627-200725-8fb49833` |
| `14` | ricker_naive | w=0.1, c=0.05, s=0.05 | 7.0848 | 7.2157 | 34.05% | 58.08% | 71980 | `001-20260627-203833-3ba1089c` |
| `15` | ricker_naive | w=0.3, c=0.05, s=0.05 | 7.1900 | 7.3224 | 58.10% | 71.28% | 71987 | `001-20260627-210938-6c452fee` |
| `16` | ricker_naive | w=0.1, c=0.02, s=0.02 | 7.0852 | 7.2129 | 20.94% | 35.41% | 71952 | `001-20260627-214043-cc7ab328` |
| `17` | ricker_naive | w=0.1, c=0.1, s=0.1 | 7.0593 | 7.1889 | 29.17% | 65.51% | 71148 | `001-20260627-221149-48004d43` |
| `18` | ricker_naive | w=0.1, c=0.05, s=0.025 | 7.0845 | 7.2146 | 44.37% | 65.92% | 71499 | `001-20260627-224317-7eea761e` |
| `19` | ricker_naive | w=0.1, c=0.05, s=0.1 | 7.1015 | 7.2294 | 48.60% | 77.51% | 72701 | `001-20260627-231435-41bbbd2a` |
| `20` | orthogonal_ricker | w=0.03, c=0.05, s=0.05 | 7.0321 | 7.1613 | 11.94% | 31.94% | 104580 | `001-20260627-234522-278d047a` |
| `21` | orthogonal_ricker | w=0.1, c=0.05, s=0.05 | 7.0480 | 7.1771 | 29.83% | 54.96% | 104520 | `001-20260628-000646-982f2a63` |
| `22` | orthogonal_ricker | w=0.3, c=0.05, s=0.05 | 7.0656 | 7.1944 | 47.37% | 65.01% | 104379 | `001-20260628-002811-8bd1eb51` |
| `23` | orthogonal_ricker | w=0.1, c=0.02, s=0.02 | 7.0457 | 7.1755 | 16.10% | 30.65% | 103408 | `001-20260628-004937-53101a2f` |
| `24` | orthogonal_ricker | w=0.1, c=0.1, s=0.1 | 7.0399 | 7.1668 | 27.74% | 63.12% | 102917 | `001-20260628-011116-01691995` |
| `25` | orthogonal_ricker | w=0.1, c=0.05, s=0.025 | 7.0521 | 7.1792 | 39.05% | 62.26% | 102956 | `001-20260628-013301-a51b2cc5` |
| `26` | orthogonal_ricker | w=0.1, c=0.05, s=0.1 | 7.0539 | 7.1802 | 41.32% | 71.36% | 102961 | `001-20260628-015445-d63a624d` |
| `27` | l1_naive | w=0.05 | 7.0151 | 7.1466 | 7.25% | 21.58% | 77798 | `001-20260628-021630-9f9b1431` |
| `28` | l1_naive | w=0.15 | 7.0104 | 7.1409 | 9.05% | 26.65% | 77874 | `001-20260628-024516-e0118629` |
| `29` | l1_naive | w=0.5 | 7.0330 | 7.1622 | 20.91% | 53.68% | 77871 | `001-20260628-031400-644f63a4` |
| `30` | l1_naive | w=1 | 7.0631 | 7.1930 | 41.22% | 76.83% | 77895 | `001-20260628-034244-7abc40ca` |
| `31` | orthogonal_l1 | w=0.05 | 7.0137 | 7.1450 | 7.69% | 22.83% | 109025 | `001-20260628-041128-45348673` |
| `32` | orthogonal_l1 | w=0.15 | 7.0100 | 7.1405 | 10.05% | 29.20% | 109291 | `001-20260628-043159-ebc81b5e` |
| `33` | orthogonal_l1 | w=0.5 | 7.0213 | 7.1512 | 21.90% | 55.25% | 109368 | `001-20260628-045228-dc0c9c81` |
| `34` | orthogonal_l1 | w=1 | 7.0348 | 7.1608 | 34.89% | 71.41% | 109217 | `001-20260628-051256-dc0b654d` |

## Main Observations

1. Loss curves are similar across selected methods, but not identical. The early language-modeling optimization works in all cases, while pressure strength controls the final loss penalty.
2. AdamW monitor-only baseline has final validation loss `7.0200`, near-zero mass `6.47%` at threshold `0.01`, and `19.37%` at threshold `0.03`.
3. Ricker pressure strongly increases near-zero mass. Naive Ricker is costly at moderate or high pressure. Orthogonal Ricker consistently reduces that validation-loss cost at matched nominal settings.
4. At `w=0.1, c=0.05, s=0.05`, naive Ricker ends at validation loss `7.0848`; orthogonal Ricker ends at `7.0480`.
5. At `w=0.3, c=0.05, s=0.05`, naive Ricker ends at validation loss `7.1900`; orthogonal Ricker ends at `7.0656`.
6. L1 pressure is the strongest early candidate in this screen. `l1_naive w=0.15` and `orthogonal_l1 w=0.15` slightly improve final validation loss relative to AdamW while increasing near-zero mass.
7. Orthogonal methods are faster than the corresponding naive pressure methods in this implementation because they avoid the same naive augmented-loss backward path, but runtime was not the primary controlled variable in this screen.

## Figures

- `figures/05-pythia-14m-pressure-fixed-2048-summary.pdf`: all-run loss versus near-zero activation mass tradeoff.
- `figures/06-pythia-14m-pressure-fixed-2048-learning-curves.pdf`: representative learning curves for AdamW plus the best validation-loss run per pressure family.
- `figures/07-pythia-14m-pressure-fixed-2048-clipping-frontiers.pdf`: post-hoc clipping frontiers for the representative runs.

## Interpretation Boundary

These results are useful for planning the full ablation, but they are not sufficient for a top-tier paper claim by themselves.

Required next evidence:

- Repeat key candidates over multiple seeds.
- Extend the token budget beyond 134M tokens.
- Use a larger validation sample or full deterministic validation pass for final reported metrics.
- Add a scale ladder only after the 14M setting is stable.
- Separate training-induced near-zero mass from post-hoc clipping-induced exact sparsity in every table and figure.
