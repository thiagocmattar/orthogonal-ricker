# Methods

## Research Question

TODO

## Baseline

Pythia-14M architecture pretraining on MiniPile is the first concrete baseline path.

Architecture source: `EleutherAI/pythia-14m-deduped`.

Initialization: random.

Dataset: `JeanKaddour/minipile`.

This is a low-resource MiniPile pretraining baseline, not a faithful reproduction of the original Pythia 300B-token training run.

Important distinction: this repo uses the Pythia architecture/config, not the released Pythia checkpoint weights. Runs that load checkpoint weights are continuation/fine-tuning runs and should not be logged as pretraining baselines.

## Proposed Method

TODO

## Evaluation Setup

TODO: calibrate throughput with a short local tokenized MiniPile run from random initialization before choosing the full baseline budget.

For paper runs, use local cached/tokenized MiniPile data rather than streaming so token counts, ordering, and runtime are auditable.

Training/calibration tracking:

- Training samples random contiguous token blocks with replacement from the local token cache.
- `training.log_every` controls train event logging.
- `validation.eval_every_steps` controls validation loss frequency.
- Validation uses a separately tokenized validation split cache when `validation.enabled` is true.
- Current tracked summaries include train loss, validation loss, throughput, wall-clock time, validation wall-clock time, estimated epoch, learning rate, gradient norm, weight norm, peak GPU memory, and final checkpoint metadata.
- Calibration and baseline runs can save final model weights under `checkpoints/final/` when `checkpoint.save_final` is true.
- Activation-pressure runs log task loss and pressure loss separately, task/pressure gradient interference metrics, near-zero activation mass at configured thresholds, and Adam-step projection metrics for orthogonal methods.

Current cleanup note:

- Pretraining configs now require `model.initialization: random`.
- The harness constructs the model with `AutoModelForCausalLM.from_config(...)`, not `from_pretrained(...)`.
- Previous Pythia runs that loaded released checkpoint weights were removed from the paper map and experiment log.
- The first random-initialized full-MiniPile checkpoint has now been run with FP32 parameters and bf16 autocast.

First valid full-MiniPile random-init checkpoint:

- Config: `configs/03-pythia-14m-minipile-random-full-10min.yaml`.
- Result: `results/03-pythia-14m-minipile-random-full-10min/003-20260627-142522-7fc1e76f/`.
- Figure: `figures/01-pythia-14m-minipile-random-full-10min-diagnostics.pdf`.
- Model: Pythia-14M architecture, random initialization, no checkpoint weights loaded.
- Parameter dtype: float32.
- Compute precision: bf16 autocast.
- Full train token cache: 1,000,000 documents, 1,491,711,416 tokens.
- Validation token cache: 500 documents, 693,668 tokens.
- Settings: micro-batch 4, gradient accumulation 8, block size 2048, learning rate 0.00003, 100 warmup steps.
- Completed 1,316 optimizer steps and 86,245,376 tokens in 600.3 training seconds.
- Estimated fraction of one MiniPile token pass: 0.0578 epochs.
- Observed throughput: 143,666 tokens/sec.
- Validation overhead: 11.6 seconds total; final configured validation pass took 1.64 seconds.
- Train loss moved from 10.8567 at step 1 to 7.6701 at the final logged step.
- Validation loss moved from 10.8504 at step 1 to 7.5450 at the final step.
- Peak allocated GPU memory: 5,997.0 MB; peak reserved GPU memory: 7,428.0 MB.
- Final checkpoint size: 53.67 MB.
- Note: an earlier run in the same result folder (`002-20260627-141159-10a3e24a`) went non-finite because the random model parameters were float16. The harness now forces FP32 parameters for random initialization and aborts on non-finite losses.

Activation pressure implementation:

- Pressure site support includes `mlp_hiddens`, `attention_outputs`, and `residual_streams`; configs must list pressure sites explicitly.
- `mlp_hiddens` captures post-activation MLP hidden tensors at `gpt_neox.layers.N.mlp.act`.
- `attention_outputs` captures the first tensor output from `gpt_neox.layers.N.attention` before residual addition.
- `residual_streams` captures block inputs `H_l` with a pre-hook on `gpt_neox.layers.N`.
- Earlier pressure configs target only `mlp_hiddens`. Configs `65` and `66` are the first full-pass training runs that apply pressure to all three configured activation sites.
- All-site post-hoc clipping frontiers are evaluation sweeps over `mlp_hiddens`, `attention_outputs`, and `residual_streams`.
- Pressure semantics: L1 and Ricker score scalar activation elements `A_l[b,t,j]` and average those scores into one pressure loss. Current pressure and sparsity metrics are elementwise activation quantities, not structured channel sparsity over fixed MLP hidden dimensions.
- Naive methods: `ricker_naive` and `l1_naive` optimize `task_loss + weight * pressure_loss`.
- Orthogonal methods: `orthogonal_ricker` and `orthogonal_l1` compute task gradients and pressure gradients separately; AdamW steps on task gradients only; then a memoryless pressure correction is applied in AdamW step space.
- Orthogonal projection fires only when the pressure update direction conflicts with the AdamW task update direction. The trust budget caps the final pressure/task update ratio.
- Post-hoc clipping sweeps load a saved checkpoint and evaluate validation loss versus achieved exact-zero fraction for configured thresholds or quantiles.
- RMS-normalized post-hoc clipping uses `threshold = multiplier * RMS(A)` per captured activation tensor and forward pass. For the current `mlp_hiddens` site, `A` is one layer's MLP hidden activation tensor for the current validation batch.

Early method smoke evidence:

- Ricker naive smoke: `results/04-pythia-14m-minipile-ricker-naive-smoke/004-20260627-150149-42219854/`; final task loss 8.8505, final validation loss 9.0221, pressure gradient norm 0.7826, task/pressure gradient cosine -0.0079, aggregate `abs(a) <= 1e-2` mass 10.30%.
- L1 naive smoke: `results/05-pythia-14m-minipile-l1-naive-smoke/004-20260627-150208-e5734c8a/`; final task loss 8.8517, final validation loss 9.0234, pressure gradient norm 0.0964, task/pressure gradient cosine 0.0114, aggregate `abs(a) <= 1e-2` mass 5.20%.
- Orthogonal Ricker smoke: `results/06-pythia-14m-minipile-orthogonal-ricker-smoke/002-20260627-150025-e84f3b57/`; projection fired, raw pressure/task update ratio 2.9745, final ratio capped to 0.5, aggregate `abs(a) <= 1e-2` mass 8.10%.
- Orthogonal L1 smoke: `results/07-pythia-14m-minipile-orthogonal-l1-smoke/002-20260627-150050-5217fb84/`; projection fired, final pressure/task update ratio 0.2015, aggregate `abs(a) <= 1e-2` mass 5.60%.
- Post-hoc clipping smoke: `results/03-pythia-14m-minipile-random-full-10min-clipping-sweep/002-20260627-150326-6a61b34d/`; thresholds `[0, 0.001, 0.01, 0.03]`, 2 validation batches, achieved exact sparsity from 0.0% to 19.28%, validation loss from 7.6309 to 7.6324. Figure: `figures/02-pythia-14m-minipile-clipping-frontier-smoke.pdf`.

These smoke runs verify plumbing and metric emission. They are not paper-quality evidence because they use very short runs and tiny validation samples.

Short full-MiniPile pressure pretraining check:

- Configs: `configs/08-pythia-14m-minipile-ricker-naive-short.yaml` through `configs/11-pythia-14m-minipile-orthogonal-l1-short.yaml`.
- Purpose: quick method behavior check with random initialization, full local MiniPile token cache reuse, 180-second wall-clock caps, periodic validation, and final checkpoints.
- Figure: `figures/03-pythia-14m-pressure-short-learning-curves.pdf`.
- Ricker naive: 176 optimizer steps, 11,534,336 tokens, final train loss 10.0723, final validation loss 10.0623, final `abs(a) <= 0.01` MLP hidden mass 10.254%.
- L1 naive: 188 optimizer steps, 12,320,768 tokens, final train loss 10.0048, final validation loss 10.0075, final `abs(a) <= 0.01` MLP hidden mass 6.626%.
- Orthogonal Ricker: 259 optimizer steps, 16,973,824 tokens, final train loss 9.7525, final validation loss 9.6941, final `abs(a) <= 0.01` MLP hidden mass 12.167%, final pressure/task update ratio capped at 0.5.
- Orthogonal L1: 294 optimizer steps, 19,267,584 tokens, final train loss 9.6588, final validation loss 9.5532, final `abs(a) <= 0.01` MLP hidden mass 6.847%, final pressure/task update ratio 0.0486.

Matched post-hoc clipping check:

- Thresholds: `[0, 0.001, 0.003, 0.01, 0.03]`.
- Evaluation: 4 validation batches per threshold.
- Figure: `figures/04-pythia-14m-pressure-short-clipping-frontiers.pdf`.
- Max achieved exact sparsity at threshold `0.03`: Ricker naive 28.54%, L1 naive 19.83%, orthogonal Ricker 33.20%, orthogonal L1 20.45%.

Interpretation boundary: these 180-second checks validate implementation behavior and metric emission. They are not paper-quality evidence about final pretraining quality.

Fixed-step activation-pressure screen:

- Configs: `configs/12-pythia-14m-minipile-adamw-fixed-2048.yaml` through `configs/48-pythia-14m-minipile-orthogonal-l1-fixed-2048-w5.yaml`.
- Detailed readout: `docs/humans/02-fixed-step-pressure-screen.md`.
- Figures: `figures/05-pythia-14m-pressure-fixed-2048-summary.pdf`, `figures/06-pythia-14m-pressure-fixed-2048-learning-curves.pdf`, and `figures/07-pythia-14m-pressure-fixed-2048-clipping-frontiers.pdf`.
- Budget: 2,048 optimizer steps, 134,217,728 tokens per run, 0.08998 estimated MiniPile token-cache epochs.
- AdamW baseline was rerun as monitor-only with `activation_pressure.method: none`, so activation near-zero mass is measured without adding an auxiliary loss.
- AdamW monitor-only final validation loss was 7.0200; final `abs(a) <= 0.01` MLP hidden mass was 6.47%.
- Naive Ricker increases near-zero activation mass but can impose a large validation-loss cost as pressure increases.
- Orthogonal Ricker reduces that cost at matched nominal settings. Example: at `w=0.1, c=0.05, s=0.05`, naive Ricker final validation loss was 7.0848, while orthogonal Ricker final validation loss was 7.0480.
- L1 pressure is the strongest early candidate in this screen. `l1_naive w=0.15` reached final validation loss 7.0104; `orthogonal_l1 w=0.15` reached final validation loss 7.0100.
- High-pressure/wide-Ricker expansion configs `35` through `48` keep the same fixed budget. Requested Ricker point `w=0.3, c=s=0.05` was already present as configs `15` and `22`; the expansion adds the remaining `w in {0.3, 1.0}`, `c=s in {0.05, 0.1, 0.5}` settings plus L1/OL1 `w in {2.0, 5.0}`.
- Expansion figures: `figures/17-pythia-14m-pressure-fixed-2048-high-pressure-rn-learning-curves.pdf`, `figures/18-pythia-14m-pressure-fixed-2048-high-pressure-or-learning-curves.pdf`, `figures/19-pythia-14m-pressure-fixed-2048-high-pressure-l1-learning-curves.pdf`, and `figures/20-pythia-14m-pressure-fixed-2048-high-pressure-clipping-frontiers.pdf`.
- Fixed-step post-hoc clipping frontiers now use thresholds `[0, 0.001, 0.003, 0.01, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3]`, which maps representative runs through the 80-90% exact-zero activation sparsity region.
- Interpretation boundary: this is a one-seed planning screen. The full ablation should repeat key candidates over multiple seeds, use longer token budgets, and use a larger or full deterministic validation pass.

## Expected Ablations

TODO: design the full ablation around the fixed-step screen. Initial candidates to carry forward are AdamW monitor-only, L1 weights near `0.15`, orthogonal L1 weights near `0.15`, mild Ricker `w=0.03, c=0.05, s=0.05`, and orthogonal Ricker at moderate pressure for the sparsity/loss tradeoff.

TODO: run a short `weight_decay=0` ablation to test whether weight decay materially shapes activation distributions or only acts as a small background regularizer. AdamW `weight_decay`, `betas`, and `eps` are now explicit in configs and new-run manifests.

TODO: test an architecture ablation that replaces Pythia/GPT-NeoX GELU MLP activations with ReLU, keeping the rest of the pretraining recipe fixed. This should be treated as an architecture-change ablation, not as a default setting.

Completed full-pass high-pressure configs:

- `configs/56-pythia-14m-minipile-orthogonal-ricker-full-pass-w1-c0p05-s0p05.yaml`
- `configs/57-pythia-14m-minipile-ricker-naive-full-pass-w1-c0p05-s0p05.yaml`
- `configs/58-pythia-14m-minipile-l1-naive-full-pass-w5.yaml`
- `configs/59-pythia-14m-minipile-orthogonal-l1-full-pass-w5.yaml`

These runs use the same one-MiniPile-token-cache-pass budget as the AdamW full-pass baseline: 22,762 optimizer steps and 1,491,730,432 tokens. They are high-pressure stress tests, not final selected paper settings. Final validation losses were AdamW 4.8317, OR w1 c0.05 s0.05 5.2281, RN w1 c0.05 s0.05 5.2403, L1N w5 5.2144, and OL1 w5 5.2342. Final `abs(a) <= 0.01` activation mass was AdamW 6.03%, OR 63.81%, RN 72.73%, L1N 81.05%, and OL1 73.39%.

High-pressure full-pass figures:

- `figures/31-pythia-14m-minipile-full-pass-high-pressure-learning-curves.pdf`
- `figures/32-pythia-14m-minipile-full-pass-high-pressure-weight-norms.pdf`
- `figures/33-pythia-14m-minipile-full-pass-high-pressure-clipping-frontiers.pdf`
- `figures/34-pythia-14m-minipile-full-pass-high-pressure-gradient-diagnostics.pdf`
- `figures/35-pythia-14m-minipile-full-pass-high-pressure-activation-histograms.pdf`
- `figures/36-pythia-14m-minipile-full-pass-high-pressure-weight-histograms.pdf`
- `figures/37-pythia-14m-minipile-full-pass-high-pressure-attention-weight-histograms.pdf`
- `figures/38-pythia-14m-minipile-full-pass-high-pressure-residual-stream-histograms.pdf`
- `figures/39-pythia-14m-minipile-full-pass-high-pressure-attention-output-histograms.pdf`
- `figures/40-pythia-14m-minipile-full-pass-high-pressure-all-site-clipping-frontiers.pdf`
- `figures/49-pythia-14m-minipile-full-pass-high-pressure-mlp-hiddens-clipping-frontiers.pdf`
- `figures/50-pythia-14m-minipile-full-pass-high-pressure-residual-streams-clipping-frontiers.pdf`
- `figures/51-pythia-14m-minipile-full-pass-high-pressure-attention-outputs-clipping-frontiers.pdf`

All-site full-pass pressure extension:

- Configs: `configs/65-pythia-14m-minipile-orthogonal-ricker-all-site-full-pass-w1-c0p05-s0p05.yaml` and `configs/66-pythia-14m-minipile-orthogonal-l1-all-site-full-pass-w5.yaml`.
- Pressure sites: `mlp_hiddens`, `attention_outputs`, and `residual_streams`.
- Budget: one MiniPile token-cache pass per run, 22,762 optimizer steps and 1,491,730,432 tokens.
- OR all-site w1 c0.05 s0.05 final validation loss was 5.1033; final aggregate `abs(a) <= 0.01` activation mass was 44.05%; peak allocated GPU memory was 6,931.4 MB.
- OL1 all-site w5 final validation loss was 4.9192; final aggregate `abs(a) <= 0.01` activation mass was 31.80%; peak allocated GPU memory was 6,931.7 MB.
- AdamW full-pass reference final validation loss was 4.8317; final aggregate `abs(a) <= 0.01` activation mass was 6.03%.

All-site pressure figures:

- `figures/41-pythia-14m-minipile-full-pass-all-site-pressure-learning-curves.pdf`
- `figures/42-pythia-14m-minipile-full-pass-all-site-pressure-clipping-frontiers.pdf`
- `figures/43-pythia-14m-minipile-full-pass-all-site-pressure-mlp-activation-histograms.pdf`
- `figures/44-pythia-14m-minipile-full-pass-all-site-pressure-residual-stream-histograms.pdf`
- `figures/45-pythia-14m-minipile-full-pass-all-site-pressure-attention-output-histograms.pdf`
- `figures/46-pythia-14m-minipile-full-pass-all-site-pressure-mlp-hiddens-clipping-frontiers.pdf`
- `figures/47-pythia-14m-minipile-full-pass-all-site-pressure-residual-streams-clipping-frontiers.pdf`
- `figures/48-pythia-14m-minipile-full-pass-all-site-pressure-attention-outputs-clipping-frontiers.pdf`

## Expected Scale Ladders

TODO: after the Pythia-14M MiniPile random-init baseline is stable and calibrated, consider scaling within the Pythia family up to 160M if memory and runtime measurements justify it. Do not add scale-up configs until the 14M path is reproducible.
