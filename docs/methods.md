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

- Site support: `mlp_hiddens` for Pythia/GPT-NeoX, hooked at `gpt_neox.layers.N.mlp.act`.
- Naive methods: `ricker_naive` and `l1_naive` optimize `task_loss + weight * pressure_loss`.
- Orthogonal methods: `orthogonal_ricker` and `orthogonal_l1` compute task gradients and pressure gradients separately; AdamW steps on task gradients only; then a memoryless pressure correction is applied in AdamW step space.
- Orthogonal projection fires only when the pressure update direction conflicts with the AdamW task update direction. The trust budget caps the final pressure/task update ratio.
- Post-hoc clipping sweeps load a saved checkpoint and evaluate validation loss versus achieved exact-zero fraction for configured thresholds or quantiles.

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

- Configs: `configs/12-pythia-14m-minipile-adamw-fixed-2048.yaml` through `configs/34-pythia-14m-minipile-orthogonal-l1-fixed-2048-w1.yaml`.
- Detailed readout: `docs/humans/02-fixed-step-pressure-screen.md`.
- Figures: `figures/05-pythia-14m-pressure-fixed-2048-summary.pdf`, `figures/06-pythia-14m-pressure-fixed-2048-learning-curves.pdf`, and `figures/07-pythia-14m-pressure-fixed-2048-clipping-frontiers.pdf`.
- Budget: 2,048 optimizer steps, 134,217,728 tokens per run, 0.08998 estimated MiniPile token-cache epochs.
- AdamW baseline was rerun as monitor-only with `activation_pressure.method: none`, so activation near-zero mass is measured without adding an auxiliary loss.
- AdamW monitor-only final validation loss was 7.0200; final `abs(a) <= 0.01` MLP hidden mass was 6.47%.
- Naive Ricker increases near-zero activation mass but can impose a large validation-loss cost as pressure increases.
- Orthogonal Ricker reduces that cost at matched nominal settings. Example: at `w=0.1, c=0.05, s=0.05`, naive Ricker final validation loss was 7.0848, while orthogonal Ricker final validation loss was 7.0480.
- L1 pressure is the strongest early candidate in this screen. `l1_naive w=0.15` reached final validation loss 7.0104; `orthogonal_l1 w=0.15` reached final validation loss 7.0100.
- Fixed-step post-hoc clipping frontiers now use thresholds `[0, 0.001, 0.003, 0.01, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3]`, which maps representative runs through the 80-90% exact-zero activation sparsity region.
- Interpretation boundary: this is a one-seed planning screen. The full ablation should repeat key candidates over multiple seeds, use longer token budgets, and use a larger or full deterministic validation pass.

## Expected Ablations

TODO: design the full ablation around the fixed-step screen. Initial candidates to carry forward are AdamW monitor-only, L1 weights near `0.15`, orthogonal L1 weights near `0.15`, mild Ricker `w=0.03, c=0.05, s=0.05`, and orthogonal Ricker at moderate pressure for the sparsity/loss tradeoff.

## Expected Scale Ladders

TODO: after the Pythia-14M MiniPile random-init baseline is stable and calibrated, consider scaling within the Pythia family up to 160M if memory and runtime measurements justify it. Do not add scale-up configs until the 14M path is reproducible.
