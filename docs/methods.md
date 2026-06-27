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

## Expected Ablations

TODO

## Expected Scale Ladders

TODO: after the Pythia-14M MiniPile random-init baseline is stable and calibrated, consider scaling within the Pythia family up to 160M if memory and runtime measurements justify it. Do not add scale-up configs until the 14M path is reproducible.
