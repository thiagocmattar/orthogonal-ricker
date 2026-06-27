# Methods

## Research Question

TODO

## Baseline

Pythia-14M on MiniPile is the first concrete baseline path.

Model: `EleutherAI/pythia-14m-deduped`.

Dataset: `JeanKaddour/minipile`.

This is a low-resource MiniPile baseline, not a faithful reproduction of the original Pythia 300B-token pretraining run.

## Proposed Method

TODO

## Evaluation Setup

TODO: calibrate throughput with a short local tokenized MiniPile run before choosing the full baseline budget.

For paper runs, use local cached/tokenized MiniPile data rather than streaming so token counts, ordering, and runtime are auditable.

Training/calibration tracking:

- Training samples random contiguous token blocks with replacement from the local token cache.
- `training.log_every` controls train event logging.
- `validation.eval_every_steps` controls validation loss frequency.
- Validation uses a separately tokenized validation split cache when `validation.enabled` is true.
- Current tracked summaries include train loss, validation loss, throughput, wall-clock time, estimated epoch, learning rate, gradient norm, weight norm, peak GPU memory, and final checkpoint metadata.
- Calibration and baseline runs can save final model weights under `checkpoints/final/` when `checkpoint.save_final` is true.

Current calibration notes:

- `configs/01-pythia-14m-minipile-smoke.yaml` tokenized 128 MiniPile train documents into 190,960 GPT-NeoX/Pythia tokens.
- This implies 1,491.9 tokens/document on the smoke sample. If representative, MiniPile train has roughly 1.49B tokens across 1M documents.
- Latest calibration result: `results/01-pythia-14m-minipile-smoke/008-20260627-110058-1442200f/`.
- Calibration settings: Pythia-14M, CUDA, bf16, block size 2048, micro-batch size 4, 50 optimizer steps, validation every 25 steps.
- Measured throughput: 117,421 tokens/sec over 409,600 training tokens.
- Final train loss: 6.0995.
- Final validation loss: 4.9626.
- Peak allocated GPU memory: 5,785.7 MB.
- Final checkpoint saved under `checkpoints/final/`, size 26.84 MB.
- Estimated raw training-loop time for one MiniPile pass with Pythia-14M: about 3.5 hours.
- This estimate excludes full tokenization time, checkpointing, validation, plotting, and long-run thermal/runtime drift.

## Expected Ablations

TODO

## Expected Scale Ladders

TODO: after the Pythia-14M MiniPile baseline is stable and calibrated, consider scaling within the Pythia family up to 160M if memory and runtime measurements justify it. Do not add scale-up configs until the 14M path is reproducible.
