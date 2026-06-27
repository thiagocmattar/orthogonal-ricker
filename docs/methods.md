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
- The next valid calibration must be rerun from random initialization.

## Expected Ablations

TODO

## Expected Scale Ladders

TODO: after the Pythia-14M MiniPile random-init baseline is stable and calibrated, consider scaling within the Pythia family up to 160M if memory and runtime measurements justify it. Do not add scale-up configs until the 14M path is reproducible.
