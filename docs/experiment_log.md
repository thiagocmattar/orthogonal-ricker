# Experiment Log

| Date | Experiment | Config | Result path | Notes |
| ---- | ---------- | ------ | ----------- | ----- |
| 2026-06-27 | MiniPile tokenization smoke | `configs/01-pythia-14m-minipile-smoke.yaml` | `results/01-pythia-14m-minipile-smoke/007-20260627-110043-4b11dd68/` | 128 train docs / 190,960 tokens; 32 validation docs / 39,020 tokens; block size 2048. |
| 2026-06-27 | Pythia-14M tracked calibration | `configs/01-pythia-14m-minipile-smoke.yaml` | `results/01-pythia-14m-minipile-smoke/008-20260627-110058-1442200f/` | CUDA bf16, block size 2048, micro-batch 4, 50 steps, validation every 25 steps, 117,421 tokens/sec, final train loss 6.0995, final val loss 4.9626, peak GPU memory 5,785.7 MB, final checkpoint 26.84 MB. |
| TODO | Pythia-14M MiniPile baseline | `configs/02-pythia-14m-minipile-baseline.yaml` | TODO | TODO: choose run budget after calibration. |
