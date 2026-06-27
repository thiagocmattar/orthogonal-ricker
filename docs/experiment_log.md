# Experiment Log

| Date | Experiment | Config | Result path | Notes |
| ---- | ---------- | ------ | ----------- | ----- |
| 2026-06-27 | Pythia-14M full-MiniPile token cache check | `configs/03-pythia-14m-minipile-random-full-10min.yaml` | `results/03-pythia-14m-minipile-random-full-10min/001-20260627-141155-645cf7bd/` | Reused and validated full MiniPile token cache: 1,000,000 train docs, 1,491,711,416 train tokens; 500 validation docs, 693,668 validation tokens. |
| 2026-06-27 | Pythia-14M random-init full-MiniPile pretraining checkpoint | `configs/03-pythia-14m-minipile-random-full-10min.yaml` | `results/03-pythia-14m-minipile-random-full-10min/003-20260627-142522-7fc1e76f/` | 10-minute run, random initialization, FP32 parameters, bf16 autocast, 86,245,376 tokens, 0.0578 epochs, 143,666 tokens/sec, train loss 10.8567 to 7.6701, val loss 10.8504 to 7.5450, peak allocated 5,997.0 MB, peak reserved 7,428.0 MB. |
| 2026-06-27 | Failed random-init pretraining attempt | `configs/03-pythia-14m-minipile-random-full-10min.yaml` | `results/03-pythia-14m-minipile-random-full-10min/002-20260627-141159-10a3e24a/` | Do not use as a result. Random model parameters were float16, causing non-finite losses. Harness now forces FP32 parameters and aborts on non-finite loss. |
| TODO | Pythia-14M MiniPile pretraining baseline | `configs/02-pythia-14m-minipile-baseline.yaml` | TODO | TODO: choose run budget after random-init calibration. |

Note: Earlier Pythia runs that loaded released checkpoint weights were removed because they were continuation/fine-tuning checks, not random-initialized pretraining runs.
