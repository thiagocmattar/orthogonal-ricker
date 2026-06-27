# Paper Map

| Paper item | Claim / purpose | Config | Result | Figure |
| ---------- | --------------- | ------ | ------ | ------ |
| Baseline table | Pythia-14M architecture pretraining on MiniPile from random initialization | `configs/02-pythia-14m-minipile-baseline.yaml` | TODO | TODO |
| Throughput note | 10-minute random-init full-MiniPile checkpoint | `configs/03-pythia-14m-minipile-random-full-10min.yaml` | `results/03-pythia-14m-minipile-random-full-10min/003-20260627-142522-7fc1e76f/` | `figures/01-pythia-14m-minipile-random-full-10min-diagnostics.pdf` |
| Method plumbing checks | Activation pressure smoke suite: naive Ricker/L1 and Adam-step orthogonal Ricker/L1 | `configs/04-pythia-14m-minipile-ricker-naive-smoke.yaml` through `configs/07-pythia-14m-minipile-orthogonal-l1-smoke.yaml` | See `docs/experiment_log.md` | TODO |
| Post-hoc clipping frontier | Smoke check for validation loss versus exact activation sparsity | `configs/03-pythia-14m-minipile-random-full-10min.yaml` checkpoint | `results/03-pythia-14m-minipile-random-full-10min-clipping-sweep/002-20260627-150326-6a61b34d/` | `figures/02-pythia-14m-minipile-clipping-frontier-smoke.pdf` |
| Short pressure trajectory check | Compare early train loss, validation loss, pressure loss, and MLP near-zero mass for four activation pressure methods | `configs/08-pythia-14m-minipile-ricker-naive-short.yaml` through `configs/11-pythia-14m-minipile-orthogonal-l1-short.yaml` | See `docs/experiment_log.md` | `figures/03-pythia-14m-pressure-short-learning-curves.pdf` |
| Short pressure clipping frontier | Compare post-hoc clipping frontiers for pressure-trained checkpoints | Checkpoints from configs `08` through `11` | See `docs/experiment_log.md` | `figures/04-pythia-14m-pressure-short-clipping-frontiers.pdf` |
| Scale ladder | TODO | TODO | TODO | TODO |
