# Paper Evidence Map

This is the authoritative index from paper items to pinned saved evidence and
regenerable outputs.

| Paper item | Claim or purpose | Scaffold and configs | Pinned runs and artifacts | Scaffold output | Regeneration command |
| --- | --- | --- | --- | --- | --- |
| Final A1 eleven-cell learning-rate screen | Completed seed-0 evidence at the fixed 400M-token horizon; `6.4e-2` has the lowest final selection loss among the eleven tested values. This selects the A1 learning rate but does not establish a global optimum or quantify seed uncertainty | `experiments/01-a1-lr-screen`; configs `001`–`011` | Runs `001-20260825-191155-6b7376de`, `001-20260825-191154-b9299c46`, `001-20260825-195141-f842c400`, `001-20260826-123606-46e7454f`, `001-20260826-135546-928279bb`, `001-20260826-174611-04b42898`, `001-20260826-182559-bb05a50c`, `001-20260826-190546-4df1c441`, `001-20260826-221407-812e78f4`, `001-20260826-225355-07a74682`, and `001-20260826-233349-87400e7d`; each run's `config.yaml`, `manifest.json`, and `metrics.json`, plus its tracked recipe | `experiments/01-a1-lr-screen/figs/01-a1-learning-rate-screen.pdf`; companion `.png`, `.md`, and `.provenance.json` files with the same stem | `python -m paper_exp.plots.a1_lr_screen` |
| A1 training progress | Descriptive seed-0 validation-loss trajectories and logged effective learning-rate schedules for all eleven cells at the fixed 400M-token horizon; no seed uncertainty or convergence claim | `experiments/01-a1-lr-screen`; configs `001`–`011` | Runs `001-20260825-191155-6b7376de`, `001-20260825-191154-b9299c46`, `001-20260825-195141-f842c400`, `001-20260826-123606-46e7454f`, `001-20260826-135546-928279bb`, `001-20260826-174611-04b42898`, `001-20260826-182559-bb05a50c`, `001-20260826-190546-4df1c441`, `001-20260826-221407-812e78f4`, `001-20260826-225355-07a74682`, and `001-20260826-233349-87400e7d`; each run's `config.yaml`, `manifest.json`, `metrics.json`, and `events.jsonl`, plus its tracked recipe | `experiments/01-a1-lr-screen/figs/02-a1-training-progress.pdf`; companion `.png`, `.md`, and `.provenance.json` files with the same stem | `python -m paper_exp.plots.a1_lr_screen --progress` |
| A2 L1 spillover response | Seed-0 directional evidence for the quality-sparsity tradeoff under h-only naive L1: all five L1 cells worsen final validation loss, the targeted `h` near-zero response increases with lambda, and untargeted responses are heterogeneous. No seed-robustness, compensation, compute, or speedup claim | `experiments/02-a2-l1-screen`; configs `012`-`018` | Pretraining runs `001-20260827-150809-2eb832f6`, `001-20260827-150808-8117d1fe`, `001-20260827-173546-360c077f`, `001-20260827-193752-3fbbd6c0`, `001-20260827-220532-79995961`, and `001-20260828-000829-0959f855`; diagnostic run `001-20260828-082044-a031175f`, especially `activation_histograms.json`; every run's tracked/saved config, manifest, and required metrics/checkpoint | `experiments/02-a2-l1-screen/figs/01-a2-spillover-response.pdf`; companion `.png`, `.md`, and `.provenance.json` files with the same stem | `python -m paper_exp.plots.a2_spillover` |
| A2 layer-5 activation distributions | Descriptive seed-0 control-versus-L1-lambda-1 view at all six sites, with exact-zero atoms separate from conditional nonzero densities and stored-range tail mass disclosed | `experiments/02-a2-l1-screen`; configs `012`-`018` | Same exact six pretraining runs and diagnostic run `001-20260828-082044-a031175f` pinned by the A2 spillover response; `activation_histograms.json` layer-5 rows are the specialized evidence | `experiments/02-a2-l1-screen/figs/02-a2-layer5-distributions.pdf`; companion `.png`, `.md`, and `.provenance.json` files with the same stem | `python -m paper_exp.plots.a2_spillover` |

Each populated row must:

- state a claim or descriptive purpose no stronger than the saved evidence;
- name exact scaffold, config IDs, and run IDs;
- name specialized diagnostic artifacts where applicable;
- use one unique sequential output prefix within the owning scaffold's `figs/`;
- include the deterministic command that regenerates the output;
- agree with the experiment plan's seed, uncertainty, and confirmation rules.

Do not use wildcard paths, latest-run discovery, notebook-only state, or an
internal report as the sole evidence locator. A cross-tranche figure belongs
to the later plan-defined scaffold whose tracked recipe pins every source.
