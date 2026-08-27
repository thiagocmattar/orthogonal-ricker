# Paper Evidence Map

This is the authoritative index from paper items to pinned saved evidence and
regenerable outputs.

| Paper item | Claim or purpose | Scaffold and configs | Pinned runs and artifacts | Scaffold output | Regeneration command |
| --- | --- | --- | --- | --- | --- |
| Final A1 eleven-cell learning-rate screen | Completed seed-0 evidence at the fixed 400M-token horizon; `6.4e-2` has the lowest final selection loss among the eleven tested values. This selects the A1 learning rate but does not establish a global optimum or quantify seed uncertainty | `experiments/01-a1-lr-screen`; configs `001`–`011` | Runs `001-20260825-191155-6b7376de`, `001-20260825-191154-b9299c46`, `001-20260825-195141-f842c400`, `001-20260826-123606-46e7454f`, `001-20260826-135546-928279bb`, `001-20260826-174611-04b42898`, `001-20260826-182559-bb05a50c`, `001-20260826-190546-4df1c441`, `001-20260826-221407-812e78f4`, `001-20260826-225355-07a74682`, and `001-20260826-233349-87400e7d`; each run's `config.yaml`, `manifest.json`, and `metrics.json`, plus its tracked recipe | `experiments/01-a1-lr-screen/figs/01-a1-learning-rate-screen.pdf`; companion `.png`, `.md`, and `.provenance.json` files with the same stem | `python -m paper_exp.plots.a1_lr_screen` |
| A1 training progress | Descriptive seed-0 validation-loss trajectories and logged effective learning-rate schedules for all eleven cells at the fixed 400M-token horizon; no seed uncertainty or convergence claim | `experiments/01-a1-lr-screen`; configs `001`–`011` | Runs `001-20260825-191155-6b7376de`, `001-20260825-191154-b9299c46`, `001-20260825-195141-f842c400`, `001-20260826-123606-46e7454f`, `001-20260826-135546-928279bb`, `001-20260826-174611-04b42898`, `001-20260826-182559-bb05a50c`, `001-20260826-190546-4df1c441`, `001-20260826-221407-812e78f4`, `001-20260826-225355-07a74682`, and `001-20260826-233349-87400e7d`; each run's `config.yaml`, `manifest.json`, `metrics.json`, and `events.jsonl`, plus its tracked recipe | `experiments/01-a1-lr-screen/figs/02-a1-training-progress.pdf`; companion `.png`, `.md`, and `.provenance.json` files with the same stem | `python -m paper_exp.plots.a1_lr_screen --progress` |

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
