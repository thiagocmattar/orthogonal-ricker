# Paper Evidence Map

This is the authoritative index from paper items to pinned saved evidence and
regenerable outputs.

| Paper item | Claim or purpose | Scaffold and configs | Pinned runs and artifacts | Scaffold output | Regeneration command |
| --- | --- | --- | --- | --- | --- |
| Interim A1 eight-cell learning-rate screen | Completed seed-0 evidence through `6.4e-2` at the fixed 400M-token horizon; this interim output does not freeze `lr_14m` after the reviewed eleven-cell extension | `experiments/01-a1-lr-screen`; configs `001`–`008` | Runs `001-20260825-191155-6b7376de`, `001-20260825-191154-b9299c46`, `001-20260825-195141-f842c400`, `001-20260826-123606-46e7454f`, `001-20260826-135546-928279bb`, `001-20260826-174611-04b42898`, `001-20260826-182559-bb05a50c`, and `001-20260826-190546-4df1c441`; each run's `config.yaml`, `manifest.json`, and `metrics.json`, plus its tracked recipe | `experiments/01-a1-lr-screen/figs/01-a1-learning-rate-screen.pdf`; companion `.png`, `.md`, and `.provenance.json` files with the same stem | `python -m paper_exp.plots.a1_lr_screen` |

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
