# Experiment Log

This registry starts with the definitive experiment workflow. Historical
campaign runs are intentionally not carried into this branch; Git history is
their archive.

Do not add a scientific row until the reviewed experiment plan authorizes the
config and the run has a durable manifest. Record failed and invalid attempts as
well as successful ones.

| Date | Scaffold | Config | Run | Terminal status | Case class | Evidence status | Purpose | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-08-25 | [`experiments/01-a1-lr-screen`](../experiments/01-a1-lr-screen/) | [`001-a1-lr-5e-4`](../experiments/01-a1-lr-screen/run/001-a1-lr-5e-4.yaml) | [`001-20260825-191155-6b7376de`](../experiments/01-a1-lr-screen/raw/001-a1-lr-5e-4/001-20260825-191155-6b7376de/) | completed | eligible | valid | A1 peak-learning-rate screen | Completed the exact reviewed budget and artifact contract; retained as an eligible nonselected cell. |
| 2026-08-25 | [`experiments/01-a1-lr-screen`](../experiments/01-a1-lr-screen/) | [`002-a1-lr-1e-3`](../experiments/01-a1-lr-screen/run/002-a1-lr-1e-3.yaml) | [`001-20260825-191154-b9299c46`](../experiments/01-a1-lr-screen/raw/002-a1-lr-1e-3/001-20260825-191154-b9299c46/) | completed | eligible | valid | A1 peak-learning-rate screen | Completed the exact reviewed budget and artifact contract; retained as an eligible nonselected cell. |
| 2026-08-25 | [`experiments/01-a1-lr-screen`](../experiments/01-a1-lr-screen/) | [`003-a1-lr-2e-3`](../experiments/01-a1-lr-screen/run/003-a1-lr-2e-3.yaml) | [`001-20260825-195141-f842c400`](../experiments/01-a1-lr-screen/raw/003-a1-lr-2e-3/001-20260825-195141-f842c400/) | completed | eligible | valid | A1 peak-learning-rate screen | Selected by the reviewed A1 rule; freezes `lr_14m` at `0.002` for downstream review. |

Status rules:

- `Terminal status` mirrors the immutable source manifest: `running`,
  `completed`, or `failed`.
- `Case class` is `eligible`, `scientific_failure`,
  `infrastructure_failure`, or `unresolved`, under
  `experimental-design/protocol.md`.
- `Evidence status` is `valid`, `provisional`, or `invalid`, with the reason in
  `Notes`.
- `Scaffold` names the exact chronological tranche directory that owns the
  recipe, raw attempt, and figures.
- Link the exact scaffold config and raw run directory; never use a wildcard or
  "latest."
- Do not copy scalar results into this log when the saved artifact is the
  authority. Summarize only the decision and interpretation boundary.

The registry currently contains the completed A1 tranche. Later scientific
rows require newly reviewed plan scope.
