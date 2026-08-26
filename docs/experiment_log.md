# Experiment Log

This registry starts with the definitive experiment workflow. Historical
campaign runs are intentionally not carried into this branch; Git history is
their archive.

Do not add a scientific row until the reviewed experiment plan authorizes the
config and the run has a durable manifest. Record failed and invalid attempts as
well as successful ones.

| Date | Scaffold | Config | Run | Terminal status | Case class | Evidence status | Purpose | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-08-25 | [`experiments/01-a1-lr-screen`](../experiments/01-a1-lr-screen/) | [`001-a1-lr-5e-4`](../experiments/01-a1-lr-screen/run/001-a1-lr-5e-4.yaml) | [`001-20260825-191155-6b7376de`](../experiments/01-a1-lr-screen/raw/001-a1-lr-5e-4/001-20260825-191155-6b7376de/) | completed | eligible | valid | A1 peak-learning-rate screen | Eligible nonselected cell under the final eight-cell selection rule. |
| 2026-08-25 | [`experiments/01-a1-lr-screen`](../experiments/01-a1-lr-screen/) | [`002-a1-lr-1e-3`](../experiments/01-a1-lr-screen/run/002-a1-lr-1e-3.yaml) | [`001-20260825-191154-b9299c46`](../experiments/01-a1-lr-screen/raw/002-a1-lr-1e-3/001-20260825-191154-b9299c46/) | completed | eligible | valid | A1 peak-learning-rate screen | Eligible nonselected cell under the final eight-cell selection rule. |
| 2026-08-25 | [`experiments/01-a1-lr-screen`](../experiments/01-a1-lr-screen/) | [`003-a1-lr-2e-3`](../experiments/01-a1-lr-screen/run/003-a1-lr-2e-3.yaml) | [`001-20260825-195141-f842c400`](../experiments/01-a1-lr-screen/raw/003-a1-lr-2e-3/001-20260825-195141-f842c400/) | completed | eligible | valid | A1 peak-learning-rate screen | Eligible nonselected cell under the final eight-cell selection rule. |
| 2026-08-26 | [`experiments/01-a1-lr-screen`](../experiments/01-a1-lr-screen/) | [`004-a1-lr-4e-3`](../experiments/01-a1-lr-screen/run/004-a1-lr-4e-3.yaml) | [`001-20260826-123606-46e7454f`](../experiments/01-a1-lr-screen/raw/004-a1-lr-4e-3/001-20260826-123606-46e7454f/) | completed | eligible | valid | A1 peak-learning-rate screen | Eligible nonselected cell under the final eight-cell selection rule. |
| 2026-08-26 | [`experiments/01-a1-lr-screen`](../experiments/01-a1-lr-screen/) | [`005-a1-lr-8e-3`](../experiments/01-a1-lr-screen/run/005-a1-lr-8e-3.yaml) | [`001-20260826-135546-928279bb`](../experiments/01-a1-lr-screen/raw/005-a1-lr-8e-3/001-20260826-135546-928279bb/) | completed | eligible | valid | A1 peak-learning-rate screen | Eligible nonselected cell under the final eight-cell selection rule. |
| 2026-08-26 | [`experiments/01-a1-lr-screen`](../experiments/01-a1-lr-screen/) | [`006-a1-lr-1p6e-2`](../experiments/01-a1-lr-screen/run/006-a1-lr-1p6e-2.yaml) | [`001-20260826-174611-04b42898`](../experiments/01-a1-lr-screen/raw/006-a1-lr-1p6e-2/001-20260826-174611-04b42898/) | completed | eligible | valid | A1 peak-learning-rate screen | Eligible nonselected cell under the final eight-cell selection rule. |
| 2026-08-26 | [`experiments/01-a1-lr-screen`](../experiments/01-a1-lr-screen/) | [`007-a1-lr-3p2e-2`](../experiments/01-a1-lr-screen/run/007-a1-lr-3p2e-2.yaml) | [`001-20260826-182559-bb05a50c`](../experiments/01-a1-lr-screen/raw/007-a1-lr-3p2e-2/001-20260826-182559-bb05a50c/) | completed | eligible | valid | A1 peak-learning-rate screen | Eligible nonselected cell under the final eight-cell selection rule. |
| 2026-08-26 | [`experiments/01-a1-lr-screen`](../experiments/01-a1-lr-screen/) | [`008-a1-lr-6p4e-2`](../experiments/01-a1-lr-screen/run/008-a1-lr-6p4e-2.yaml) | [`001-20260826-190546-4df1c441`](../experiments/01-a1-lr-screen/raw/008-a1-lr-6p4e-2/001-20260826-190546-4df1c441/) | completed | eligible | valid | A1 peak-learning-rate screen | Selected by the predeclared final eight-cell rule; this is the upper tested boundary, and no further A1 LR extension is authorized. |

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

The registry contains the complete eight-cell A1 cohort. The predeclared rule
selects config `008-a1-lr-6p4e-2`, run
`001-20260826-190546-4df1c441`; `lr_14m=6.4e-2` is frozen in the
[decision register](experimental-design/decisions.md). The complete curve,
table, and provenance are owned by
[`experiments/01-a1-lr-screen/figs`](../experiments/01-a1-lr-screen/figs/).
Later scientific rows require newly reviewed plan scope.
