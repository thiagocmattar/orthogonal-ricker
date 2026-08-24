# Experiment Log

This registry starts with the definitive experiment workflow. Historical
campaign runs are intentionally not carried into this branch; Git history is
their archive.

Do not add a scientific row until the reviewed experiment plan authorizes the
config and the run has a durable manifest. Record failed and invalid attempts as
well as successful ones.

| Date | Scaffold | Config | Run | Terminal status | Case class | Evidence status | Purpose | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

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

`TODO:` Populate this table after the definitive experiment plan is supplied,
reviewed, and launched.
