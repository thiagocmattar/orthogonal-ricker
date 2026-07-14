# Results

Run outputs are written here:

```text
results/<config_id>/<run_id>/
```

Example:

```text
results/01-baseline/001-20260626-173041-33e1187b/
```

Completed run directories should contain the core envelope:

- `config.yaml`
- `metrics.json`
- `manifest.json`
- `predictions.jsonl`

## Lifecycle Status

Smoke and calibration/pretrain runs save `config.yaml` and `manifest.json` at
launch. Their manifest transitions as follows:

```text
running -> completed
        -> failed
```

- `running`: the immutable config and launch provenance are durable; metrics
  and predictions are not expected yet.
- `completed`: metrics and predictions were written before the terminal
  manifest, which includes `finished_at`.
- `failed`: an escaping exception was recorded with its type and message and a
  `finished_at` timestamp; partial artifacts may remain. Exiting a lifecycle
  without explicitly completing it is also recorded as a failure.

The Git commit and dirty state in these manifests describe launch time and are
not recomputed at completion. A statusless historical manifest is still
treated as completed when the full core envelope is coherent.

This lifecycle currently covers only smoke and calibration/pretrain. Data
preparation, activation and weight histogram diagnostics, activation
propagation, and clipping sweeps retain the legacy end-of-run writer until a
second migration.

For calibration/pretrain, `predictions.jsonl` is currently the training and
validation event history also saved as `events.jsonl`, not generated-token
predictions. `calibration/wall_seconds_train` includes validation executed
inside the timed training interval; validation time is also broken out in its
own metrics.

Keep raw results here. Regenerated figures belong in `figures/`.
