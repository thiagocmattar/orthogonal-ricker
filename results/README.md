# Results

Run artifacts are written under:

```text
results/<config-id>/<run-id>/
```

`config-id` is the config filename without `.yaml`. `run-id` uses a sequential
attempt prefix plus UTC timestamp and short unique suffix:

```text
001-YYYYMMDD-HHMMSS-xxxxxxxx
```

Config IDs remain globally unique even though configs are grouped into
same-named launch folders. The manifest command records the numeric case runner
that launched a tranche.

Results are local artifacts and are ignored by Git. Do not commit checkpoints,
large event streams, or derived diagnostic outputs accidentally.

## Core Envelope

The common completed envelope is:

```text
config.yaml
manifest.json
metrics.json
predictions.jsonl
```

Both training workflows (`calibrate` and `pretrain`) additionally write:

```text
events.jsonl
checkpoints/final/    # when checkpoint.save_final is true
```

For both training workflows, `predictions.jsonl` duplicates the train and
validation event history stored in `events.jsonl`; it is not generated-token
output. Keep that meaning explicit in downstream tools. The final checkpoint is
conditional on `checkpoint.save_final` in either workflow.

Pretraining metrics use the `training/` namespace. Throughput-calibration
metrics use `calibration/`; downstream tools must not conflate the two modes.

Specialized workflows add one primary artifact:

- clipping: `clipping_frontier.jsonl`;
- activation histograms: `activation_histograms.json`;
- weight histograms: `weight_histograms.json`;
- activation propagation: `activation_propagation.json`.

See [`docs/diagnostics.md`](../docs/diagnostics.md) for their measurement
contracts.

## Run Lifecycle

Every retained workflow uses the same explicit lifecycle:

```text
running -> completed
        -> failed
```

At launch, the harness creates the run directory and atomically writes:

- the immutable `config.yaml` snapshot;
- a `manifest.json` with `status: running`, `started_at`, config/run identity,
  command, environment, and launch Git provenance.

On success, required metrics and predictions are written before the terminal
manifest is atomically published with `status: completed` and `finished_at`.
Pretraining writes its event stream and required checkpoint before completion.
Clipping and the retained histogram/propagation diagnostics write their
specialized artifact before the common result envelope and terminal manifest.
Data preparation publishes its cache metadata only after the declared token
cache is complete.

On an escaping exception, the original error is re-raised after the manifest is
updated to `status: failed`, `finished_at`, and a failure type/message. Partial
artifacts may remain. A lifecycle scope that exits without explicit completion
is also failed.

Terminal manifests are derived from the immutable launch snapshot. Git commit
and dirty state always describe launch time, even if the working tree changes
later.

## Consumption Rules

- A paper input must name an exact config ID, run ID, and specialized artifact.
- A source is consumable only when its terminal manifest says `completed`, its
  config/run identity matches its directory, and its required artifacts are
  coherent.
- Activation and weight diagnostics must pin every source with both
  `config_id` and `run_id`; they never select the latest run.
- A statusless historical run remains a valid record when its core envelope is
  coherent. It is not definitive paper input on this branch unless the reviewed
  plan explicitly accepts and pins it.
- A failed or incomplete attempt is never silently selected as completed.
- Provisional use requires an explicit limitation and exact durable source.
- Never overwrite an attempt. A retry creates the next run ID.

Run the read-only integrity scan with:

```bash
make check
paper-exp check --strict
```

The scan checks run groups owned by configs in the current checkout plus
artifact groups explicitly indexed in the experiment log or paper map. Local
archives from earlier branches are outside the current workflow and are not
silently treated as release evidence.
