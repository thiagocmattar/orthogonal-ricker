# Plotting Contract

Plots are scientific artifacts. The plotting layer reads saved evidence and
presents it; it does not train a model, rerun validation, choose a scientific
cohort, or repair missing inputs.

The current `paper-exp plot` command renders one explicitly named saved run
artifact. Definitive multi-run paper figure families will be added only after
the reviewed experiment plan names their cohorts, estimands, and outputs.

## Four Boundaries

Every paper figure follows four visible boundaries:

```text
explicit pinned artifacts
    -> loader and schema validation
    -> pure scientific reduction
    -> explicit renderer
    -> shared validated export
```

### 1. Load and Pin

The loader receives exact artifact paths or exact config/run identities. It
validates terminal status, schema, source identities, and required fields before
rendering begins.

Paper figures must not:

- select the latest completed run;
- scan result folders and silently choose a replacement;
- consume a failed or incomplete source without a documented provisional use;
- combine checkpoints, histograms, or propagation artifacts whose recorded
  source identities disagree.

Exploratory tools may accept an explicit unpinned path, but their outputs are
not paper artifacts until inputs are frozen.

### 2. Reduce

Scientific transformations belong in small pure functions. A reduction takes
validated saved values and returns plot-ready numbers without filesystem access
or Matplotlib state. Important calculations require focused numerical tests,
especially:

- pooled count fractions and denominators;
- matched deltas and uncertainty;
- clipping baselines and frontiers;
- exact-zero atom removal from continuous densities;
- logical-product accounting;
- frontier or selection membership.

A renderer must not infer missing results or recompute an experiment.

### 3. Render

A focused family module owns the scientific cohort, panel contents, labels,
axis scales, annotations, and figure-specific layout. Shared styling does not
decide scientific content.

Use:

- `src/paper_exp/plots/style.py` for palette, typography, and presentation
  defaults;
- `src/paper_exp/plots/histograms.py` for tested pooled-histogram reductions;
- `src/paper_exp/plots/export.py` for layouts, publication checks, and export;
- one focused module under `src/paper_exp/plots/` for each artifact family.

Do not create a generic plotting framework around hypothetical future figures.
Add one clear family module when the plan defines a real family.

### 4. Export

Build a Matplotlib `Figure` once and export vector PDF plus optional PNG from
that same object. Export is staged beside the final output and promoted only
after validation succeeds. A multi-figure paper suite must stage every output
and publish all-or-nothing so a late failure cannot leave a mixed suite.

PDF metadata must be deterministic where supported. Publication profiles check
the authored canvas, maximum height, minimum text size, and text containment.
Do not use a tight bounding box for a final-size publication profile because it
changes the validated canvas and PDF MediaBox.

## Current Explicit Diagnostic Plots

Render one saved artifact with:

```bash
make plot KIND=<kind> \
  RUN_DIR=experiments/NN-phase-tranche/raw/<config>/<run> \
  OUTPUT=experiments/NN-phase-tranche/figs/01-name.pdf [PNG=1]
```

or:

```bash
paper-exp plot \
  --kind <kind> \
  --run-dir experiments/NN-phase-tranche/raw/<config-id>/<run-id> \
  --output experiments/NN-phase-tranche/figs/01-name.pdf \
  --png
```

Supported kinds are:

Every kind requires the run's exact `config.yaml` and a terminal
`status: completed` `manifest.json`. The additional required inputs are:

| Kind | Additional saved input | Purpose |
| --- | --- | --- |
| `run` | `events.jsonl` and `metrics.json` | Loss, optimization norms, and recorded run statistics |
| `clipping` | `clipping_frontier.jsonl` | Exact-zero clipping versus validation loss |
| `activation-histograms` | `activation_histograms.json` | Exact-zero atom and conditional nonzero activation density |
| `weight-histograms` | `weight_histograms.json` | Pooled saved weight distributions |
| `activation-propagation` | `activation_propagation.json` | Activation exact-zero and logical zero-product heatmaps |

These are explicit diagnostic views, not a paper cohort registry. A single-run
export must use the source run's scaffold `figs/` directory. Each export writes
a deterministic `.provenance.json` sidecar with scaffold/source identities and
SHA-256 hashes for every required input and generated output.

## Visual Integrity

Use the following defaults unless a reviewed plan or venue requirement gives a
scientific reason to differ:

- vector PDF as the canonical output and optional 300-DPI PNG for inspection;
- embedded TrueType fonts;
- white background and subtle gray grid;
- colorblind-safe color plus marker or line-style redundancy;
- stable series identity within a figure family;
- readable labels at final paper size;
- units on axes where units exist;
- visible sample size, seed count, or denominator;
- uncertainty or per-run points when the saved evidence supports them;
- an explicit note for one-seed, exploratory, or provisional evidence;
- legends outside dense data regions when practical.

Axis rules:

- do not truncate an axis to exaggerate an effect;
- if a scientifically useful zoom is used, label it directly in the figure;
- label logarithmic scales;
- keep shared scales genuinely shared or mark panel-specific scales explicitly;
- include threshold-zero or control values when they define the interpretation.

## Sparse-Activation Presentation

- Label direct `x == 0` measurements as exact zero.
- Label near-zero thresholds numerically.
- Show an exact-zero probability atom separately from the density conditional
  on nonzero values.
- Preserve histogram underflow and overflow in denominators and disclose them.
- Use actual-operand counters for logical-product claims.
- Label `R_block` and `R_model` as logical opportunities, not speedups.
- Do not imply latency, throughput, FLOP/s, or energy improvement without a
  separately specified and measured runtime experiment.

## Paper Figure Registry and Provenance

When the definitive plan adds a paper figure, add one row to
[`paper_map.md`](paper_map.md) containing:

- claim or purpose;
- exact config IDs;
- exact pinned run IDs and specialized artifacts;
- exact owning scaffold and numbered `figs/` output filename;
- deterministic regeneration command.

A strict figure suite resolves every input before calling any renderer. On
success it writes a deterministic provenance sidecar containing:

- suite/figure identifiers and output filenames;
- relative source paths;
- source config and run IDs;
- source manifest status and launch Git provenance;
- SHA-256 for every consumed config, manifest, and specialized artifact;
- SHA-256 and size for every generated output;
- no wall-clock timestamp or machine-specific absolute path.

Missing input, source disagreement, renderer failure, or publication-profile
failure prevents promotion and provenance publication.

## Review Workflow

Before replacing a paper artifact:

1. Confirm every source run is terminally verified and pinned.
2. Run focused reduction and plotting tests.
3. Render the complete affected suite into a temporary comparison directory.
4. Inspect inputs, series, values, axes, sample size, uncertainty, limitations,
   text containment, PDF geometry, and PNG rendering.
5. Run `make test` and `make check`.
6. Promote atomically and verify the provenance sidecar.

Generated files beneath every scaffold's `figs/` are ignored by default. Track
only paper artifacts and provenance explicitly selected for release by the
reviewed plan.
