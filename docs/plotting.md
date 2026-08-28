# Plotting Contract

Plots are scientific artifacts. The plotting layer reads saved evidence and
presents it; it does not train a model, rerun validation, choose a scientific
cohort, or repair missing inputs.

The `paper-exp plot` command renders one explicitly named saved run artifact.
The A1 learning-rate screen has a separate tracked fixed-cohort recipe because
the reviewed plan names its exact eleven runs, estimand, selection rule, and
output suite.

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

## A1 Fixed-Cohort Outputs

Regenerate the complete A1 table, curve, and provenance with:

```bash
python -m paper_exp.plots.a1_lr_screen
```

Regenerate the two-panel A1 training-progress figure with:

```bash
python -m paper_exp.plots.a1_lr_screen --progress
```

The recipe pins configs `001`–`011` and their exact accepted run IDs. It
validates tracked/saved config identity, launch provenance, terminal status,
the fixed 1,526-update / 400,031,744-token budget, selection-partition
coverage, and finite final selection loss before applying the predeclared
lowest-loss rule. An exact tie favors the lower peak learning rate.

The all-or-nothing output suite is owned by
`experiments/01-a1-lr-screen/figs/01-a1-learning-rate-screen` and contains
vector PDF, inspection PNG, a complete Markdown table, and deterministic
provenance. It is seed-0 exploratory evidence (`n = 1` per learning rate) at
the fixed 400M-token horizon. If the selected rate is the upper tested
boundary, the companion Markdown states that limitation directly. The curve is
selection-neutral: it uses one visual treatment for every tested rate and keeps
seed, sample-size, horizon, and selection context in the companion caption and
table rather than in an image footnote.

The second output is owned by
`experiments/01-a1-lr-screen/figs/02-a1-training-progress` and contains a
vector PDF, inspection PNG, self-contained Markdown caption, and deterministic
provenance sidecar. Its left panel plots all saved `validation` events as
validation loss versus cumulative training tokens. Its right panel plots the
effective learning rate recorded in all saved `train` events versus cumulative
training tokens, using a base-2 logarithmic y-axis. The event streams contain
nine validation measurements and 154 logged learning-rate values per run. The
learning-rate panel shows those recorded updates without synthesizing a
token-zero point or reconstructing unlogged updates. Both panels use the same
selection-neutral series identities and one shared legend below the plots; the
companion caption records the seed, sample size, horizon, and uncertainty
limit.

## A2 Fixed-Cohort Outputs

Regenerate all three accepted A2 figure packages with:

```bash
python -m paper_exp.plots.a2_spillover
```

The recipe pins pretraining configs `012`-`017`, their exact accepted run IDs,
and activation-histogram diagnostic config `018` run
`001-20260828-082044-a031175f`. It validates tracked/saved config identity,
terminal provenance, complete-selection coverage, checkpoint presence, and the
stored histogram count envelope before reducing any result.

`experiments/02-a2-l1-screen/figs/01-a2-spillover-response` maps the
percentage-point response at `m` against an explicit measured-attention pool
over `{a, q_post, k_post, v}`. Both reductions sum threshold hits and totals
before division. The five positive-lambda cells are directly labeled; the
companion Markdown defines the pool and reports validation loss, `n_h(0.01)`,
`n_m(0.01)`, attention-site responses, and pooled RMS values. This activation
summary is not `R_model` or a logical-product opportunity metric.

`experiments/02-a2-l1-screen/figs/02-a2-layerwise-distributions` is the
supplemental six-layer by six-site atlas. Each panel overlays the ReLU control,
L1 `lambda = 2`, and L1 `lambda = 5`. Exact-zero atoms are removed before an
exact adjacent-bin rebin; all-atom cells are labeled rather than drawn as zero
densities.

`experiments/02-a2-l1-screen/figs/03-a2-site-distributions` is the cleaner
main-paper candidate. Rows are the same three conditions, columns are sites,
and integer counts are pooled across layers only within a site. Every panel
reports its exact-zero atom and mass outside the displayed window.

Both density figures use 0.16-wide count-preserving bins, linear conditional
densities, and identical site-specific x/y scales. No KDE or interpolation is
used. Stored-range and display-window tails remain in the nonzero denominator
and are disclosed in the companion Markdown and provenance. All three packages
contain PDF, PNG, Markdown, and deterministic provenance. They are seed-0
directional evidence (`n = 1` per condition): they do not establish seed
robustness, functional compensation, compute reduction, or runtime speedup.

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
- visible sample size, seed count, or denominator in the figure or companion
  caption;
- uncertainty or per-run points when the saved evidence supports them;
- an explicit figure or companion-caption note for one-seed, exploratory, or
  provisional evidence;
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
