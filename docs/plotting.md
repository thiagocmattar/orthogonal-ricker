# Plotting Contract

The plotting code is a repository-specific paper-figure library. It is not a
general chart framework: each figure family keeps its scientific cohort,
reductions, axes, and captions explicit, while proven presentation conventions
are shared.

Report 04 is the current visual baseline:

```text
report/04-2026-07-11-post-layernorm-relu-ol1-comparison/
```

The report embeds figures `79`, `80`, `82`, `83`, and `85` through `90`.
Figures `81` and `84` are generated Report 04 diagnostics but are not embedded
in the current PDF. All twelve outputs remain part of the regeneration
contract.

## Module Ownership

| Module | Owns |
| --- | --- |
| `plots.py` | Stable CLI/import facade, batch dispatch, run selection, public `generate_*` wrappers, and legacy figure families not yet extracted |
| `plot_api.py` | Final-size `GridLayout`, count-derived panel grids, one-build PDF/PNG export, and publication-profile validation |
| `plot_catalog.py` | Searchable Report 04 figure type, filename, wrapper, input-kind, and report-embedding metadata |
| `plot_style.py` | Scoped rc parameters, colorblind-safe palettes, stable method IDs/styles, and export defaults |
| `plot_common.py` | Small presentation-neutral helpers already used by multiple figure families |
| `plot_report04.py` | Report 04 cohorts, compute-accounting constants, pure reductions, checkpoint preparation, and explicit renderers for figures `79` through `90` |

Existing callers should continue to import from `paper_exp.plots`. The facade
re-exports Report 04 constants and the public `generate_*` wrappers used by the
dispatcher and existing callers. New family implementation code and numerical
tests should import private helpers from their owning module directly.

## Report 04 Figure Index

Search for the wrapper name in `plots.py` to change selection or export
behavior, and for the corresponding `_plot_*` name in `plot_report04.py` to
change the figure itself.

| Figure | Public wrapper | Saved input |
| --- | --- | --- |
| `79` | `generate_report04_learning_diagnostics` | Training `events.jsonl` |
| `80` | `generate_report04_activation_heatmaps` | Activation histograms |
| `81` | `generate_report04_activation_densities` | Activation histograms |
| `82` | `generate_report04_site_clipping_frontiers` | Per-site clipping frontiers |
| `83` | `generate_report04_joint_compute_frontier` | Joint clipping frontiers |
| `84` | `generate_report04_parameter_diagnostics` | Final checkpoints |
| `85` | `generate_post_layernorm_relu_propagation_heatmaps` | Activation-propagation counts |
| `86` | `generate_post_layernorm_relu_zero_product_heatmaps` | Activation-propagation counts |
| `87` | `generate_report04_three_relu_architecture` | Architecture constants only |
| `88` | `generate_report04_activation_weight_densities` | Histograms and final checkpoints |
| `89` | `generate_report04_layernorm_parameters` | Final checkpoints |
| `90` | `generate_report04_pythia_family_compute_ceiling` | Architecture constants only |

List this catalog from the command line without reading results or writing
figures:

```bash
python -m paper_exp.cli plot-catalog
python -m paper_exp.cli plot-catalog --embedded-only
```

Each deterministic row names the plot type, output file, required saved
artifacts, public wrapper, and whether the current report embeds the figure.
Code can resolve the same entry by figure number, exact filename, `plot_type`,
or public-wrapper name through `get_report04_figure`. Add a new figure type to
this catalog when the suite gains a renderer; this prevents discoverability
from depending on a search through the facade.

Regenerate the visual-baseline suite through its dedicated boundary:

```bash
python -m paper_exp.cli plot-report04 --results results --figures figures --png
```

The default is the published pre-RN visual baseline: seven training methods
and the four-method propagation diagnostic from config `102`. The completed RN
comparison is an explicit opt-in that adds training run `105` and uses the
five-method propagation diagnostic from config `106`:

```bash
python -m paper_exp.cli plot-report04 --include-rn \
  --figures tmp/report04-rn-preview --png
```

Keep that opt-in cohort in a separate preview directory until the report text
is deliberately revised; otherwise its shared filenames would replace figures
whose current captions and conclusions describe the published pre-RN cohort.

Both cohorts are strict by default. The command resolves every selected input
before calling a renderer and reports all missing cohort members together. It
then renders the full suite in a sibling staging directory and promotes it with
rollback only after every figure succeeds, so neither a missing input nor a
late renderer failure can leave a mixed suite. During exploration, the explicit
partial mode keeps the historical behavior of rendering complete figure
families and skipping incomplete ones:

```bash
python -m paper_exp.cli plot-report04 --allow-partial
```

The general `plots` command also uses this partial behavior so its existing
mixed-family dispatch remains backward compatible.

After every successful strict `plot-report04` run, the command atomically
writes `report04-provenance.json` beside the figures. The deterministic sidecar
contains the catalog metadata for figures `79` through `90`, exact run and
artifact paths relative to the selected results directory, and a SHA-256 for
every consumed artifact. Each input also records hashes for the saved
`config.yaml` and `manifest.json`, plus the manifest's launch `git_commit` and
`git_dirty` state. It identifies whether the `published-pre-rn` or
`rn-comparison` cohort was selected and contains no timestamp or absolute
path. Schema version 3 also records the filename, size, and SHA-256 of each
generated PDF/PNG, so the sidecar can detect a stale or mixed artifact set;
figures `87` and `90` have empty input lists. Figure `88` always records
runs `100`/`101` and the exact seven `config_id`/`run_id` checkpoints named by
both pre-RN histogram artifacts; strict preflight rejects disagreement between
the two artifacts instead of substituting a newer checkpoint. Figures `84`
and `89` record seven training checkpoints by default and all eight with
`--include-rn`. Failed strict preflight and `--allow-partial` do not write a
sidecar.
Direct `generate_report04_figures` callers retain the figure-only return
contract unless they explicitly pass `write_provenance=True` with
`strict=True`; the dedicated strict CLI enables it automatically.
The sidecar is generated and ignored by default. Deliberately select it for a
release only after reviewing the figure suite and its input hashes; use
`git add -f figures/report04-provenance.json` when that release decision is
made.

## Data Path

```text
saved artifacts
  -> completed-run selection in plots.py
  -> family generate_* wrapper in plots.py
  -> preparation/reduction in plot_report04.py
  -> explicit Matplotlib renderer
  -> numbered PDF and optional PNG
```

Renderers must not train a model, recompute an experiment, silently substitute
a cohort, or invent a missing value. Important scientific reductions should be
pure and covered by small numerical tests before presentation code consumes
them.

## Shared Visual Contract

Use the Report 04 conventions unless a figure has a documented reason to
differ:

- vector PDF with embedded TrueType fonts and optional 300-DPI PNG;
- an authored width of 7.16 inches and maximum height of 8.8 inches for the
  Report 04 two-column profile, with visible text at least 8 points;
- no `bbox_inches="tight"` for publication exports: the requested canvas is
  the PDF MediaBox and is validated before saving;
- white background and a subtle gray grid;
- color plus marker or linestyle when comparing methods;
- stable method identity across panels;
- a factual subtitle stating budget, sample size, or denominator;
- an interpretation note stating seed uncertainty and relevant limitations;
- frameless legends outside dense data regions when practical;
- direct counters for exact-zero claims rather than histogram-bin inference;
- an exact-zero probability atom shown separately from the density conditional
  on nonzero values; never turn a point mass into a bin-width-dependent spike;
- explicit distinction between logical compute opportunities and measured
  speedups;
- no hidden axis truncation or unlabelled panel-specific scales.

Figure-specific choices stay with the family renderer. Examples include the
representative layer, activation sites, cohort membership, axis ranges,
architecture dimensions, and compute denominators.

## Editing Report 04 Safely

1. Read `docs/methods.md`, the Report 04 rows in `docs/paper_map.md`, and the
   report captions before changing a reduction or label.
2. Change the smallest owning section in `plot_report04.py`. Shared style
   changes belong in `plot_style.py`; do not add local palette variants.
3. Keep artifact-supported cohorts explicit. The default published cohort
   excludes RN and uses propagation run `102`. `--include-rn` adds run `105`
   to the training-event and final-checkpoint figures and switches propagation
   figures `85`/`86` to run `106`. Runs `100`/`101` contain seven pre-RN
   histogram methods, and there are no RN clipping sweeps; figures `80`--`83`
   and the activation half of `88` must not infer or substitute those missing
   diagnostics in either cohort.
4. Grid renderers derive rows from the actual panel count. The public wrappers
   for figures `80`, `82`, `85`, and `86` accept an optional `GridLayout` when
   a different column count or final canvas is needed. Exercise method counts
   `1`, `2`, `4`, `5`, and `7` before changing a cardinality-sensitive layout.
5. Run the focused contracts:

   ```bash
   python -m pytest -p no:cacheprovider \
     tests/test_plot_api.py \
     tests/test_plot_catalog.py \
     tests/test_plot_catalog_cli.py \
     tests/test_plot_report04_cli.py \
     tests/test_report04_contract.py \
     tests/test_report04_math.py \
     tests/test_plot_selection.py
   ```

6. Regenerate all affected PDF and PNG outputs into a temporary directory.
   Compare PNG pixels at the same Matplotlib/font versions and inspect the
   rendered figures. PDF byte hashes include creation timestamps, so compare
   page geometry and rendered content rather than requiring identical hashes.
7. Run the full tests and `python -m paper_exp.cli check` before replacing a
   paper artifact.

## Input Reproducibility

Report 04 currently selects the latest coherent completed run under each
declared experiment ID within the explicitly selected cohort. The strict suite
additionally requires the event and checkpoint artifacts for each training
method to resolve to the same run. Figure `88` is stricter: it resolves each
weight distribution from the exact `config_id`/`run_id` recorded consistently
by both histogram artifacts. This is safe against partial and failed runs, but
a later completed rerun can still change the other data-backed figures. The
provenance sidecar records the cohort, selected run inputs, saved run configs,
manifests, launch Git state, and hashes; it is an audit record, not yet a replay
selector. Before release, paper figure selection should pin the exact selected
run IDs and expected artifacts for the remaining figure families.
Exploratory figures may continue to use latest-completed selection when that
behavior is stated explicitly.
