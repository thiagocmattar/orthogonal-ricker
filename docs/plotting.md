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
| `plot_style.py` | Shared rc parameters, colorblind-safe palettes, method colors and markers, and export defaults |
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
- white background and a subtle gray grid;
- color plus marker or linestyle when comparing methods;
- stable method identity across panels;
- a factual subtitle stating budget, sample size, or denominator;
- an interpretation note stating seed uncertainty and relevant limitations;
- frameless legends outside dense data regions when practical;
- direct counters for exact-zero claims rather than histogram-bin inference;
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
3. Keep mechanical refactors separate from scientific/content changes. In
   particular, integrating RN configs `105` and `106` is not part of the
   extraction that established this module boundary.
   The current joint-frontier annotation layout assumes five series, while the
   propagation layouts require four methods in a 2-by-2 grid. Change those
   cardinalities and their tests deliberately before adding RN to the figures.
4. Run the focused contracts:

   ```bash
   python -m pytest -p no:cacheprovider tests/test_report04_contract.py tests/test_report04_math.py tests/test_plot_selection.py
   ```

5. Regenerate all affected PDF and PNG outputs into a temporary directory.
   Compare PNG pixels at the same Matplotlib/font versions and inspect the
   rendered figures. PDF byte hashes include creation timestamps, so compare
   page geometry and rendered content rather than requiring identical hashes.
6. Run the full tests and `python -m paper_exp.cli check` before replacing a
   paper artifact.

## Input Reproducibility

Report 04 currently selects the latest coherent completed run under each
declared experiment ID. This is safe against partial and failed runs, but a
later completed rerun can still change a figure. Before release, paper figure
selection should pin the exact selected run IDs and expected artifacts.
Exploratory figures may continue to use latest-completed selection when that
behavior is stated explicitly.
