# Figures

Paper-ready figures generated from saved random-initialized pretraining results go here.

## Figure Inventory

[`docs/paper_map.md`](../docs/paper_map.md) is the canonical figure inventory.
It maps each paper purpose to its exact configs, saved results, and numbered
figure outputs. Update that map when adding, replacing, or retiring a paper
figure; do not maintain a second exhaustive filename list here.

Use sequential, unique numeric prefixes:

```text
01-pythia-14m-minipile-random-full-10min-diagnostics.pdf
02-pythia-14m-minipile-clipping-frontier-smoke.pdf
```

Each prefix identifies one canonical figure, even when both PDF and PNG copies
exist. Check both this directory and the paper map before assigning the next
number. Do not reuse a prefix for a different filename or leave competing
canonical variants under the same prefix.

The current visual baseline is Report 04 and figures `79` through `90`:

```text
report/04-2026-07-11-post-layernorm-relu-ol1-comparison/
figures/79-*.pdf through figures/90-*.pdf
```

Preserve their typography, color language, information density, panel spacing,
and compute-accounting clarity when extending the plotting package. A new
figure may differ when its scientific content requires it, but should remain
visually coherent with this family.

## Regeneration Workflow

Regenerate figures with:

```bash
make plots
```

Figures should be reproducible from files under `results/`. Do not rely on notebook-only plotting for paper figures.

Generate changed figures into a temporary comparison directory first, outside
`figures/`. Compare the candidate with the current artifact for exact input
runs, series, labels, axes, sample size or uncertainty, layout, PDF rendering,
and optional PNG rendering. Promote it to `figures/` only after that review;
never overwrite a paper artifact as the first validation step.

## Plotting Standards

Every figure is a research artifact, not decoration. Aim for figures that can go into the paper unchanged.

Use the shared Matplotlib style centralized in `src/paper_exp/plots.py`.
Family-specific loaders and renderers may live in focused modules, but should
reuse that style. Do not add notebook-only styling that cannot be reproduced by
`make plots`.

Honesty rules:

- Do not truncate y-axes to exaggerate effects. If a zoomed view is necessary, label it clearly on the plot.
- Use log scales for heavy-tailed quantities when appropriate, and label the axis as log-scaled.
- Prefer distributions, confidence intervals, percentile bands, or per-run points over means alone when the saved results support them.
- Annotate `n` when comparing groups or summarizing multiple runs.
- Make uncertainty and sample size visible when they affect the interpretation.

Mechanics:

- Save paper figures as PDF. PNG copies are useful for quick inspection.
- Use colorblind-safe palettes. Avoid red/green-only contrasts.
- Label axes with units when units exist.
- Keep titles, labels, and legends readable at paper column size.
- If plotting time-series data, state the timezone or use UTC consistently.
