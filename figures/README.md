# Figures

Paper-ready figures generated from saved random-initialized pretraining results go here.

Use sequential figure names:

```text
01-pythia-14m-minipile-random-full-10min-diagnostics.pdf
02-pythia-14m-minipile-clipping-frontier-smoke.pdf
03-pythia-14m-pressure-short-learning-curves.pdf
04-pythia-14m-pressure-short-clipping-frontiers.pdf
```

Regenerate figures with:

```bash
make plots
```

Figures should be reproducible from files under `results/`. Do not rely on notebook-only plotting for paper figures.

## Plotting Standards

Every figure is a research artifact, not decoration. Aim for figures that can go into the paper unchanged.

Use the centralized Matplotlib style in `src/paper_exp/plots.py`. Do not add notebook-only styling that cannot be reproduced by `make plots`.

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
