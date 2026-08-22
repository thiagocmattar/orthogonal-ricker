# Figures

Generated diagnostic and paper figures live here and are ignored by default.
The definitive experiment plan will assign paper figure families and sequential
output prefixes. Until that plan is supplied and reviewed, there are no paper
figures in this branch.

## Explicit Diagnostic Plotting

Render one exact saved run artifact with:

```bash
make plot \
  KIND=run \
  RUN_DIR=results/<config-id>/<run-id> \
  OUTPUT=figures/01-run-diagnostics.pdf \
  PNG=1
```

Other supported kinds are `clipping`, `activation-histograms`,
`weight-histograms`, and `activation-propagation`.

These convenience plots do not select or certify a paper cohort. Paper inputs
must be pinned in [`docs/paper_map.md`](../docs/paper_map.md). Every diagnostic
export also writes a `.provenance.json` sidecar containing the exact source
identity and input/output hashes.

## Naming

Use one unique sequential prefix per canonical paper figure:

```text
01-descriptive-name.pdf
01-descriptive-name.png    # optional inspection copy of the same figure
```

Do not reuse a prefix for a different figure or keep competing canonical
filenames with the same prefix.

## Publication Requirements

- Read only pinned saved artifacts.
- Keep loading, pure reduction, rendering, and export separate.
- Generate PDF and optional PNG from the same Matplotlib `Figure`.
- Use the shared colorblind-safe style and final-size publication checks.
- Show sample size, seed count, denominator, and uncertainty when relevant.
- Label zoomed or logarithmic axes explicitly.
- Separate an exact-zero probability atom from conditional nonzero density.
- Label logical compute opportunities as logical, not measured speedups.
- Stage and inspect outputs before atomic promotion.
- Record deterministic input and output hashes for a paper suite.

The complete contract is in [`docs/plotting.md`](../docs/plotting.md).
