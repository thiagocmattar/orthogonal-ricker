# Toy Ricker Activation Distribution Example

Source folder:

```text
exploratory/20260625_toy_ricker_activation_distribution/
```

Config and runner:

```text
configs/ablations/20260625_toy_ricker_activation_distribution/
src/lm_harness/toy_ricker_activation.py
tests/test_toy_ricker_activation.py
```

Purpose:

This is a synthetic, single-batch random-regression example designed to visualize how Adam-step orthogonal Ricker and Adam-step orthogonal L1 reshape activation distributions. It is not language-model evidence and should not be used for endpoint-quality claims.

Self-contained boundary:

- This document contains the design, parameters, expected outputs, and interpretation.
- Exact artifact regeneration requires the runner and source files listed above, or a reimplementation from the design below.

Design:

- Data: `512` random inputs with `x_ij ~ N(0, 1/64)`.
- Targets: `y_i ~ 0.1 * N(0, 1)`.
- Model: `64 -> 512 -> 1` MLP with GELU.
- Optimizer: AdamW, learning rate `0.01`, weight decay `0.001`.
- Ricker pressure site: first-layer pre-activations `z = fc1(x)`.
- L1 pressure site: post-activation hidden tensor `phi(z)`.
- Ricker parameters: `c = 0.06`, `sigma = 0.06`, weight `3.0`, step budget `2.5`.
- L1 uses the same Adam-step orthogonal correction helper with `activation_l1_pressure`.

The source default has `steps = 5000`. The latest regenerated handoff figure used:

```powershell
.\.venv\Scripts\python.exe configs\ablations\20260625_toy_ricker_activation_distribution\run_toy_ricker_activation_distribution.py --steps 200 --record-interval 1
```

Plan-only inspection:

```powershell
.\.venv\Scripts\python.exe configs\ablations\20260625_toy_ricker_activation_distribution\run_toy_ricker_activation_distribution.py --plan
```

Expected runtime:

- About 1-2 minutes on CPU for the current scripted artifact generation, depending on step count and plot generation.
- This is not a harness training run, but keep it labeled as a toy visualization.

Primary outputs:

- `activation_pressure_comparison.png/pdf`: combined figure with Ricker vs AdamW, L1 vs AdamW, and train MSE curves.
- `activation_distributions.png/pdf`: Ricker versus AdamW two-panel distribution figure.
- `activation_l1_distributions.png/pdf`: L1 versus AdamW two-panel distribution figure.
- `activation_sweep.png/pdf`: Ricker pressure sweep visualization.
- `activation_histograms.csv`, `activation_l1_histograms.csv`, `activation_sweep_histograms.csv`: shared-bin histogram densities.
- `results.json`, `activation_l1_results.json`, `activation_sweep_results.json`: metrics and plotting payloads.
- `index.html`: browser-viewable artifact index.

Latest reported distribution summary:

| Variant | Pre hard | Pre gap | Pre active | Post hard | Post gap | Post active | Relative MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AdamW | 12.25% | 65.77% | 21.98% | 23.57% | 63.28% | 13.15% | baseline |
| Adam-step orthogonal Ricker | 91.31% | 3.40% | 5.30% | 92.03% | 3.71% | 4.26% | `4.577e-04` |
| Adam-step orthogonal L1 | 84.53% | 9.89% | 5.59% | 86.67% | 9.25% | 4.08% | `1.442e-05` |

Interpretation:

- The toy confirms that both pressure types can produce high near-zero activation mass in a controlled MLP.
- Ricker produces a stronger hard-zero distribution shift in this setup.
- L1 has lower relative MSE in this toy setup.
- The Ricker distribution success gate was reported as false in the current artifact summary, so do not overstate the example as a clean success criterion.

Porting value:

- Use this toy before porting into a full LM to verify optimizer ordering, projection behavior, and pressure-gradient extraction.
- Preserve the AdamW-only control, the Ricker pressure variant, and the L1 pressure variant.
- Verify that pressure does not enter AdamW moments in the orthogonal variants.

Optional validation:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_toy_ricker_activation.py -q
```
