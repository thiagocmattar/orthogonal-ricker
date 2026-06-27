# Kernel And Speedup Tests

Primary source package:

```text
teal-speedup-agent-handoff/
```

Harness wrapper:

```text
src/lm_harness/teal_latency.py
docs/TEAL_LATENCY_WORKFLOW.md
```

Purpose:

This folder tells a transfer agent how to repeat activation sparsity versus latency experiments after a method produces threshold-tolerant activations. The kernel tests are separate from training. They answer whether exact zeros at a chosen tensor and shape can produce real latency wins.

Self-contained boundary:

- This document contains the benchmark protocol, request schema, interpretation rules, and integration path.
- Exact TEAL reproduction requires transferring `teal-speedup-agent-handoff/` alongside this folder or vendoring that package into the target repository.

There are two repeatable tracks:

- Operator track: benchmark arbitrary linear projection shapes as batch-1 decode GEMV. This is the "general matmul" proxy in this handoff, but it does not cover batched training GEMM.
- Model-inference track: wrap selected `nn.Linear` modules with fixed-threshold sparse decode kernels and measure dense model versus sparse model decode throughput.

TEAL operation:

```text
y = W @ sparse(x)
```

The package benchmarks batch-1 decode-style GEMV. It thresholds `x` inside the sparse path and skips zeroed columns. It is not a training-speedup benchmark and not an end-to-end generation benchmark by default.

Setup from the handoff package:

```powershell
cd teal-speedup-agent-handoff
py -m venv .venv-teal
.\.venv-teal\Scripts\python.exe -m pip install --upgrade pip
.\.venv-teal\Scripts\python.exe -m pip install -r requirements-windows-cu128.txt
.\.venv-teal\Scripts\python.exe -m pip install -e .
```

Linux/WSL uses `requirements-linux-cu128.txt` instead.

Recommended environment:

```powershell
$env:TRITON_CACHE_DIR = "$PWD\.triton-cache"
```

Validation:

```powershell
.\.venv-teal\Scripts\python.exe -m pytest tests
```

Quick smoke benchmark:

```powershell
.\.venv-teal\Scripts\python.exe -m teal_speedup.benchmark_latency --points 5 --warmup 3 --rep 5
```

Full latency reproduction:

```powershell
.\.venv-teal\Scripts\python.exe -m teal_speedup.benchmark_latency --points 101 --warmup 10 --rep 30
```

General projection-shape benchmark:

```powershell
.\.venv-teal\Scripts\python.exe -m teal_speedup.benchmark_latency --in-size 4096 --out-size 14336 --points 101 --warmup 10 --rep 30
```

Use `--in-size` as the activation width and `--out-size` as the output width of the downstream linear projection. Report the shape beside every speedup number.

Expected package outputs:

```text
benchmark_results/teal_latency_vs_sparsity.csv
benchmark_results/teal_latency_vs_sparsity.png
```

Important kernel entry points:

- `teal_speedup.sparse_gemv.make_column_major`
- `teal_speedup.sparse_gemv.singlepass_threshold_sparse_gemv`
- `teal_speedup.sparse_gemv.adaptive_sparse_gemv`
- `teal_speedup.sparse_gemv.teal_sparse_linear`
- `teal_speedup.llm_integration.FixedThresholdSparseLinear`

Layout requirement:

The sparse kernels expect the weight in the layout produced by:

```python
weight = weight.T.contiguous().T
```

LLM integration path:

- Start with MLP-only decode.
- Patch gate/up/down projections for SwiGLU-like MLPs if the target model has them.
- Use sparse path only for batch size `1` and `seq_len == 1`.
- Fall back to dense for prefill, larger batch, CPU tensors, non-FP16 tensors, or unsupported shapes.
- Use `FixedThresholdSparseLinear` for fixed-threshold experiments.

Minimal model-inference recipe:

1. Load the dense model in FP16 and record dense batch-1 decode tokens/sec.
2. Choose one projection family, usually MLP down first.
3. Replace only that projection family with `FixedThresholdSparseLinear` at the selected threshold.
4. Keep attention and all other layers dense for the first comparison.
5. Measure sparse-model batch-1 decode tokens/sec with the same prompt length, generation length, cache settings, and hardware.
6. Report quality threshold, achieved activation sparsity, operator speedup, and end-to-end decode speedup separately.

Harness request format:

`python run.py teal-latency` consumes request JSON rows. Required fields:

```json
{
  "rung": "rung2",
  "source": "checkpoint_or_dense_baseline",
  "variant": "ricker_or_l1_or_dense",
  "method": "task_only_adamw_step_orthogonal_ricker",
  "target_sparsity": 0.9,
  "threshold": 0.05,
  "achieved_sparsity": 0.9,
  "val_loss": 2.34,
  "in_features": 4096,
  "out_features": 14336,
  "n_layer": 32
}
```

Dense baseline rows should use:

```json
{
  "source": "dense_baseline",
  "threshold": null,
  "achieved_sparsity": 0.0
}
```

Optional fields used in current workflows:

- `request_id`
- `projection`
- `projection_count`
- `benchmark_kind`

Experiment chain:

1. Train or load a dense/Ricker/L1 checkpoint.
2. Run a post-hoc activation clipping sweep on the relevant sites.
3. Select thresholds that preserve validation loss within the declared tolerance.
4. Record achieved sparsity and the downstream linear shape.
5. Build TEAL request rows for dense and sparse candidates.
6. Run latency benchmark.
7. Report quality and latency separately.

Current calibration from repo docs:

- Small `2048 -> 512` GEMV is overhead-bound.
- Large `4096 -> 14336` MLP-like shape reached about `4.84x` near `90%` sparsity.
- Very large 70B/405B-like MLP shapes reached about `5.6x` to `5.7x` near `90%` sparsity.

Do not overclaim:

- The benchmark is batch-1 decode GEMV unless extended.
- The benchmark is an arbitrary linear-shape operator proxy, not a proof for batched GEMM or training matmul.
- It does not prove training speedup.
- It does not prove end-to-end model speedup until model integration and full decode timing are measured.
- Dynamic unstructured sparsity can lose to dense kernels at lower sparsity or small shapes because indexing and launch overhead dominate.
