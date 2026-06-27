# Activation Pressure Transfer Handoff

This folder is a transfer package for moving the activation-sparsity methods in this repository into another training or inference codebase. It focuses on the methods that were actually used in the harness rather than the full experiment history.

Read order:

1. `methods/README.md` for the method map and source pointers.
2. `methods/model_porting_contract.md` for the minimum interface a new model or training loop must expose.
3. `methods/activation_sites.md` for the tensors each method can target.
4. `methods/implementation_common.md` for reference pressure functions and the Adam-step orthogonal helper.
5. One method folder under `methods/` for the exact math and porting notes.
6. `metrics/README.md` for additional measures to monitor with activation-pressure methods.
7. `toy_example/README.md` for the synthetic activation-distribution example.
8. `kernel_speedup_tests/README.md` for TEAL latency and sparse-kernel reproduction.

Main methods covered:

- `01_ricker_naive`: direct Ricker activation pressure added to the task loss. This is the deliberately naive comparator and can cause premature task convergence or high-loss plateaus when pressure is too strong.
- `02_l1_naive`: direct mean-absolute-activation pressure added to the task loss. This is the non-orthogonal counterpart of the L1 pressure source used by the harness.
- `03_adam_step_orthogonal_ricker`: canonical Ricker method in this repo. AdamW moments are task-only, and the Ricker correction is projected away from conflicting Adam task-step direction before a trust-budget cap.
- `04_adam_step_orthogonal_l1`: same Adam-step orthogonal machinery as canonical Ricker, but with direct activation L1 pressure.
- `05_activation_clipping`: hard-zero threshold or quantile masking in the model forward pass, plus post-hoc threshold sweeps for inference-style evaluation.

Important interpretation boundary:

- Activation pressure can increase near-zero activation mass, but that is not by itself sparse compute, routing, or speedup evidence. Sparse compute claims require explicit thresholding/gating/operators and latency or FLOP instrumentation.
- Single smoke runs and the synthetic toy example are plumbing or intuition checks, not evidence about representative language-model pretraining.
- Dense AdamW remains the endpoint-quality control. Compare every transferred method against matched AdamW with the same data, model, seed plan, and token budget.

Primary source files:

- `src/lm_harness/sparsity_alm.py`
- `src/lm_harness/task_safe_gradients.py`
- `src/lm_harness/train.py`
- `src/lm_harness/model.py`
- `src/lm_harness/toy_ricker_activation.py`
- `src/lm_harness/teal_latency.py`
- `teal-speedup-agent-handoff/`

Self-containment note:

- The pressure methods and activation clipping are specified in this folder with math, reference snippets, and integration contracts.
- The toy and TEAL sections document how to reproduce this repository's artifacts, but their exact runners/packages live outside this folder. Transfer those source folders too if exact reproduction is required.
