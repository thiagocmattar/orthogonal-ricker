# S1-B2 Learned-ATG Engineering Results

## 1. Status and Scope

Configs `211--219` are nine engineering-only learned Adaptive Threshold Gate
(ATG) pilots. Every run uses random-initialized Pythia-14M, the A6-POST
topology, learned one-sided `G+` gates at `a/m/h/q/k/v`, absolute thresholds,
36 independent per-layer/site parameters, `kappa_init=0.10`, and the
hard-forward/soft-backward surrogate. They cross temperature
`tau={0.01,0.03,0.10}` and threshold-learning-rate multiplier
`TLRM={0.1,1,10}` in center-first order.

All nine pilots completed 128 steps and passed the hard checkpoint and
optimizer contract. Config `220`, run
`001-20260719-190523-f28590eb`, completed the pooled selection-partition
propagation diagnostic. The engineering block and its decision are complete.
The 2026-07-19 registered revision selects config `213` at `tau=0.03`, TLRM
`10`, and threshold LR `3e-4`; all 26 S1-B2 configs are materialized and ready.
B2 remains 0/26 complete, so executable scientific completion remains 56/132.

These runs are plumbing evidence, not scientific cells. Validation loss was
checked only for finiteness. It was not ranked and must not select a
temperature, TLRM, or replacement default.

## 2. Pilot Matrix

Every run saw 8,388,608 training tokens with the common 128-step schedule hash
`9d9f708a79511390da9559b88e06e797aa216149af709c841923c56f926e1120`.
The threshold learning rate is the model learning rate `3e-5` multiplied by
TLRM. `Frozen 96/112/128` is the global `atg/frozen_threshold_flag` at the
three final-quarter logging steps. `z_active` is final-training-minibatch
telemetry; it is not the pooled diagnostic endpoint.

| Config | Role | `tau` | TLRM | Threshold LR | Val. loss (finite only) | Mean `abs(kappa-0.10)` | Frozen 96/112/128 | `z_active` |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 211 | preregistered default (vetoed) | 0.03 | 1 | `3e-5` | 10.127606 | 0.000090902 | 1/0/1 | 68.5076% |
| 212 | axial | 0.03 | 0.1 | `3e-6` | 10.127613 | 0.000009019 | 1/1/1 | 68.5065% |
| 213 | registered revision | 0.03 | 10 | `3e-4` | 10.127472 | 0.000912218 | 0/0/0 | 68.5462% |
| 214 | axial | 0.01 | 1 | `3e-5` | 10.127644 | 0.000103059 | 1/0/0 | 68.5267% |
| 215 | axial | 0.10 | 1 | `3e-5` | 10.127662 | 0.000101151 | 1/1/0 | 68.5001% |
| 216 | boundary | 0.01 | 0.1 | `3e-6` | 10.127636 | 0.000010482 | 1/1/1 | 68.4897% |
| 217 | boundary | 0.01 | 10 | `3e-4` | 10.127539 | 0.001033058 | 0/0/0 | 68.6433% |
| 218 | boundary | 0.10 | 0.1 | `3e-6` | 10.127633 | 0.000010140 | 1/1/1 | 68.4768% |
| 219 | boundary | 0.10 | 10 | `3e-4` | 10.127477 | 0.000997896 | 0/0/0 | 68.4626% |

## 3. Formal Default Acceptance Result

The preregistered dynamic gate requires, at each of the final three
final-quarter train events:

- finite, nonzero aggregate threshold gradient, optimizer step, and `kappa`
  step;
- `atg/frozen_threshold_flag == 0` and
  `atg/nonfinite_threshold_flag == 0`;
- pooled active-site exact-zero mass strictly between 0 and 99.5%; and
- no universal collapse across the six active sites.

Config `211` passed every condition except `no_frozen_flag`. Its global frozen
pattern was `1/0/1` at steps `96/112/128`: only `layer_0__v` froze at step 96,
and only `layer_2__a` froze at step 128. Both parameters had nonzero gradients
but an FP32-zero parameter step. Exact model and optimizer reload still passed,
including 36 unique FP32 threshold parameters in a separate zero-weight-decay
optimizer group with finite state. Config `211` is therefore a valid completed
engineering run but a provisional, failed default for the dynamic acceptance
gate.

## 4. Complete-Selection Diagnostic

An exact zero is the direct numeric comparison `x == 0`, with no tolerance.
Config `220` pools integer counts across all 311,296 complete selection tokens,
152 sequences, 38 batches, all six layers, and every tensor coordinate. It
excludes the final 443 incomplete cache tokens. All columns below are pooled
percentages, not averages of batch or layer percentages.

Every row has the same all-six-site A6-POST topology, so
`R_block_max=100%`, `R_model_max=29.9524%`, and `U_arch=R_block`. POST-RoPE Q/K
gate outputs are exactly the QK operands, while V gate outputs are exactly the
PV operands.

| Config | `R_block` | `R_model` | `z_a` | `z_m` | `z_h` | `z_Q_gate` | `z_K_gate` | `z_V_gate` | `z_context_wo` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 211 | 70.8973 | 21.2354 | 52.9980 | 53.0305 | 74.2284 | 70.7056 | 72.6338 | 70.1295 | 24.2633 |
| 212 | 70.9099 | 21.2392 | 53.0176 | 53.0412 | 74.1875 | 70.7705 | 72.6464 | 70.1408 | 24.3384 |
| 213 | 70.8568 | 21.2233 | 52.9701 | 53.0818 | 74.3591 | 70.6032 | 72.8159 | 69.9629 | 23.9282 |
| 214 | 70.8923 | 21.2339 | 52.9968 | 53.0344 | 74.2515 | 70.7009 | 72.7053 | 70.1051 | 24.2029 |
| 215 | 70.8842 | 21.2315 | 52.9965 | 53.0239 | 74.2087 | 70.6824 | 72.5960 | 70.1297 | 24.4112 |
| 216 | 70.8818 | 21.2308 | 52.9698 | 52.9892 | 74.1784 | 70.7039 | 72.6286 | 70.1134 | 24.3946 |
| 217 | 70.8939 | 21.2344 | 52.9719 | 53.0783 | 74.5120 | 70.6643 | 72.9193 | 69.9740 | 23.8759 |
| 218 | 70.8851 | 21.2318 | 52.9955 | 53.0162 | 74.1797 | 70.7031 | 72.5950 | 70.1264 | 24.4060 |
| 219 | 70.8229 | 21.2131 | 52.9675 | 53.0546 | 74.1944 | 70.5728 | 72.6418 | 69.9465 | 23.8219 |

## 5. Engineering Pattern and Registered Decision

- The observed frozen-flag pattern is ordered by TLRM rather than temperature.
  All three TLRM `0.1` rows report `1/1/1`; the three TLRM `1` rows are mixed;
  all three TLRM `10` rows report `0/0/0`.
- Final mean `abs(kappa-0.10)` scales by approximately one order of magnitude
  per TLRM rung: `9.0e-6--1.05e-5` at `0.1`,
  `9.09e-5--1.03e-4` at `1`, and `9.12e-4--1.03e-3` at `10`.
- Despite that update-scale pattern, the 128-step pooled endpoints remain
  narrow: `R_model=21.2131--21.2392%` and
  `R_block=70.8229--70.9099%`. This short engineering screen does not establish
  a quality or sparsity winner.

Registered revision, 2026-07-19: retain the preregistered center temperature
`tau=0.03` and select config `213` at TLRM `10`, giving threshold LR `3e-4`.
This is the smallest one-factor correction to config `211`, and config `213`
passes the original `no_frozen_flag` criterion at all final-quarter points
96/112/128. The choice uses only that preregistered engineering criterion.
Validation loss was checked for finiteness, never ranked, and did not enter the
decision; neither pooled sparsity nor `R_block`/`R_model` selected the default.
The other TLRM `10` temperatures remain boundary observations.

## 6. Artifact Audit

Diagnostic `220` is a schema-v3, statusless-complete five-file result launched
from clean commit `31710d28eabb2058823968d230b0b0038acbedfa`. It pins the nine
canonical completed source runs in exact center-first order. The saved
partition identity, source manifests, learned A6-POST topology, per-layer/site
threshold metadata, exact-zero counts, product denominators, ceilings, and
endpoint arithmetic passed review. The canonical propagation artifact is:

```text
results/220-s1-b2-eng-atg-selection-propagation/001-20260719-190523-f28590eb/activation_propagation.json
SHA256 f603130384684092ccd78c4c842d8ea0e8357a4546ff4c8f1360eed4010a22bf
```
