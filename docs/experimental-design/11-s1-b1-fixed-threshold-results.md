# S1-B1 Fixed-Threshold Results

## 1. Status and Scope

The 24-cell B1 main attention factorial is complete. It crosses fixed gate
family `{G+, Gpm}`, Q/K placement `{PRE, POST}`, scope `{QK, QKV}`, and
`kappa={0.03, 0.10, 0.30}` under AdamW without activation pressure. This is
one-seed, 2,048-step screening evidence: it supports feasibility and matched
within-stratum comparisons, not a global ranking or paper-level conclusion.

All six site-isolation training cells, configs `197--202`, and their pooled
selection-partition diagnostic `203` are complete. The remaining six
branch-scope cells, configs `204--209`, are ready but have not launched.

## 2. Methods and Denominators

`G+` is the one-sided gate
`G+_kappa(x) = x 1[x >= kappa]`; `Gpm` is the signed-magnitude gate
`Gpm_kappa(x) = x 1[abs(x) >= kappa]`. Every row retains ordinary ReLU at
attention input (`a`), MLP input (`m`), and MLP hidden (`h`). The fixed gate is
applied to Q/K and, for QKV rows, V. PRE applies Q/K gates before RoPE; POST
applies them after RoPE. V is always gated after QKV splitting. The final
LayerNorm is ungated. In repository architecture nomenclature, QK rows are
`A5-QK-PRE` or `A5-QK-POST`, while QKV rows are `A6-PRE` or `A6-POST`.

All rows use random-initialized Pythia-14M, LR `3e-5`, seeds `0/0`, 100 warmup
steps, 65,536 tokens/update, and 134,217,728 training tokens. Their common
training-schedule hash is
`db4fa092d7092d29edc3bf1e2005af69f4a92b8bf6e6d88cbb6a166e12be02fc`.
Final loss and endpoint diagnostics use the frozen document-disjoint
`selection` partition, hash
`ffc857a6f0771929dd75c93bc17729de98a692f3a175ac5742cc9d101ff4ea47`:
152 complete sequences, 38 batches, and 311,296 tokens. The final 443 cache
tokens are excluded.

An exact zero is the direct numeric comparison `x == 0`, without a tolerance.
Integer counts are pooled across the complete partition, all six layers, and
all tensor coordinates; percentages are not averages of batch or layer rates.
The pooled denominator is 239,075,328 elements for each width-128 site
`a/m/q/k/v`, and 956,301,312 for width-512 `h`. An em dash means that the gate
is absent. The canonical names `z_Q_gate`, `z_K_gate`, and `z_V_gate` denote
explicit gate-output rates. For POST rows, Q/K gate outputs are exactly the QK
operands; for PRE rows, RoPE transforms those outputs, so they are not the
post-RoPE QK-operand zero rates. V bypasses RoPE, so `z_V_gate` is the exact
zero rate of the V operand consumed by PV in both placements.

`R_block` directly unions exact-zero scalar-product opportunities across QKV,
valid-causal QK, valid-causal PV, Wo, W1, and W2. Its common denominator is
857,085,050,880 products. `R_model` keeps the same numerator and adds the dense
LM head, giving a denominator of 2,861,492,600,832 products. Future causal
positions are excluded. These are logical opportunities, not measured sparse
kernel speedups. QK and QKV topology ceilings are respectively 20.3233% and
29.9524% for `R_model`.

Define topology-normalized utilization as
`U_arch = R_block / R_block_max = R_model / R_model_max`. It measures the
fraction of the selected topology's maximum logical product opportunity that
the observed exact zeros realize; it is not hardware utilization.

| Topology | Rows | `R_block_max` | `R_model_max` | Observed `U_arch` |
| --- | ---: | ---: | ---: | ---: |
| QK | 12 | 67.8522% | 20.3233% | 32.25--67.75% |
| QKV | 12 | 100.0000% | 29.9524% | 26.36--73.26% |

## 3. Canonical Main-Factorial Results

All `R` and `z` columns are percentages. Wall time is the timed training
interval, including scheduled validation. Training config `n` is paired with
its canonical complete-selection diagnostic config `n+1`.

| Config | Gate | Place | Scope | `kappa` | Val. loss | `R_block` | `R_model` | `z_a` | `z_m` | `z_h` | `z_Q_gate` | `z_K_gate` | `z_V_gate` | Wall (h) | Tokens/s |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 149 | G+ | PRE | QK | 0.03 | 7.016120 | 29.0668 | 8.7062 | 51.0683 | 51.1178 | 45.9550 | 18.6264 | 30.6382 | — | 0.3319 | 112,340 |
| 151 | G+ | PRE | QK | 0.10 | 7.019052 | 32.9723 | 9.8760 | 51.0569 | 51.0939 | 45.9486 | 32.2173 | 44.0695 | — | 0.3317 | 112,389 |
| 153 | G+ | PRE | QK | 0.30 | 7.030792 | 45.6220 | 13.6649 | 51.0686 | 51.0680 | 45.9091 | 87.5803 | 90.0063 | — | 0.3329 | 112,004 |
| 155 | G+ | PRE | QKV | 0.03 | 7.017116 | 44.2099 | 13.2419 | 50.9910 | 51.1383 | 45.9741 | 19.7419 | 32.5525 | 50.5459 | 0.3474 | 107,315 |
| 157 | G+ | PRE | QKV | 0.10 | 7.021950 | 52.4094 | 15.6979 | 50.8481 | 51.0304 | 46.0729 | 34.1005 | 44.8574 | 66.3155 | 0.3481 | 107,101 |
| 159 | G+ | PRE | QKV | 0.30 | 7.033689 | 72.8309 | 21.8146 | 50.7284 | 50.9284 | 46.2113 | 87.0436 | 90.6377 | 90.7266 | 0.3479 | 107,167 |
| 161 | G+ | POST | QK | 0.03 | 7.030291 | 33.8808 | 10.1481 | 51.1329 | 51.1524 | 45.9737 | 25.5007 | 37.7821 | — | 0.3312 | 112,554 |
| 163 | G+ | POST | QK | 0.10 | 7.030897 | 37.0560 | 11.0992 | 51.1175 | 51.1430 | 45.9322 | 36.3162 | 49.1377 | — | 0.3315 | 112,451 |
| 165 | G+ | POST | QK | 0.30 | 7.035586 | 45.9678 | 13.7685 | 51.1009 | 51.0964 | 45.9357 | 84.4378 | 87.5561 | — | 0.3331 | 111,941 |
| 167 | G+ | POST | QKV | 0.03 | 7.032897 | 48.7161 | 14.5916 | 51.0204 | 51.1130 | 45.8628 | 25.2456 | 39.1586 | 50.7160 | 0.3493 | 106,730 |
| 169 | G+ | POST | QKV | 0.10 | 7.034079 | 56.4635 | 16.9122 | 50.9640 | 51.0828 | 46.0917 | 37.1150 | 50.2319 | 66.1578 | 0.3483 | 107,056 |
| 171 | G+ | POST | QKV | 0.30 | 7.034050 | 73.2602 | 21.9432 | 50.7972 | 50.9951 | 46.1447 | 84.6225 | 88.4606 | 90.7297 | 0.3487 | 106,931 |
| 173 | Gpm | PRE | QK | 0.03 | 7.013190 | 21.8848 | 6.5550 | 51.1647 | 51.2235 | 45.9977 | 2.8477 | 7.1193 | — | 0.3336 | 111,754 |
| 175 | Gpm | PRE | QK | 0.10 | 7.014335 | 25.3871 | 7.6040 | 51.1710 | 51.2292 | 45.9753 | 8.9391 | 17.9066 | — | 0.3279 | 113,693 |
| 177 | Gpm | PRE | QK | 0.30 | 7.025904 | 40.0691 | 12.0017 | 51.0727 | 51.0838 | 45.9335 | 62.5476 | 67.4954 | — | 0.3276 | 113,801 |
| 179 | Gpm | PRE | QKV | 0.03 | 7.014321 | 26.3649 | 7.8969 | 51.0689 | 51.1318 | 45.9063 | 2.7993 | 7.0718 | 15.8134 | 0.3422 | 108,961 |
| 181 | Gpm | PRE | QKV | 0.10 | 7.017746 | 38.8958 | 11.6502 | 50.9616 | 51.0692 | 46.0486 | 8.1565 | 17.0139 | 48.2179 | 0.3413 | 109,242 |
| 183 | Gpm | PRE | QKV | 0.30 | 7.023521 | 63.8974 | 19.1388 | 50.7059 | 50.9756 | 46.3242 | 57.4434 | 62.3686 | 88.1383 | 0.3416 | 109,154 |
| 185 | Gpm | POST | QK | 0.03 | 7.012649 | 22.1071 | 6.6216 | 51.1389 | 51.2062 | 45.9732 | 2.8265 | 7.0616 | — | 0.3269 | 114,063 |
| 187 | Gpm | POST | QK | 0.10 | 7.014022 | 26.1422 | 7.8302 | 51.1495 | 51.2049 | 45.9782 | 8.8161 | 17.1204 | — | 0.3280 | 113,675 |
| 189 | Gpm | POST | QK | 0.30 | 7.024922 | 40.4946 | 12.1291 | 51.0875 | 51.1133 | 45.9676 | 60.1231 | 65.1524 | — | 0.3328 | 112,023 |
| 191 | Gpm | POST | QKV | 0.03 | 7.013135 | 26.6520 | 7.9829 | 50.9908 | 51.0648 | 45.9225 | 2.8164 | 7.2521 | 15.8840 | 0.3395 | 109,821 |
| 193 | Gpm | POST | QKV | 0.10 | 7.016880 | 39.6462 | 11.8750 | 50.9490 | 51.0581 | 46.0062 | 8.0775 | 16.5758 | 48.3774 | 0.3352 | 111,220 |
| 195 | Gpm | POST | QKV | 0.30 | 7.020426 | 65.1298 | 19.5079 | 50.7280 | 51.0309 | 46.3953 | 57.0075 | 61.7422 | 88.1640 | 0.3495 | 106,681 |

The 24 training rows consumed 8.108 serial GPU-hours.

## 4. Matched Observations

- Raising `kappa` from 0.03 to 0.30 increases `R_model` in all eight matched
  family/placement/scope strata, by 3.6204 to 11.5250 percentage points. The
  matched selection-loss increase is 0.001153 to 0.016573.
- At every one of the 12 matched placement/scope/threshold points, `Gpm` has
  lower loss than `G+` by 0.002795 to 0.019762, and lower `R_model` by 1.6394
  to 6.6087 points. This is a quality--sparsity tradeoff at one seed.
- Adding V to QK increases `R_model` in all 12 matched rows, by 1.3419 to
  8.1747 points. Its loss delta ranges from -0.004497 to +0.003410, so this
  factorial does not show a consistent loss direction for adding V.
- POST has higher `R_model` than PRE in all 12 matched rows, by 0.0666 to
  1.4419 points. POST has higher loss in all six `G+` pairs and lower loss in
  all six `Gpm` pairs; placement therefore cannot be summarized independently
  of gate family in this screen.
- The ordinary branch sites remain narrow across the factorial:
  `z_a=50.7059--51.1710%`, `z_m=50.9284--51.2292%`, and
  `z_h=45.8628--46.3953%`. Most compute variation comes from the explicit
  attention gates.
- QK rows run at 111,754--114,063 tokens/s; QKV rows run at
  106,681--111,220 tokens/s. This is consistent with extra V-gate and
  instrumentation overhead; no sparse speedup can be inferred from these dense
  PyTorch paths.

## 5. Artifact Audit

- Canonical training configs are the 24 odd-numbered configs `149--195`; each
  has a valid `status: completed` manifest, clean launch provenance, 2,048
  completed steps, 134,217,728 tokens, metrics, events, predictions, and a
  durable final model checkpoint.
- Canonical diagnostics are the corresponding even-numbered configs
  `150--196`; each is a valid schema-v3 `statusless_complete` workflow over
  exactly 311,296 selection tokens and has config, metrics, manifest,
  predictions, and `activation_propagation.json` artifacts.
- Every diagnostic pins its source config and exact run id. The table is read
  from those pooled counters and the training endpoints, not from final
  minibatch telemetry.
- Config `203`, run `001-20260719-160449-6ea5e005`, is the canonical pooled
  diagnostic for site-isolation configs `197--202`. It passed source-manifest,
  topology, gate-operand identity, exact-zero, and product-count review over
  all 311,296 complete selection tokens.
- Config `190` attempt 2 is a retained accidental duplicate. It is
  noncanonical and excluded; attempt 1 supplies config `189`'s endpoint.
- Exact run identities and review notes are in
  [`run-registry.yaml`](run-registry.yaml). Config identities are in
  [`config-registry.yaml`](config-registry.yaml).

## 6. Canonical Site-Isolation Results

`Place` denotes Q/K placement and is therefore not applicable to V-only rows:
those runs leave Q/K ungated and gate post-split V directly before PV.
All six rows use fixed absolute `kappa=0.10` at the isolated attention site.

| Config | Gate | Place | Site | Training run | Val. loss | Wall (h) | Tokens/s |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: |
| 197 | G+ | POST | Q | `001-20260719-134717-3cfc0e0f` | 7.021459 | 0.3261 | 114,335 |
| 198 | G+ | POST | K | `001-20260719-141336-5d660aa4` | 7.024131 | 0.3157 | 118,087 |
| 199 | G+ | — | V | `001-20260719-143237-b913af6d` | 7.015473 | 0.3101 | 120,227 |
| 200 | Gpm | POST | Q | `001-20260719-145118-8455717e` | 7.013554 | 0.3090 | 120,670 |
| 201 | Gpm | POST | K | `001-20260719-150955-18a75af8` | 7.014055 | 0.3099 | 120,296 |
| 202 | Gpm | — | V | `001-20260719-152835-0369b367` | 7.015006 | 0.3112 | 119,813 |

All endpoint columns below are pooled percentages from diagnostic `203`.
An em dash means that the explicit gate is absent, not that its dense operand
is unmeasured. The ungated Q/K/V operands contain at most four exact zeros in
their 239,075,328-element pooled tensors.

| Config | `R_block` | `R_model` | `R_block_max` | `R_model_max` | `U_arch` | `z_a` | `z_m` | `z_h` | `z_Q_gate` | `z_K_gate` | `z_V_gate` | `z_context_wo` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 197 | 27.2430 | 8.1599 | 67.8522 | 20.3233 | 40.1506 | 51.1476 | 51.1904 | 45.9628 | 28.0312 | — | — | 0.0000 |
| 198 | 30.2627 | 9.0644 | 67.8522 | 20.3233 | 44.6009 | 51.1246 | 51.1604 | 45.9654 | — | 38.2130 | — | 0.0000 |
| 199 | 38.3045 | 11.4731 | 71.4226 | 21.3928 | 53.6308 | 50.9152 | 51.1259 | 46.0451 | — | — | 65.8719 | 4.1569 |
| 200 | 21.6582 | 6.4871 | 67.8522 | 20.3233 | 31.9197 | 51.1172 | 51.1874 | 46.0077 | 8.0649 | — | — | 0.0000 |
| 201 | 24.2179 | 7.2538 | 67.8522 | 20.3233 | 35.6922 | 51.1453 | 51.2020 | 46.0048 | — | 16.9982 | — | 0.0000 |
| 202 | 33.1142 | 9.9185 | 71.4226 | 21.3928 | 46.3638 | 50.8987 | 51.0137 | 46.0122 | — | — | 48.2980 | 0.0644 |

At this one seed and threshold:

- Within each gate family, Q-only, K-only, and V-only increase `R_model` in
  that order. V-only also has the larger topology ceiling because V zeros can
  reach PV and Wo; this is a topology difference, not a global ranking.
- `G+` realizes 1.5546--1.8105 more `R_model` percentage points than matched
  `Gpm`, while `Gpm` has 0.000467--0.010075 lower selection loss. This repeats
  the main-factorial quality--sparsity tradeoff without establishing a
  population effect.
- POST-RoPE Q/K gate outputs remain exact zeros at the corresponding QK
  operands. Their context outputs are effectively dense. V-only zeros survive
  exactly to the PV operand and produce 4.1569% (`G+`) or 0.0644% (`Gpm`)
  exact-zero attention-context coordinates before Wo.

## TODO: Close S1-B1

- Launch the one-sided branch-scope configs `204--209`: topology
  `{A1-H, A3, A6-POST}` crossed with `kappa={0.03,0.10}`.
- Run the next pooled branch-scope diagnostic (planned config `210`), audit its
  exact-zero/product denominators, and add the six canonical rows here.
- Mark B1 complete only after all 36 scientific cells and both pooled closure
  diagnostics have durable, reviewed artifacts. Preserve the one-seed and
  short-budget interpretation boundary.
