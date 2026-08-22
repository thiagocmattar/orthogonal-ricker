# Experimental Plan

This is not the final experiment plan. Don't treat this as the final case. This is the current v0.

## 1\. Core claims to establish

The experiments should support three claims:

1. **Sparsity spillover:** increasing MLP sparsity through (\\ell\_1) pressure systematically reduces sparsity at untargeted attention sites.
2. **Global consequence:** local MLP sparsity overstates model-wide sparsifiable compute; (R\_{\\text{model}}) captures the actual global effect.
3. **Mitigation:** architectural thresholding, optionally combined with (\\ell\_1) pressure and gradient orthogonalization, improves the validation-loss vs. (R\_{\\text{model}}) frontier.

This matches the current paper framing.

---

# 2\. Standard protocol

All primary experiments:

* train **from scratch**
* dataset: **MiniPile**
* models: **Pythia 14M, 31M, 70M, 160M, 410M**
* matched initialization/data seeds between treatments
* report results after the same number of tokens
* use identical optimizer/scheduler/batch settings within comparisons

### **Run lengths**

| Purpose | Tokens |
| :---- | ----: |
| Smoke/debug | 10M |
| Screening | **400M** |
| Main paper run | **1.7B** |
| Long-training robustness | **3.4B** |
| Pretrained-checkpoint continuation | **300M** |

A screening result must be promoted to 1.7B before becoming a main-paper result.

---

# 3\. Required logging

At fixed checkpoints, preferably:

0, 50M, 200M, 425M, 850M, 1.275B, 1.7B

record for every layer/site:

* validation loss
* exact zero fraction
* near-zero fraction at (|a|\<10^{-2}) and optionally (10^{-1})
* activation RMS
* MLP hidden sparsity
* attention Q/K/V/output sparsity
* residual sparsity
* (R\_{\\text{model}})
* MLP and attention weight norms

For orthogonalized runs also record:

* task-gradient norm
* sparsity-gradient norm
* cosine similarity
* projected-gradient norm

Keep **exact zeros** and **near-zero mass** clearly separated.

---

# 4\. Phase A — Pythia-14M discovery

## A1. Learning rate

Train dense baseline for 1.7B tokens:

\[ \\eta\\in {10^{-5},3\\cdot10^{-5},10^{-4},3\\cdot10^{-4},10^{-3}}. \]

- [ ] 1 seed for full grid
- [ ] repeat best two with 3 seeds
- [ ] select LR using mean final validation loss
- [ ] freeze LR for subsequent 14M experiments

**Output:** appendix LR curve/table.

---

## A2. (\\ell\_1) pressure \+ spillover

At the selected LR:

\[ \\lambda\\in{0,0.1,0.5,1,2,5}. \]

Run:

* plain (\\ell\_1)
* (\\ell\_1) \+ orthogonalization

Each for **1.7B tokens**, initially one seed.

Promote weak, medium and strong (\\lambda) values to **3–5 seeds**.

### **Main expected observation**

\[ \\lambda\\uparrow \\Rightarrow S\_{\\text{MLP}}\\uparrow, \\qquad S\_{\\text{attention}}\\downarrow. \]

Do not require every layer to behave identically.

### **Main Figure 1 — Spillover**

Include:

1. MLP sparsity heatmap: layer × (\\lambda)
2. attention sparsity heatmap: layer × (\\lambda)
3. scatter: \[ \\Delta S\_{\\text{MLP}} \\text{ vs. } \\Delta S\_{\\text{attention}} \]
4. one representative activation-density comparison

This should be the strongest figure in the paper.

---

## A3. Orthogonalization

Compare:

* L1
* L1 \+ orthogonalization

across the same (\\lambda) grid.

Plot:

\[ \\lambda \\rightarrow \\text{validation loss} \]

and

\[ R\_{\\text{model}} \\rightarrow \\text{validation loss}. \]

**Expected result:** orthogonalization allows stronger pressure with less task degradation and reduces sensitivity to (\\lambda).

If this effect is weak, demote it to an appendix result.

---

# 5\. Phase B — architectural thresholding at 14M

## B1. Topology screening

This structural preview predates `A2`. Use the normative 11-row registry in
[`docs/methods.md`](docs/methods.md), where `A2` means the `{m, h}` topology;
do not derive configs from the abbreviated historical list below.

Test:

A0

A1-H

A3

A4-Q

A4-K

A4-V

A5-QK-PRE

A5-QK-POST

A6-PRE

A6-POST

Primary gate:

\[ G^\\pm \]

with:

\[ \\kappa\\in{0.03,0.10,0.30}. \]

Screen each for:

\[ \\boxed{400M\\text{ tokens}} \]

with one seed.

Rank by the Pareto tradeoff:

\[ (\\text{validation loss},R\_{\\text{model}}). \]

Promote approximately **2–3 best topology families** to 1.7B-token runs.

### **Main Figure 2**

Scatter:

\[ x=R\_{\\text{model}}, \\qquad y=\\text{validation loss} \]

with topology labels.

**Question answered:** which placement gives useful global sparsity without excessive quality loss?

---

## B2. Gate form

For promoted topologies compare:

\[ G^+(x)=x1\[x\\ge\\kappa\] \]

vs.

\[ G^\\pm(x)=x1\[|x|\\ge\\kappa\]. \]

Use the same three (\\kappa) values.

- [ ] 400M screening
- [ ] promote representative points
- [ ] 1.7B × 3 seeds for final comparison

**Expected:** (G^+) gives more sparsity but (G^\\pm) preserves quality better.

---

# 6\. Phase C — final 14M frontier

For the best 1–2 architectures compare:

1. dense baseline
2. post-training thresholding
3. L1
4. L1 \+ orthogonalization
5. threshold only
6. threshold \+ L1
7. threshold \+ L1 \+ orthogonalization

Screen combinations at 400M.

Promote only Pareto-relevant configurations to:

\[ 1.7B\\text{ tokens} \\times 3\\text{–}5\\text{ seeds}. \]

### **Main Figure 3 — final frontier**

\[ x=R\_{\\text{model}}, \\qquad y=\\text{validation loss}. \]

**Desired conclusion:** architectural intervention expands the global quality–compute frontier beyond local L1 pressure and post-training thresholding.

---

# 7\. Phase D — scaling

Models:

\[ 31M,;70M,;160M,;410M. \]

Do **not** repeat the full 14M search.

## D1. LR calibration

For each size screen:

\[ {\\eta\_{14}/3,\\eta\_{14},3\\eta\_{14}} \]

for 400M tokens.

Select baseline LR and freeze it for all methods at that scale.

---

## D2. Replicate spillover

At each scale run:

\[ \\lambda= {0,\\lambda\_{\\text{weak}}, \\lambda\_{\\text{medium}}, \\lambda\_{\\text{strong}}}. \]

All headline runs:

\[ \\boxed{1.7B\\text{ tokens}}. \]

Use one seed for the complete sweep and 3 seeds for baseline plus key pressure conditions.

### **Main Figure 4 — Spillover across scale**

Show:

\[ \\Delta S\_{\\text{MLP}} \\text{ vs. } \\Delta S\_{\\text{attention}} \]

for every model.

**Success criterion:** the qualitative spillover relationship persists across model sizes.

It does not need to grow monotonically with scale.

---

## D3. Scale the winning intervention

Freeze the topology/gate selected at 14M.

At each larger size train:

* dense
* L1 only
* threshold only
* full method, quality-preserving setting
* full method, aggressive setting

for **1.7B tokens**.

Do not retune topology at each model size.

### **Main Figure 5**

Quality–(R\_{\\text{model}}) frontier across model sizes.

### **Main Table**

| Model | Dense loss | ΔMLP sparsity | ΔAttention sparsity | L1 (R\_{\\text{model}}) | Full-method (R\_{\\text{model}}) | Δloss |
| :---- | ----: | ----: | ----: | ----: | ----: | ----: |
| 14M |  |  |  |  |  |  |
| 31M |  |  |  |  |  |  |
| 70M |  |  |  |  |  |  |
| 160M |  |  |  |  |  |  |
| 410M |  |  |  |  |  |  |

---

# 8\. Two robustness checks

## E1. Longer training

For 14M and 70M continue:

* dense
* strong L1
* full method

from 1.7B to **3.4B tokens**.

Question:

> Does spillover persist rather than disappear with longer training?

One or two seeds are sufficient.

---

## E2. Pretrained-model adaptation

Use pretrained Pythia 70M and/or 410M.

Continue training for **300M tokens** with:

* control
* MLP L1
* full method

Question:

> Does an already-trained dense model also exhibit spillover when sparsity pressure is introduced?

This is supportive evidence, not the main protocol.

---

# 9\. Paper organization

The final Results section should follow the scientific argument, not experiment chronology:

### **§1. Local sparsity pressure causes spillover**

Figure 1\.

### **§2. Spillover reduces model-wide sparsification gains**

Show local MLP statistic versus (R\_{\\text{model}}).

### **§3. Architectural placement controls global sparsity**

Figure 2\.

### **§4. Architecture \+ optimization improve the frontier**

Figure 3\.

### **§5. The phenomenon and intervention persist with scale**

Figures 4–5 \+ scaling table.

Put LR tuning, full (\\lambda/\\kappa) grids, training curves and weight norms in the appendix.

---

# 10\. Execution rules for the agent

- [ ] Reproduce one existing result before new experiments.
- [ ] Use matched seeds/data ordering.
- [ ] Never select headline results from 400M screening runs.
- [ ] Freeze topology after the 14M study.
- [ ] Do not tune large models independently to produce favorable results.
- [ ] Report all headline points with multiple seeds.
- [ ] Automatically generate figures from raw result files.
- [ ] Keep a manifest containing every run/configuration/status.
- [ ] Do not drop valid negative or divergent runs.
- [ ] Distinguish exact sparsity, near-zero mass, (R\_{\\text{model}}), and actual runtime.
- [ ] Do not claim functional compensation unless directly tested.
- [ ] Do not claim wall-clock acceleration unless measured.

## Priority if compute is limited

**Must have:**

1. 14M L1 spillover
2. 14M topology study
3. 14M final frontier
4. spillover replication 31M→410M
5. winning intervention at 410M

**Then add:**

6. orthogonalization robustness
7. 3.4B-token check
8. pretrained-checkpoint experiment

The paper succeeds if the experiments make one simple result unavoidable:

\[ \\boxed{ \\text{Local MLP sparsity pressure} \\rightarrow \\text{MLP sparsity }\\uparrow ;\\text{but attention sparsity }\\downarrow } \]

and then demonstrate that **measuring and controlling sparsity model-wide produces a meaningfully better quality–compute picture than looking only at the targeted MLPs.**
