# 01. Pythia-14M Broad Screening Matrix

## 1. Purpose

This is the discovery rung of the scaling ladder. It gives broad coverage of
activation placement, gate family, threshold behavior, pressure family,
orthogonalization, and learning rate while avoiding an uninterpretable full
Cartesian product. The primary matrix has 134 cells; every cell is a
Pythia-14M random-initialization pretraining run.

The screen can identify candidates and failure modes. It cannot establish final
rankings, scale trends, or significance by itself.

## 2. Fixed Screen Contract

Unless a row says otherwise:

| Field | Value |
| --- | --- |
| Model | `EleutherAI/pythia-14m-deduped` architecture, random initialization |
| Dataset/cache | Existing full MiniPile training-token cache |
| Sequence length | 2,048 |
| Optimizer steps | 2,048 |
| Sequences per update | 32 |
| Tokens per update | 65,536 |
| Total training tokens | 134,217,728 |
| Model-initialization/data-order seeds | `0 / 0`; stored separately |
| Central model LR | `3.0e-5` |
| AdamW recipe | Existing harness recipe: betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `0.01` |
| Warmup | 100 steps, then the existing constant-LR behavior |
| Training logging | Step 1, every 50 steps, and the final step |
| Online selection validation | Complete frozen selection partition at step 1, every 250 steps, and the final step |
| Checkpoint | Final model checkpoint; optimizer state is required only where separately specified |
| Final selection validation | Frozen campaign selection partition; see Section 8 |
| Endpoint diagnostics | Complete selection-partition exact-zero and direct logical-product summary for every valid cell |

The 2,048-step budget is about 9.0% of the current 22,762-step
one-cache-pass-equivalent token budget. Sampling uses random contiguous blocks
with replacement, so this is not a literal traversal. It is a fixed token budget
over the same training cache, not a different dataset subset.

## 3. Nomenclature

For block `l`, the compute-facing gate sites are:

- `a`: `attention_inputs`, after the attention LayerNorm and before QKV.
- `m`: `mlp_inputs`, after the MLP LayerNorm and before W1.
- `h`: `mlp_hiddens`, after W1 and before W2.
- `q`, `k`, `v`: separate post-QKV gates; Q/K placement is explicitly PRE or
  POST RoPE, while V is always gated after the split and before PV.
- `c`: the attention context after PV and before the output projection Wo.

Gate families are:

\[
G^+_\kappa(x)=x\,\mathbf 1[x\geq\kappa],\qquad
G^\pm_\kappa(x)=x\,\mathbf 1[|x|\geq\kappa].
\]

- Ordinary ReLU is `G+` at `kappa=0`.
- `Gpm` is the signed symmetric-magnitude gate previously called sReLU.
  `Gpm` at `kappa=0` is the identity apart from signed zero, not ReLU.
- Fixed ATG means fixed `kappa`.
- Learned ATG means hard-forward, soft-backward learned `kappa`.

Use `gplus` and `gpm` in registries and filenames; use `GPLUS` and `GPM` inside
stable design ids. Reserve `G+` and `Gpm` for prose, equations, and reports.

Method codes are:

- `AdamW`: task objective only; gate activations are monitored but unpressured.
- `L1N`: naive L1 activation pressure added to the training objective.
- `OL1`: L1 pressure applied as an Adam-step orthogonal correction.
- `RN`: naive Ricker activation pressure added to the training objective.
- `OR`: Ricker pressure applied as an Adam-step orthogonal correction.

Stable design ids and filenames normalize architecture tokens by removing
internal hyphens: use `A1H`, `A5QKPRE`, `A5QKPOST`, `A6PRE`, and `A6POST`.
The human-facing architecture ids remain `A1-H`, `A5-QK-PRE`,
`A5-QK-POST`, `A6-PRE`, and `A6-POST`.

## 4. Architecture Codes

The following fractional architecture panel isolates the main effects and the
selected interactions without enumerating every subset of seven possible
gates.

| Id | Active gates | Q/K placement | AdamW parent | Question |
| --- | --- | --- | --- | --- |
| `A0` | none; stock GELU | -- | -- | Stock Pythia control |
| `A1-H` | `h` | -- | `A0` | Hidden-MLP ReLU only |
| `A3` | `a,m,h` | -- | `A1-H` | Three-ReLU projection path |
| `A4-Q` | `a,m,h,q` | POST | `A3` | Q-only attention-core contribution |
| `A4-K` | `a,m,h,k` | POST | `A3` | K-only attention-core contribution |
| `A4-V` | `a,m,h,v` | not applicable | `A3` | V/PV contribution |
| `A5-QK-PRE` | `a,m,h,q,k` | PRE | `A3` | Q/K gating before RoPE |
| `A5-QK-POST` | `a,m,h,q,k` | POST | `A3` | Q/K gating after RoPE |
| `A6-PRE` | `a,m,h,q,k,v` | PRE | `A5-QK-PRE` | Six-ReLU PRE architecture |
| `A6-POST` | `a,m,h,q,k,v` | POST | `A5-QK-POST` | Six-ReLU POST architecture |
| `A4-C` | `a,m,h,c` | -- | `A3` | Context gate immediately before Wo |
| `A7-POST-C` | `a,m,h,q,k,v,c` | POST | `A6-POST` | Q/K/V plus post-PV context gate |

`A4-C` and `A7-POST-C` are dependency-gated because the current harness does
not yet implement the post-PV context gate. If that implementation is not
ready, their two cells remain `blocked`; they are not silently replaced.

## 5. Primary 134-Cell Matrix

This is the predeclared design core. `A4-C` and `A7-POST-C` account for two
dependency-gated cells. If the context gate is accepted, execute all 134. If it
is not, freeze a registered 132-cell executable-core revision, keep both rows
blocked, remove context-gate claims, and use 182 rather than 184 as the maximum
design envelope. Do not substitute other cells.

### B0: architecture and model-LR anchors -- 22 cells

Run all 12 architecture rows above with AdamW, ordinary ReLU gates where
present, initialization/data-order seeds `0/0`, and LR `3.0e-5`. For the five core anchors `A0`, `A1-H`, `A3`,
`A6-PRE`, and `A6-POST`, add LR `1.0e-5` and `1.0e-4`.

| Design-id pattern | Cells | Fixed factors | Varied factors |
| --- | ---: | --- | --- |
| `S1-B0-ARCH-<arch>-LR3EM5-S0` | 12 | AdamW, ordinary gates | architecture |
| `S1-B0-LR-<arch>-<lr>-S0` | 10 | AdamW, five core anchors | LR in `{LR1EM5,LR1EM4}` |

The central `3.0e-5` is retained for compatibility with existing runs. It is
close to the Pythia-14M native LR `1e-3` linearly scaled from the official
2,097,152-token batch to 65,536 tokens (`3.125e-5`). The flank values give a
roughly half-decade screen on either side; the batch-scaled value is a campaign
inference, not an official Pythia recommendation.

### B1: fixed-threshold AdamW -- 36 cells

All cells use LR `3.0e-5`, initialization/data-order seeds `0/0`, and no activation pressure.

| Sub-block | Cells | Factors |
| --- | ---: | --- |
| Main attention factorial | 24 | family `{G+,Gpm}` x placement `{PRE,POST}` x scope `{QK,QKV}` x `kappa` `{0.03,0.10,0.30}` |
| POST site isolation | 6 | family `{G+,Gpm}` x scope `{Q,K,V}` at `kappa=0.10` |
| One-sided branch scope | 6 | `G+` topology `{A1-H; A3; A6-POST}` x `kappa` `{0.03,0.10}`; every active gate in the named topology uses the thresholded gate |

The base sites not named in a threshold scope retain the architecture's
ordinary ReLU. Boundary equality survives (`>=`). The `G+_0` controls are in
B0. A QKV-only `Gpm_0` would be identity at Q/K/V and is therefore represented
by the corresponding Three-ReLU architecture rather than mislabeled as ReLU.
Each fixed-threshold row's architecture parent is its same-topology ordinary or
identity-gate AdamW control.

Design ids:

```text
S1-B1-FIX-<GPLUS|GPM>-<PRE|POST>-<QK|QKV>-K<003|010|030>-S0
S1-B1-ISO-<GPLUS|GPM>-POST-<Q|K|V>-K010-S0
S1-B1-BRANCH-GPLUS-<A1H|A3|A6POST>-K<003|010>-S0
```

### B2: learned-ATG AdamW -- 26 cells

All learned gates use `kappa = softplus(rho)` in FP32, no weight decay on
`rho`, hard masks in the forward pass, and a soft mask only for backward
threshold gradients. The screen default is `kappa_init=0.10`, one threshold per
layer/site, transition temperature `tau=0.03`, and threshold LR equal to the
model LR. Absolute and RMS-relative thresholds are separate methods.

| Sub-block | Cells | Factors |
| --- | ---: | --- |
| Main learned attention factorial | 16 | family `{G+,Gpm}` x placement `{PRE,POST}` x scope `{QK,QKV}` x scale `{absolute,RMS-relative}` |
| Learned one-sided branch scope | 6 | scope `{h; a,m,h; all six POST}` x scale `{absolute,RMS-relative}` |
| Granularity control | 4 | `{global,per-site}` for POST-QKV `G+` and `Gpm`, absolute scale; per-layer/site is already in the main block |

Design ids:

```text
S1-B2-ATG-<GPLUS|GPM>-<PRE|POST>-<QK|QKV>-<ABS|RMS>-PLS-K010-S0
S1-B2-ATG-GPLUS-POST-<A1H|A3|A6POST>-<ABS|RMS>-PLS-K010-S0
S1-B2-GRAN-<GPLUS|GPM>-POST-QKV-ABS-<GLOBAL|SITE>-K010-S0
```

For learned branch scopes, `A1-H`, `A3`, and `A6-POST` again mean that every
active gate in the topology is learned and thresholded. The exact `tau` and
threshold-LR defaults above must pass the engineering pilot;
a veto creates a documented design revision before any B2 config is launched.
The pilot must not select these values by comparing validation endpoints.
Learned-versus-fixed causal contrasts are restricted to absolute thresholds,
where B1 supplies the matched fixed row. RMS-relative learned rows are an
unmatched exploratory scale-normalization study until fixed RMS controls are
added in a later registered block.

### B3: pressure and orthogonalization calibration -- 40 cells

Run on two prespecified reference architectures: `A3` and `A6-POST`. The model
LR is `3.0e-5`, initialization/data-order seeds are `0/0`, and the gates are ordinary ReLUs. Pressure targets
every active compute-facing gate: `{a,m,h}` for `A3` and `{a,m,h,q,k,v}` for
`A6-POST`.

| Family | Parameter rows | Methods | Architectures | Cells |
| --- | --- | --- | ---: | ---: |
| L1 | weight `{0.15,1,5}` | `L1N`, `OL1` | 2 | 12 |
| Ricker | seven `(weight,c,sigma)` tuples below | `RN`, `OR` | 2 | 28 |

Ricker tuples:

| Axis | `(weight, c, sigma)` |
| --- | --- |
| Weight | `(0.1,0.1,0.1)`, `(0.3,0.1,0.1)`, `(1,0.1,0.1)` |
| Basin scale | `(0.3,0.05,0.05)`, `(0.3,0.5,0.5)` |
| Shape | `(0.3,0.1,0.05)`, `(0.3,0.1,0.2)` |

RN/OR pairs share the same tuple; L1N/OL1 pairs share the same weight. OL1 and
OR always use `step_budget: 0.5`. The current implementation takes the mean
pressure within each captured tensor and then an equal mean across sites. It is
not weighted by tensor width or model-matmul cost, so this screen must not be
described as directly optimizing `R_model`.

```text
S1-B3-L1-<A3|A6POST>-<L1N|OL1>-W<...>-ALLACTIVE-S0
S1-B3-RK-<A3|A6POST>-<RN|OR>-W<...>-C<...>-SG<...>-ALLACTIVE-S0
```

### B4: seed-pair-1 sentinels -- 10 cells

These are short-run rank-noise sentinels, not confirmation evidence. Rerun the
following exact `0/0` cells with initialization/data-order seeds `1/1`:

1. `A0`, `A1-H`, `A3`, `A6-PRE`, and `A6-POST` AdamW at LR `3.0e-5`.
2. Fixed POST-QKV `G+_0.10` and `Gpm_0.10` AdamW.
3. Learned POST-QKV absolute per-layer/site `Gpm`, `kappa_init=0.10` AdamW.
4. `A6-POST` L1N at weight `1` and OR at `(0.3,0.1,0.1)`.

The method sentinels estimate whether pressure rankings are especially noisy;
they are not substitutes for their matched orthogonal/naive confirmation pairs.

## 6. Conditional Controls -- At Most 50 Cells

These rows are predeclared but do not launch automatically. Their trigger and
scientific role must be recorded before activation.

| Id | Maximum cells | Trigger and factors | Selection eligibility |
| --- | ---: | --- | --- |
| `C1-ATG-PRESSURE` | 12 | On one selected fixed absolute ATG, pressure all active gates: L1 weights `{1,5}` x `{L1N,OL1}`; Ricker `c=sigma` in `{kappa,2*kappa}` x weight `{0.3,1}` x `{RN,OR}` | Eligible as matched gate-pressure panel |
| `C2-METHOD-LR` | 10 | Selected architecture and central method settings at LR `{1e-5,1e-4}` for AdamW/RN/OR/L1N/OL1; omit AdamW cells already present | Secondary robustness; primary common-LR slice retained |
| `C3-BATCH` | 4 | AdamW `A3` and `A6-POST` at effective sequence batch `{16,64}` while keeping 134,217,728 tokens; steps become `{4096,1024}` and warmup becomes `{200,50}` so warmup tokens remain fixed | Diagnostic only |
| `C4-BUDGET` | 4 | One central OL1 and OR setting on ordinary POST and selected ATG with `step_budget=0.25` versus the primary `0.5` rows | Stability/cap control only; never tuned or promoted |
| `C5-SCOPE` | 8 | After C1, central RN/OR/L1N/OL1 on ordinary POST and selected fixed ATG: `QKV-only` versus `all-active-gates`; B3 and C1 supply the all-active rows, so only eight additional QKV-only cells launch | Scope contrast; label explicitly |
| `C6-LEARNED-PRESSURE` | 12 | On one selected learned ATG: L1 weights `{1,5}` x `{L1N,OL1}`; normalized Ricker geometry x two weights x `{RN,OR}` only after normalized `c,sigma` semantics exist | Required before a learned ATG can enter a five-method panel |

Learned or RMS-relative ATGs do not receive Ricker pressure until the Ricker
geometry is defined in the same normalized units. Until C6 is complete, a
learned ATG is eligible only for AdamW/L1N/OL1 architecture screening, not a
five-method promotion panel. The inherited
`kappa=0.1,c=sigma=0.05` OR run remains a negative compatibility control, not a
candidate default.

Operationally, finish and review the registered 132- or 134-cell executable
core before activating these rows. Run only the triggered subset, then freeze
promotion candidates and move to RunPod qualification and S2. Do not delay
scaling to fill an untriggered 182- or 184-cell envelope. See
`05-runpod-cloud.md` for the stage gate and cost plan.

## 7. Count and Runtime Envelope

| Block | Runs | Approximate serial time |
| --- | ---: | ---: |
| B0 architecture/LR | 22 | 6--8 h |
| B1 fixed thresholds | 36 | 10--12 h |
| B2 learned thresholds | 26 | 9--11 h, provisional |
| B3 pressure | 40 | 18--21 h |
| B4 sentinels | 10 | 3--4 h |
| Executable core | **132--134** | **46--56 GPU-h** |
| Conditional controls | at most 50 | 20--24 h |
| Maximum design envelope | **182--184** | **66--80 GPU-h** |

Observed 2,048-step planning times on the current GPU are approximately 16--20
minutes for AdamW/fixed gates, 21--25 for OR/OL1, 28--32 for L1N, and 31--34 for
RN. Learned ATG is provisionally budgeted at 20--24 minutes. This is about three
to four serial wall-clock days for the maximum envelope, before failures,
diagnostics, or human review.

## 8. Engineering and Design Blockers

Complete these before the affected block launches:

1. `TODO:` implement and test fixed `G+` with positive `kappa` at branch and Q/K/V
   sites. Current fixed `Gpm` Q/K/V gating supports PRE/POST and subsets, but
   add explicit PRE/subset round-trip coverage before the large grid. V-only
   configs must still set the validator-required harmless
   `qk_placement: post_rope`.
2. `TODO:` implement learned ATG with exact hard-forward sparsity, safe soft backward,
   FP32 threshold parameters, optimizer parameter groups, metrics, and exact
   checkpoint reload. Learned-gate configs use `checkpoint.save_optimizer:
   true` so optimizer-state round trips are actually tested. Run 128-step plumbing pilots over
   `tau={0.01,0.03,0.10}` and threshold-LR multipliers `{0.1,1,10}`. Pilots are
   engineering evidence only.
3. `TODO:` implement RMS-relative threshold semantics without allowing gradients
   through an unintended batch statistic.
4. `TODO:` implement the optional post-PV context gate for `A4-C` and
   `A7-POST-C`, including a stable activation alias, pressure capture,
   checkpoint reconstruction, propagation/product accounting, graph-union
   ceiling logic, and tests.
5. Selection and campaign-confirmation source-document lists are frozen 250/250
   in `validation-partitions.yaml`. E0.1 must materialize both token caches and
   verify their realized document/token counts and hashes before acceptance.
6. Model-initialization and data-order seeds plus a deterministic training
   schedule hash are implemented. E0.1 must verify that changing only the model
   seed changes the initialization hash while preserving the schedule and
   validation hashes.
7. `TODO:` add dynamic architecture metadata and compute ceilings to the new plotting
   path; do not reuse Report 05's hard-coded `d=128` assumptions for scale plots.
8. The historical budget backtest rejects 2,048 steps as a global rank selector
   (`rho=0.206` in the architecture/method stress cohort). The campaign approved
   the conservative feasibility/collapse and within-stratum rule on 2026-07-18;
   global top-k pruning is prohibited. See `06-s1-budget-backtest.md`.

## 9. Materialization Order

Do not create all configs speculatively. Materialize and commit in this order:

1. E0.1 validation/schedule/seed contract pilots;
2. resolve and register the S1 budget decision from
   `06-s1-budget-backtest.md`;
3. B0;
4. fixed-gate engineering pilots, then B1;
5. learned-gate engineering pilots, then B2 after acceptance;
6. B3;
7. B4 sentinels;
8. only triggered conditional controls.

Each materialized cell receives the next unused sequential config prefix and a
registry row. Stable `design_id` values do not change if config numbers move.

## 10. Learning-Rate Sources

The official Pythia architecture/training table and configuration files are the
source for native family learning rates and batch size:

- <https://github.com/EleutherAI/pythia>
- <https://arxiv.org/abs/2304.01373>
- <https://github.com/EleutherAI/pythia/tree/main/models>

The official values are cited as references. Batch-linear scaling, the flank
grid, and the decision to retain the current harness AdamW recipe are this
campaign's design choices.
