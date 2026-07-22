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

### Fixed-gate engineering gate before B1 -- 2 pilots plus one diagnostic

These are 128-step engineering-only plumbing runs. They are not scientific
screening cells and are not included in either the 36-cell B1 count or the
134-cell primary matrix. Both pilots use the Pythia-14M architecture with
random initialization, AdamW without activation pressure, model LR `3.0e-5`,
model-initialization/data-order seeds `0/0`, 65,536 effective tokens per
update, 128 steps, and 8,388,608 total training tokens. They use the frozen
selection partition and the E0 128-step random-contiguous-block schedule hash
`9d9f708a79511390da9559b88e06e797aa216149af709c841923c56f926e1120`.

| Design id | Architecture and fixed gates |
| --- | --- |
| `S1-B1-ENG-GPLUS-A6POST-K010-ST128-S0` | `A6-POST`; fixed absolute `G+` with `kappa=0.10` at every active site `{a,m,h,q,k,v}`; Q/K gates are POST RoPE |
| `S1-B1-ENG-GPM-A5QKPRE-K010-ST128-S0` | `A5-QK-PRE`; ordinary `G+_0` at `{a,m,h}` and fixed absolute `Gpm` with `kappa=0.10` at `{q,k}`; Q/K gates are PRE RoPE and V is absent |

After both pilots finish, run one pooled complete-selection propagation
diagnostic with stable design id
`S1-B1-ENG-FIXED-GATES-SELECTION-S0`. The diagnostic is also engineering-only
and does not count as a scientific cell.

The fixed-gate engineering gate passes only when all of the following hold:

1. Focused tests cover the `G+` threshold boundary and gradients, PRE and POST
   Q/K placement, Q/K/V subsets including V-only, and checkpoint round trips.
2. Both pilots complete all 128 steps with finite metrics, durable terminal
   artifacts, and clean launch provenance.
3. Checkpoint reload reconstructs the exact gate family, `kappa`, active sites,
   and Q/K placement for each pilot.
4. The pooled diagnostic produces nonzero exact-zero and directly counted
   logical-product opportunities for the configured gates, with neither pilot
   exhibiting universal gate collapse.
5. Validation loss is labeled engineering-only and is not used to choose gate
   families, placements, sites, thresholds, or any other scientific setting.

Only after this gate passes may the B1 scientific configs be materialized.

Acceptance record: configs `146` and `147` and pooled diagnostic config `148`
passed all five criteria on 2026-07-19. The accepted evidence and exact run ids
are recorded in `10-s1-b1-fixed-gate-engineering-results.md`.

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

### Learned-gate engineering gate before B2 -- 9 pilots plus one diagnostic

Before materializing B2, run a 128-step plumbing grid on learned `G+`
`A6-POST`: all six sites `{a,m,h,q,k,v}`, POST-RoPE Q/K, absolute thresholds,
`kappa_init=0.10`, and `kappa_scope=per_layer_site`. Cross transition
temperature `tau={0.01,0.03,0.10}` with threshold/model LR multiplier
`{0.1,1,10}`. Retain the B1 engineering envelope: model LR `3e-5`, seeds
`0/0`, 8,388,608 tokens, no pressure, and final model plus optimizer state.
The preregistered scientific default is `tau=0.03` and LR multiplier `1`;
the grid can veto this default for an engineering failure but cannot replace it
by ranking validation loss.

Launch the center first, then its one-factor neighbors, then the corners:
`(0.03,1)`, `(0.03,0.1)`, `(0.03,10)`, `(0.01,1)`, `(0.10,1)`,
`(0.01,0.1)`, `(0.01,10)`, `(0.10,0.1)`, `(0.10,10)`. After all nine
terminal runs are reconciled, run one pooled complete-selection propagation
diagnostic pinned to their canonical run ids.

Acceptance is plumbing-only. Every run must complete with finite metrics and
an exact model/optimizer reload containing 36 FP32 threshold parameters and a
separate zero-decay threshold optimizer group. Saved/reloaded thresholds must
match exactly. The default must have finite nonzero threshold gradient and
update norms, no nonfinite or frozen-threshold flag, no universal collapse in
its final-quarter logged points, and pooled active-site zero rates strictly
between 0 and 99.5%. Corner collapse or frozen dynamics is retained as boundary
evidence. If the default fails, B2 remains blocked pending a registered design
revision.

Registered design revision, 2026-07-19: config `211`, the original
`(tau=0.03, TLRM=1)` preregistered default, was vetoed solely because its
original `no_frozen_flag` condition failed at steps 96 and 128. The revision
keeps the preregistered center temperature `tau=0.03` and selects the exact
one-factor alternative config `213`, `TLRM=10`, giving threshold LR `3e-4` at
model LR `3e-5`. Config `213` passed `no_frozen_flag` at all final-quarter
points 96/112/128. This is a numerical-resolution correction under the
original engineering rubric: validation loss was checked only for finiteness
and was never ranked or used. All other B2 factors remain preregistered.

### B2: learned-ATG AdamW -- 26 cells

All learned gates use `kappa = softplus(rho)` in FP32, no weight decay on
`rho`, hard masks in the forward pass, and a soft mask only for backward
threshold gradients. The original screen default was `kappa_init=0.10`, one
threshold per layer/site, transition temperature `tau=0.03`, and threshold LR
equal to the model LR. The registered 2026-07-19 revision changes only the
threshold LR to `3e-4` (TLRM `10`); it preserves `tau=0.03`, `kappa_init=0.10`,
and per-layer/site sharing. Absolute and RMS-relative thresholds are separate
methods.

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
active gate in the topology is learned and thresholded. The implementation is
ready for AdamW engineering runs, including detached RMS-relative thresholds.
The exact `tau` and threshold-LR defaults above must still pass the engineering pilot;
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

Execute B3 as five prespecified eight-cell serial tranches. Within each
architecture/parameter point, launch the naive member immediately before its
matched orthogonal member; cover `A3` before `A6-POST`.

| Tranche | Eight cells |
| --- | --- |
| `T1-CENTRAL` | L1 weight `1` and Ricker `(0.3,0.1,0.1)`, both methods and both architectures |
| `T2-L1-FLANKS` | L1 weights `{0.15,5}`, both methods and both architectures |
| `T3-RK-WEIGHT` | Ricker `(0.1,0.1,0.1)` and `(1,0.1,0.1)`, both methods and both architectures |
| `T4-RK-BASIN` | Ricker `(0.3,0.05,0.05)` and `(0.3,0.5,0.5)`, both methods and both architectures |
| `T5-RK-SHAPE` | Ricker `(0.3,0.1,0.05)` and `(0.3,0.1,0.2)`, both methods and both architectures |

> Completed through `t3-rk-weight`: B3 materialized/completed is `32/40` /
> `24/40`. Tranche `t4-rk-basin` configs `275--282` are registered and staged
> but unlaunched; diagnostic `283` remains deferred.

After each tranche, reconcile all eight terminal manifests and run one pooled
complete-selection propagation diagnostic before starting the next tranche.
`T1` is also the fail-fast runtime gate for simultaneous six-site pressure on
`A6-POST`; it is not an additional scientific cell. Allocate each diagnostic
prefix only after its eight canonical source run ids exist.

### B4: seed-pair-1 sentinels -- 10 cells

These are short-run rank-noise sentinels, not confirmation evidence. Rerun the
following exact `0/0` cells with initialization/data-order seeds `1/1`:

1. `A0`, `A1-H`, `A3`, `A6-PRE`, and `A6-POST` AdamW at LR `3.0e-5`.
2. Fixed POST-QKV `G+_0.10` and `Gpm_0.10` AdamW.
3. Learned POST-QKV absolute per-layer/site `Gpm`, `kappa_init=0.10` AdamW.
4. `A6-POST` L1N at weight `1` and OR at `(0.3,0.1,0.1)`.

The method sentinels estimate whether pressure rankings are especially noisy;
they are not substitutes for their matched orthogonal/naive confirmation pairs.
Every B4 row changes only model-initialization/data-order seeds from `0/0` to
`1/1` relative to its exact S0 source; validation remains on the same frozen
selection partition. For 2,048 steps, block size 2,048, micro-batch 4, and
accumulation 8, the required seed-1 schedule hash is
`e3a2079b78a7816ae995c4289aa5946f28677ce50861b346605d42ca167e23a9`.
The schedule is deterministically generated in memory; no separate schedule
cache is required.

| Seed-1 sentinel | Exact seed-0 source |
| --- | --- |
| `A0`, `A1-H`, `A3`, `A6-PRE`, `A6-POST` AdamW | `S1-B0-ARCH-<arch>-LR3EM5-S0` |
| fixed `G+` POST-QKV `kappa=0.10` | `S1-B1-FIX-GPLUS-POST-QKV-K010-S0` |
| fixed `Gpm` POST-QKV `kappa=0.10` | `S1-B1-FIX-GPM-POST-QKV-K010-S0` |
| learned `Gpm` POST-QKV ABS PLS | `S1-B2-ATG-GPM-POST-QKV-ABS-PLS-K010-S0` |
| `A6-POST` L1N weight `1` | `S1-B3-L1-A6POST-L1N-W1-ALLACTIVE-S0` |
| `A6-POST` OR `(0.3,0.1,0.1)` | `S1-B3-RK-A6POST-OR-W0P3-C0P1-SG0P1-ALLACTIVE-S0` |

Materialize B4 only after all ten source cells above are closed. Execute the
ten sentinels as one fail-stop serial tranche and then run one pooled
complete-selection propagation diagnostic pinned to their canonical run ids.

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
learned ATG is eligible only for AdamW architecture screening, not a
five-method promotion panel. Once C6 is activated, its naive pressure rows may
proceed under their registered semantics. Learned OR and OL1 remain blocked
until orthogonal projection and `step_budget` are computed in the true update
space with heterogeneous optimizer-group learning rates. The inherited
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

1. Completed 2026-07-19: fixed positive-`kappa` `G+` is implemented and tested
   at branch and Q/K/V sites; fixed `Gpm` PRE/POST subsets, V-only placement,
   and checkpoint round trips are covered. V-only configs still set the
   validator-required harmless `qk_placement: post_rope`.
2. Completed 2026-07-19: learned ATG has exact hard-forward sparsity,
   threshold-only soft backward, FP32 threshold parameters, zero-decay optimizer
   groups with LR multipliers, training metrics, dynamic propagation metadata,
   and exact model/optimizer checkpoint round trips. AdamW engineering is ready.
   The 128-step `tau={0.01,0.03,0.10}` by threshold-LR multiplier
   `{0.1,1,10}` gate is complete. The registered revision selects
   `(tau=0.03, TLRM=10)` solely by the original `no_frozen_flag` engineering
   criterion; validation loss was never ranked.
3. Completed 2026-07-19: RMS-relative ATG uses a detached full-gate-tensor RMS
   statistic, so threshold learning cannot flow through the normalization.
4. `TODO:` before learned OR or OL1 launches, include heterogeneous optimizer-group
   learning rates in orthogonal update-space projection, norms, and
   `step_budget`. Pure AdamW learned-ATG pilots are unaffected.
5. `TODO:` implement the optional post-PV context gate for `A4-C` and
   `A7-POST-C`, including a stable activation alias, pressure capture,
   checkpoint reconstruction, propagation/product accounting, graph-union
   ceiling logic, and tests.
6. Selection and campaign-confirmation source-document lists are frozen 250/250
   in `validation-partitions.yaml`. E0.1 must materialize both token caches and
   verify their realized document/token counts and hashes before acceptance.
7. Model-initialization and data-order seeds plus a deterministic training
   schedule hash are implemented. E0.1 must verify that changing only the model
   seed changes the initialization hash while preserving the schedule and
   validation hashes.
8. `TODO:` wire saved dynamic gate metadata and scale-specific compute ceilings
   into the new plotting path; do not reuse Report 05's hard-coded `d=128`
   assumptions for scale plots.
9. The historical budget backtest rejects 2,048 steps as a global rank selector
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
