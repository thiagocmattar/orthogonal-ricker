# 05. RunPod Transition and Cost Plan

## 1. Decision Sequence

The campaign proceeds in this order:

```text
S1 core -> triggered conditional controls -> freeze candidates
        -> RunPod qualification -> S2/S3 token scaling
        -> S4/S5 Pythia-family scaling
```

The 182--184-cell maximum design envelope is a ceiling, not a batch that
launches by default. The discovery design contains 134 cells, of which two
post-PV context rows depend on context-gate implementation. The executable core
is therefore 132 cells under a registered no-context revision or 134 cells with
that implementation accepted. After reviewing it, launch only conditional
controls whose predeclared trigger fired. If no trigger fires, proceed directly
to candidate freezing and cloud qualification.

| Stage | Target | What it must answer | Exit condition |
| --- | ---: | --- | --- |
| S1 executable core | 132--134 cells | Which topology, placement, gate family, threshold scale, LR, pressure family, and orthogonalization settings remain viable at 2,048 steps? | The context decision is registered; valid artifact envelopes and selection diagnostics exist; blockers and exclusions are recorded; Pareto families and required matched controls are identifiable |
| Triggered envelope | 0--50 cells | Resolve only compatibility, LR, batch, cap, pressure-scope, or learned-gate questions exposed by the core | Every activated trigger has a recorded decision; the promotion panels can be frozen without an unresolved causal bridge |
| S2/S3 | 14M token ladder | Do short-run rankings survive more tokens and fresh confirmation seeds? | S3 primary contrasts and cohorts were frozen before confirmation was opened |
| S4/S5 | 31M--410M family ladder | Do the selected quality/sparsity effects and compute opportunities transfer with model size? | Per-size feasibility and LR calibration pass before each full panel |

If the context implementation is not accepted, keep its rows explicitly
blocked, remove context claims, and register the 132-cell core revision. Do not
substitute a different architecture silently.

## 2. Core Target

The core is broad discovery, not final evidence. It contains:

- `B0`, 22 cells: Stock, One-/Three-/Six-ReLU, PRE/POST, single-Q/K/V, and
  context architecture anchors plus model-LR flanks on five core topologies.
- `B1`, 36 cells: fixed `G+` and `Gpm` threshold surfaces over placement,
  QK/QKV scope, site isolation, and branch scope.
- `B2`, 26 cells: learned ATG placement, scope, absolute/RMS scale, and
  threshold granularity under AdamW.
- `B3`, 40 cells: matched L1N/OL1 and RN/OR pressure calibration on `A3` and
  `A6-POST`, with primary orthogonal `step_budget: 0.5`.
- `B4`, 10 cells: seed-1 sentinels for short-run rank-noise detection.

The core should identify viable families and failure modes, estimate
architecture cost versus same-topology AdamW, estimate pressure cost versus
same-topology AdamW, and expose whether 2,048 steps are a useful selector. It
does not select a single universal winner.

## 3. Conditional Maximum Envelope

Activate a row only after writing its trigger and matched-control dependency in
the registry.

| Block | Maximum | Activation trigger | Question resolved |
| --- | ---: | --- | --- |
| `C1-ATG-PRESSURE` | 12 | A fixed absolute ATG is viable and could be promoted | Is its sparsity response preserved under a complete matched pressure panel? |
| `C2-METHOD-LR` | 10 | The selected method ordering is plausibly LR-sensitive or a common-LR result is unstable | Is the apparent method effect an LR interaction? |
| `C3-BATCH` | 4 | Batch/step-count dependence could explain a selected short-run result | Does the result survive fixed-token batch controls? |
| `C4-BUDGET` | 4 | OR/OL1 cap binding, exploding pressure, or update instability is observed | Is `step_budget: 0.5` a stable control? This row never tunes the primary budget. |
| `C5-SCOPE` | 8 | A pressure-gated candidate survives and QKV-only versus all-active attribution matters | Where must pressure act to produce the observed compute opportunity? |
| `C6-LEARNED-PRESSURE` | 12 | A learned ATG is viable for complete-panel promotion and normalized Ricker semantics exist | Can learned thresholding enter a matched five-method panel? |

The full 50-cell envelope is run only if every trigger independently fires.
Otherwise the untriggered rows remain predeclared and unlaunched with their
reason recorded.

## 4. When to Move to RunPod

Prefer local execution for the S1 core. Start cloud engineering during S1 so it
does not block promotion, but do not launch a scientific cloud cell until the
readiness gates below pass.

1. Complete and review the registered 132- or 134-cell S1 executable core
   locally.
2. Run only the conditional controls required to freeze candidates.
3. Freeze the candidate panels and their config, cache, schedule, and
   validation hashes.
4. Qualify RunPod with matched E0 and S1 rehearsal runs.
5. Use RunPod for S2 when parallel turnaround is worth the cost; move no later
   than S3 so cloud execution is not first tested at a larger model. Save and
   evaluate step 2,048 inside every S2 run so primary rank survival is measured
   on one cloud SKU/image rather than across local and cloud platforms.
6. Treat S4/S5 as cloud stages unless a measured local ETC is acceptable.
7. At every new model size, run a 10-step fit/checkpoint smoke test and a
   128-step timing calibration before pricing or launching the full panel.

This is a single-GPU campaign through 410M. Independent seeds can occupy
separate one-GPU Pods. DDP or another multi-GPU path is a new implementation
and control, not an infrastructure-only retry.

## 5. Current Cloud Blockers

The current harness is not ready for long or interruptible cloud runs:

- training saves only `checkpoints/final/`; there is no periodic atomic
  checkpoint or exact resume route;
- optimizer, LR/warmup, RNG, data cursor/schedule, event offset, and learned-gate
  optimizer-group state are not all recoverable from an interrupted run;
- a pod termination can leave a manifest permanently `status: running`;
- dependency bounds are broad and no tracked environment lock or container
  digest exists;
- the current token-cache metadata stores Windows paths, uses mutable `main`
  revisions, and lacks content hashes for every campaign partition;
- no artifact inventory/upload/verification command exists, while diagnostics
  currently expect local checkpoints.

Before the first scientific cloud cell, complete these gates:

1. `TODO:` make cache metadata OS-portable, pin dataset/tokenizer/model-config
   revisions, and record byte counts and SHA-256 hashes for training, selection,
   and confirmation data.
2. `TODO:` track a dependency lock and a pinned image tag plus digest; capture
   Python, PyTorch, CUDA, cuDNN, driver, image, and Git provenance.
3. `TODO:` add atomic periodic `step-N` checkpoints and exact resume, including
   optimizer, schedule, RNG/data cursor, learned gates, and event position.
4. `TODO:` force-interrupt and resume a 128-step control within a predeclared
   numerical tolerance, without duplicate events or token accounting.
5. `TODO:` produce an artifact inventory of relative path, byte count, and
   SHA-256; upload to a temporary prefix, verify, and publish a completion marker
   last.
6. `TODO:` add stale-run reconciliation that marks an abandoned attempt
   interrupted/failed only after confirming that no process remains; preserve
   and link every retry.

On-demand Pods are required until all resume gates pass. Spot Pods can be
stopped at any time and are therefore blocked for this campaign's current
final-checkpoint-only runner.

## 6. RunPod Deployment Profile

Use RunPod Pods, not Serverless. Start with one on-demand GPU per Pod.

- Use Secure Cloud for qualification and long confirmation runs. It supports
  network volumes and reduces infrastructure variability.
- Benchmark Community Cloud RTX 4090/5090 for S2 and other short, restartable
  batches after parity passes; choose by measured cost per valid completed run,
  not hourly price alone.
- For 48 GB, compare L40S with RTX A6000/A40 when they fit the same software
  path. For 410M, fit-test L40S first and compare against A100 80 GB.
- Use the same GPU SKU, image digest, precision, and software stack for every
  member of a matched panel. Cross-SKU tokens/s values are not method effects.
- Do not buy a three- or six-month savings plan during discovery. Reconsider
  only after the sustained S5 workload and chosen SKU are measured.

For Secure Cloud with a network volume, use:

```text
/workspace/shared/data/tokenized/<cache-id>/       # immutable, read-only
/workspace/workers/<pod-id>/repo/                  # clean detached checkout
/workspace/artifacts/<campaign>/<config>/<run>/    # per-Pod publish staging on shared volume
```

Parallel Pods never share a writable Git worktree or run directory. A network
volume is active-work storage, not the archive of record; publish verified
artifacts to durable external storage and retrieve scalar artifacts locally.

## 7. Qualification and SKU Selection

Before S2, run the same hashed cache and schedule on the local RTX 5070 Ti and
candidate RunPod GPUs:

1. one 128-step AdamW parity config;
2. one 128-step highest-memory architecture/pressure config;
3. the forced-interruption/resume test after resume exists;
4. one 2,048-step representative timing run on each finalist.

Verify initial shared-parameter hashes, batch-schedule hash, finite losses,
checkpoint reload, exact-zero/product counters, manifest lifecycle, and
artifact retrieval. Freeze the acceptable numerical tolerance before comparing
endpoints.

Record peak memory, tokens/s, setup time, billed seconds, completion rate, and
cost per valid run. Select the least expensive SKU that passes memory,
reliability, numerical, and turnaround constraints. Do not infer throughput
from advertised peak FLOPs; the 14M workload is sensitive to launch and Python
overhead.

## 8. Price Snapshot

The following RunPod on-demand list prices were checked on **2026-07-18**.
Availability and live deployment prices vary by region and must be refreshed at
launch.

| GPU | VRAM | Community Cloud | Secure Cloud |
| --- | ---: | ---: | ---: |
| RTX 4090 | 24 GB | $0.34/h | $0.69/h |
| RTX 5090 | 32 GB | $0.69/h | $0.99/h |
| RTX A6000 | 48 GB | $0.33/h | $0.49/h |
| A40 | 48 GB | $0.35/h | $0.44/h |
| L40S | 48 GB | $0.79/h | $0.99/h |
| A100 SXM | 80 GB | $1.39/h | $1.49/h |

Standard network storage is currently $0.07/GB-month through 1 TB. Thus 100 GB
is about $7/month and 250--500 GB is about $17.50--$35/month. RunPod storage is
not the long-term archive; external archive charges are separate.

Sources:

- <https://www.runpod.io/pricing>
- <https://www.runpod.io/gpu-models>
- <https://docs.runpod.io/pods/pricing>
- <https://docs.runpod.io/pods/overview>
- <https://docs.runpod.io/storage/network-volumes>
- <https://docs.runpod.io/pods/storage/types>
- <https://docs.runpod.io/pods/manage-pods>

## 9. Cost Model

For a calibrated batch,

\[
C_{\rm budget}=1.20\sum_g H_g r_g+C_{\rm storage}+C_{\rm archive},
\]

where `H_g` is measured GPU wall-hours, `r_g` is the live on-demand rate, and
20% is a planning reserve for setup, retries, and finalization. Actual reported
cost uses billed seconds and excludes the reserve.

For 14M, translate existing local estimates only after measuring

\[
f_{14}=\frac{t_{\rm same\ config,cloud}}
{t_{\rm same\ config,local}},\qquad
H_{\rm cloud}=f_{14}H_{\rm local}.
\]

The next table is a rate-conversion scenario at `f_14=1.0` and 20% reserve. It
is not a RunPod throughput benchmark; multiply its dollar ranges by the
measured `f_14`. Community RTX 4090 at $0.34/h is a lower-price alternative for
short restartable batches after reliability/resume qualification. Secure RTX
4090 at $0.69/h is the planning basis for long confirmation runs.

| Batch | Runs | Existing local work estimate | Community 4090 scenario | Secure 4090 scenario |
| --- | ---: | ---: | ---: | ---: |
| S1 executable core | 132--134 | 46--56 h | $18.8--$22.8 | $38.1--$46.4 |
| All conditional controls, incremental | at most 50 | 20--24 h | $8.2--$9.8 | $16.6--$19.9 |
| S1 maximum total | 182--184 | 66--80 h | $26.9--$32.6 | $54.6--$66.2 |
| S2 maximum | 52 | 80--105 h | $32.6--$42.8 | $66.2--$86.9 |
| S3 minimum | 54 | 230--270 h | $93.8--$110.2 | $190.4--$223.6 |
| S3 maximum | 90 | 405--445 h | $165.2--$181.6 | $335.3--$368.5 |

For new model sizes, use a 128-step measured sustained throughput rather than a
14M time ratio. The analytical planning approximation is

\[
F_{\rm run}\approx6N_{\rm tokens}C_{\rm model},\qquad
t_{\rm AdamW}\approx\frac{F_{\rm run}}
{3.6\,\theta_{\rm eff}},
\]

with `F` in PFLOP and measured `theta_eff` in sustained TFLOP/s. Observed local
method-time factors relative to AdamW are approximately 1.21 for OL1, 1.26 for
OR, 1.67 for L1N, and 1.93 for RN; remeasure them on the chosen image/SKU.

The following family-scale envelope is analytical and includes the 20% reserve.
Its sustained-throughput bands are unmeasured planning assumptions, not vendor
specifications or empirical results. S4 uses Community RTX 4090 only for short
31M/70M calibration; S5 uses Secure Cloud throughout. Replace every throughput,
hour, and cost row after the per-size 128-step calibration.

| Size | Assumed sustained TFLOP/s | S4 GPU; 15-run hours | S4 budget | S5 GPU; 30-run hours | S5 budget |
| --- | ---: | --- | ---: | --- | ---: |
| 31M | 20--35 | Community RTX 4090; 2.8--4.9 h | $1.1--$2.0 | Secure RTX 4090; 53--93 h | $43.9--$77.0 |
| 70M | 30--50 | Community RTX 4090; 4.8--8.1 h | $2.0--$3.3 | Secure RTX 4090; 92--153 h | $76.2--$126.7 |
| 160M | 40--70 | Secure L40S; 9.7--16.9 h | $11.5--$20.1 | Secure L40S; 183--320 h | $217.4--$380.2 |
| 410M | 60--100 | Secure A100 SXM; 19--32 h | $34.0--$57.2 | Secure A100 SXM; 364--606 h | $650.8--$1,083.5 |
| **Total** | unmeasured | mixed; **36--62 h** | **$48.6--$82.6** | Secure mixed; **692--1,172 h** | **$988.3--$1,667.4** |

`TODO:` replace the assumed sustained-throughput bands with measured values and
store the calibration run ids before approving each size.

The provisional S4+S5 family budget is therefore **$1,037--$1,750**, plus storage
and external archive costs. These are training-panel estimates; cloud
qualification and unusually expensive full-cache diagnostics are budgeted
separately. The 410M S5 panel dominates the uncertainty and spend. It must not
launch until its fit, throughput, checkpoint/recovery, and per-run cost are
measured. Optional A3 pressure extensions are also budgeted separately rather
than hidden inside this minimum.

## 10. Launch, Finalization, and Cost Control

Current repository commands remain the scientific entry points:

```text
make install
make test
make check
python -m paper_exp.cli pretrain --config configs/<config-id>.yaml
python -m paper_exp.cli activation-propagation --config configs/<diagnostic-id>.yaml
python -m paper_exp.cli check --strict
```

There is no cloud launcher, resume command, or verified upload command yet;
those are blockers, not implied capabilities.

For each approved cloud batch:

1. refresh price and availability; record date, region/datacenter, tier, SKU,
   interruptibility, image digest, and budget ceiling;
2. deploy from a clean detached Git commit and verify environment/cache hashes;
3. run preflight and the size-specific smoke/calibration before the batch;
4. launch one matched panel at a time and monitor process, GPU, disk, events,
   loss, and expected completion;
5. inventory and verify artifacts before marking the run canonical;
6. publish the external completion marker last, retrieve scalar evidence, and
   run strict integrity checks;
7. stop or terminate idle compute immediately; retain only the active working
   set on RunPod storage;
8. reconcile estimated versus billed cost before approving the next batch.

Never store API keys, access tokens, private SSH keys, public IPs, or object-store
secrets in configs, manifests, registries, logs, or Git.

## 11. Cloud Provenance

Each cloud run registry row records execution backend/provider, cloud tier,
provisioning/interruptibility, Pod, region, and datacenter ids, GPU type/count,
image tag/digest, environment-lock hash, Python/PyTorch/CUDA/cuDNN/driver
versions, token-cache hash, live hourly rate plus source/date, billed GPU
seconds, compute/storage cost, resume linkage, resume checkpoint hash, and
artifact-upload verification time.

These are execution fields. They do not belong in the scientific config and do
not authorize changing its model, seed, token schedule, precision, or method.
