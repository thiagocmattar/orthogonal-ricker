# Pythia Sparsity Scaling Campaign

This folder is the authoritative handoff point for the next experiment round.
It turns the current sequence of local ablations into a staged campaign that can
support a paper-quality experimental section without erasing negative or failed
work.

## Current State

- Campaign id: `pythia-sparsity-scaling-v1`.
- Planning date: 2026-07-18.
- Current phase: E0.1 and all currently executable S1-B0 cells are complete;
  the fixed-gate engineering gate passed, and the first eighteen S1-B1 cells
  plus their mandatory complete-selection diagnostics are complete. The
  nineteenth B1 scientific run passed terminal review; its mandatory diagnostic
  is next to materialize. The 2,048-step screen remains restricted to
  feasibility/collapse and within-stratum comparisons.
- Next config prefix: `186`.
- The five central B0 anchors, configs `123--127`, completed and passed
  terminal review. Their final selection losses are `7.04913` (A0),
  `6.98875` (A1-H), `7.01310` (A3), `7.01645` (A6-PRE), and `7.03248`
  (A6-POST). Config `128`, the combined selection-partition endpoint
  diagnostic, also completed and passed artifact review. Configs `121--122`
  remain accepted engineering controls.
- Last completed scientific run: config `185`, fixed `Gpm` POST-RoPE Q/K at
  `kappa=0.03`, run `001-20260719-101901-c6136a3e`, selection loss `7.01265`.
- Config `134`, run `001-20260718-205354-8821b592`, completed the pooled
  exact-zero diagnostic over canonical configs `129--133`.
- All ten LR-flank training runs, configs `135--144`, are complete and passed
  terminal review. Config `145`, run `001-20260719-003521-94d90e97`, completed
  the pooled selection-partition endpoint diagnostic over all ten checkpoints
  and evaluated 311,296 tokens per method.
- Config `133` attempt 2 was an accidental duplicate created during a handoff
  race. It was terminated after step 1 and is invalid; attempt 1 completed.
- Config `131` attempt 2 was an accidental duplicate created during a handoff
  race. It was terminated after step 1 and is invalid; attempt 1 completed.
- Config `129` attempt 1, run `001-20260718-190208-c6152824`, is an invalid
  infrastructure-only partial: sandbox process containment ended it before
  the first training event. Retry 2 completed from the same immutable config.
- Configs `129--133` materialize the next ordinary architecture-parent set:
  A4-Q, A4-K, A4-V, A5-QK-PRE, and A5-QK-POST. They passed preflight and are
  complete and passed terminal review. Their matched results are consolidated
  in [`08-s1-b0-attention-parent-results.md`](08-s1-b0-attention-parent-results.md).
- Configs `135--144` materialize all 10 executable B0 learning-rate flanks:
  A0, A1-H, A3, A6-PRE, and A6-POST at LR `1e-5` and `1e-4`.
- Config `146` is the 128-step all-site A6-POST fixed-`G+`, `kappa=0.10`
  engineering pilot. Config `147` is the matched 128-step PRE-RoPE Q/K-only
  fixed-`Gpm`, `kappa=0.10` subset/round-trip pilot. They are engineering
  controls, not members of the 36-cell B1 scientific matrix. Config `146`, run
  `001-20260719-010308-764e5074`, and config `147`, run
  `001-20260719-010858-a4a15700`, completed cleanly and passed individual
  terminal review. Config `148`, run `001-20260719-011324-dd965e29`, completed
  the pooled selection diagnostic and passed the frozen engineering gate. B1
  config `149` materializes the first scientific cell: fixed `G+`, PRE-RoPE
  Q/K, `kappa=0.03`, AdamW without pressure. Run
  `001-20260719-012122-d3b3c4f5` completed at selection loss `7.01612` and
  passed terminal review. Config `150`, run `001-20260719-014532-32cc5f54`,
  completed its mandatory selection diagnostic at `R_block=0.29067` and
  `R_model=0.08706`. Config `151` materializes the next matched cell at fixed
  `G+`, PRE-RoPE Q/K, `kappa=0.10`; run
  `001-20260719-014816-62d42778` completed at selection loss `7.01905` and
  passed terminal review. Config `152`, run
  `001-20260719-021127-8e8a8e3c`, completed its mandatory selection diagnostic
  at `R_block=0.32972` and `R_model=0.09876`. Config `153`, run
  `001-20260719-021803-600ff4f4`, completed at selection loss `7.03079` and
  passed terminal review. Config `154`, run
  `001-20260719-024218-dcb80f3f`, completed its mandatory selection diagnostic
  at `R_block=0.45622` and `R_model=0.13665`; the next B1 scientific cell may
  be materialized. Config `155` materializes fixed `G+`, PRE-RoPE Q/K plus
  post-split V, `kappa=0.03`; run `001-20260719-024629-701ee0c3` completed at
  selection loss `7.01712` and passed terminal review. Its mandatory selection
  propagation diagnostic, config `156`, run
  `001-20260719-031106-0433709a`, completed at `R_block=0.44210` and
  `R_model=0.13242`. Config `157` materializes the matched fixed `G+`
  PRE-RoPE QKV, `kappa=0.10` cell; run `001-20260719-031452-66d62b47`
  completed at selection loss `7.02195` and passed terminal review. Its
  mandatory selection propagation diagnostic, config `158`, run
  `001-20260719-033946-134ea0ae`, completed at `R_block=0.52409` and
  `R_model=0.15698`. Config `159` materializes the matched fixed `G+`
  PRE-RoPE QKV, `kappa=0.30` cell; run `001-20260719-034439-ee5194a7`
  completed at selection loss `7.03369` and passed terminal review. Its
  mandatory selection propagation diagnostic, config `160`, run
  `001-20260719-040928-dd0f5070`, completed at `R_block=0.72831` and
  `R_model=0.21815`. Config `161` materializes the matched fixed `G+`
  POST-RoPE QK-only, `kappa=0.03` cell; run
  `001-20260719-041422-bacaff99` completed at selection loss `7.03029` and
  passed terminal review. Its mandatory selection propagation diagnostic,
  config `162`, run `001-20260719-043828-c242eeb4`, completed at
  `R_block=0.33881` and `R_model=0.10148`. Config `163` materializes the
  matched fixed `G+` POST-RoPE QK-only, `kappa=0.10` cell; run
  `001-20260719-044217-970e86ca` completed at selection loss `7.03090` and
  passed terminal review. Its mandatory selection propagation diagnostic,
  config `164`, run `001-20260719-050552-f4f3f4e9`, completed at
  `R_block=0.37056` and `R_model=0.11099`. Config `165` materializes the
  matched fixed `G+` POST-RoPE QK-only, `kappa=0.30` cell; run
  `001-20260719-051008-59f8fe4c` completed at selection loss `7.03559` and
  passed terminal review. Its mandatory selection propagation diagnostic,
  config `166`, run `001-20260719-053610-96aacd5d`, completed at
  `R_block=0.45968` and `R_model=0.13768`. Config `167` materializes the
  matched fixed `G+` POST-RoPE Q/K plus post-split V, `kappa=0.03` cell; run
  `001-20260719-054016-5e3f0e3d` completed at selection loss `7.03290` and
  passed terminal review. Its mandatory selection propagation diagnostic,
  config `168`, run `001-20260719-060544-6394fcab`, completed at
  `R_block=0.48716` and `R_model=0.14592`. Config `169` materializes the
  matched fixed `G+` POST-RoPE Q/K plus post-split V, `kappa=0.10` cell; run
  `001-20260719-060936-1cfd8103` completed at selection loss `7.03408` and
  passed terminal review. Its mandatory selection propagation diagnostic,
  config `170`, run `001-20260719-063500-541b449d`, completed at
  `R_block=0.56464` and `R_model=0.16912`. Config `171` materializes the
  matched fixed `G+` POST-RoPE Q/K plus post-split V, `kappa=0.30` cell; run
  `001-20260719-063931-ae5b970a` completed at selection loss `7.03405` and
  passed terminal review. Its mandatory selection diagnostic, config `172`,
  run `001-20260719-070611-60dbfac9`, completed at `R_block=0.73260` and
  `R_model=0.21943`. Config `173` materializes fixed `Gpm`, PRE-RoPE Q/K,
  `kappa=0.03`, with the Three-ReLU A3 identity parent as its matched control;
  run `001-20260719-071126-022f081f` completed at selection loss `7.01319`
  and passed terminal review. Its mandatory selection diagnostic, config
  `174`, run `001-20260719-073848-15430afb`, completed at
  `R_block=0.21885` and `R_model=0.06555`. Config `175` materializes fixed
  `Gpm`, PRE-RoPE Q/K, `kappa=0.10`, with the Three-ReLU A3 identity parent as
  its matched control; run `001-20260719-074413-02346f45` completed at
  selection loss `7.01434` and passed terminal review. Its mandatory selection
  diagnostic, config `176`, run `001-20260719-080808-9b05f3e2`, completed at
  `R_block=0.25387` and `R_model=0.07604`. Config `177` materializes fixed
  `Gpm`, PRE-RoPE Q/K, `kappa=0.30`, with the Three-ReLU A3 identity parent as
  its matched control; run `001-20260719-081330-30489164` completed at
  selection loss `7.02590` and passed terminal review. Its mandatory selection
  diagnostic, config `178`, run `001-20260719-083904-4dbd36ba`, completed at
  `R_block=0.40069` and `R_model=0.12002` and passed artifact, endpoint, and
  PRE-RoPE zero-survival review. Config `179` materializes fixed `Gpm`,
  PRE-RoPE Q/K plus post-split V, `kappa=0.03`, with the Three-ReLU A3 identity
  parent as its matched control; run `001-20260719-084406-a4fee547` completed at
  selection loss `7.01432` and passed terminal review. Its mandatory selection
  diagnostic, config `180`, run `001-20260719-091040-566be9ab`, completed at
  `R_block=0.26365` and `R_model=0.07897` and passed artifact, endpoint, and
  PRE-RoPE zero-survival review. Config `181` materializes the matched fixed
  `Gpm` PRE-RoPE Q/K plus post-split V cell at `kappa=0.10`; run
  `001-20260719-091515-36a71a24` completed at selection loss `7.01775` and
  passed terminal review. Its mandatory selection diagnostic, config `182`, run
  `001-20260719-094101-86e1b7ca`, completed at `R_block=0.38896` and
  `R_model=0.11650` and passed artifact, endpoint, and PRE-RoPE zero-survival
  review. Config `183` materializes the matched fixed `Gpm` PRE-RoPE Q/K plus
  post-split V cell at `kappa=0.30`; run
  `001-20260719-094646-118743a1` completed at selection loss `7.02352` and
  passed terminal review. Its mandatory selection diagnostic, config `184`, run
  `001-20260719-101310-a220dcd4`, completed at `R_block=0.63897` and
  `R_model=0.19139` and passed artifact, endpoint, and PRE-RoPE zero-survival
  review. Config `185` materializes fixed `Gpm`, POST-RoPE Q/K,
  `kappa=0.03`, with the Three-ReLU A3 identity parent as its matched control;
  run `001-20260719-101901-c6136a3e` completed at selection loss `7.01265` and
  passed terminal review. Its mandatory selection diagnostic is next to materialize.
- B0 has completed all `20 / 20` currently executable scientific cells
  (`20 / 22` declared); the two post-PV context cells remain blocked. The
  declared S1 core has completed `39 / 134` cells (`39 / 132` executable).
  Diagnostic configs do not count as scientific cells.
- S1 must never use a global rank cutoff. Complete matched method panels from
  viable families advance to the 8,192-step rung under the frozen policy in
  `06-s1-budget-backtest.md`.
- Configs `1` through `120` and their existing results remain historical
  evidence. They must not be renamed, rewritten, or moved.
- The primary discovery design contains 134 predeclared 2,048-step cells. Two
  post-PV context cells depend on context-gate implementation, so the formally
  executable core is 132--134 cells. Up to 50 conditional control cells may be
  activated, for a maximum design envelope of 182--184 cells.
- The execution order is core first, then only conditional controls whose
  predeclared trigger fires, then the scaling ladder. The 182- or 184-cell
  envelope is a ceiling, not a mandatory batch.
- Estimated serial runtime on the currently measured RTX 5070 Ti path is about
  46--56 GPU-hours for the core screen and 66--80 GPU-hours for the full
  envelope. Learned-gate estimates are provisional until the 128-step pilot.
- RunPod is the preferred scaling provider, but scientific cloud runs are
  blocked until cache portability, environment locking, exact resume, artifact
  verification, and local/cloud parity gates pass.

Planned rows are not evidence that a cell ran. Only terminal manifests and run
registry rows explicitly marked as valid are campaign evidence.

## Latest Completed Launch Set

The S1-B0 learning-rate controls completed locally across 2026-07-18 and
2026-07-19. Each architecture now has matched `1e-5`, `3e-5`, and `1e-4`
endpoints at 2,048 steps; config `145` measured exact-zero and logical-product
endpoints for both flank slices. The consolidated evidence and handoff are in
[`09-s1-b0-learning-rate-flank-results.md`](09-s1-b0-learning-rate-flank-results.md).

| Architecture | Loss at `1e-5` | Loss at `3e-5` | Loss at `1e-4` |
| --- | ---: | ---: | ---: |
| A0 | 8.35104 | 7.04913 | 5.93887 |
| A1-H | 8.38652 | 6.98875 | 5.87474 |
| A3 | 8.40015 | 7.01310 | 5.91768 |
| A6-PRE | 8.38739 | 7.01645 | 5.93539 |
| A6-POST | 8.38743 | 7.03248 | 6.06320 |

B0 has completed all 20 currently executable scientific cells. Two post-PV
context cells remain dependency-gated; they are not silently counted as
complete.

## Latest Completed Engineering Launch Set

`E0.1`, completed locally on 2026-07-18, validates the campaign's data,
schedule, seed, and provenance contract. Validation loss is plumbing-only and
must not be used to rank methods.

| Config | Canonical pretrain run | Init/data seeds | Steps | Validation loss | Wall (s) | Tokens/s | Peak GPU (MiB) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `121` | `004-20260718-145119-0106c58b` | `0 / 0` | 128 | 10.109134 | 66.80 | 125,955 | 5,996 |
| `122` | `001-20260718-145419-e88ac5f4` | `1 / 0` | 128 | 10.097261 | 67.71 | 124,290 | 5,996 |

- Both runs reached 8,388,608 tokens and share schedule hash
  `9d9f708a79511390da9559b88e06e797aa216149af709c841923c56f926e1120`
  and selection-partition hash
  `ffc857a6f0771929dd75c93bc17729de98a692f3a175ac5742cc9d101ff4ea47`.
- Their initialization hashes differ as required: config `121` has
  `df48838c379b4b28ddebdf3cb5ce003e46f5fcd6f50f40e1549446a2fbc998ed`;
  config `122` has
  `d118172cbe98a14ad42acd78942f1914083e9559dada9bd225585fb2f130aa07`.
- Prepare-data run `003-20260718-144833-a9e5ac30` materialized and verified
  document-disjoint 250/250 partitions. Selection has 311,739 tokens;
  confirmation has 381,929. The two preceding empty sandbox-lock attempts are
  retained as invalid partial attempts in `run-registry.yaml`.

## Documents

1. [`01-screening-matrix.md`](01-screening-matrix.md) defines the exact broad
   Pythia-14M screen, stable design ids, parameters, counts, dependencies, and
   estimated runtime.
2. [`02-scaling-ladder.md`](02-scaling-ladder.md) defines token scaling at 14M
   and model scaling through Pythia-410M, including the learning-rate reference
   rule.
3. [`03-evaluation-and-promotion.md`](03-evaluation-and-promotion.md) defines
   validation, exact-zero and compute metrics, matched contrasts, promotion,
   uncertainty, and the paper figure suite.
4. [`04-runbook.md`](04-runbook.md) defines config allocation, launch and
   terminal checks, retries, registries, handoff, and open-source preservation.
5. [`05-runpod-cloud.md`](05-runpod-cloud.md) defines the core-to-envelope
   decision gate, RunPod readiness and operating profile, dated prices,
   expected cost envelopes, and the transition to cloud scaling.
6. [`06-s1-budget-backtest.md`](06-s1-budget-backtest.md) records the historical
   short-run rank-survival veto and the S1 budget decision required before a
   scientific launch.
7. [`07-s1-b0-anchor-results.md`](07-s1-b0-anchor-results.md) consolidates the
   first scientific launch set, endpoint diagnostic, caveats, and handoff.
8. [`08-s1-b0-attention-parent-results.md`](08-s1-b0-attention-parent-results.md)
   consolidates the attention-parent controls, pooled endpoints, matched
   parent-child contrasts, caveats, and next launch set.
9. [`09-s1-b0-learning-rate-flank-results.md`](09-s1-b0-learning-rate-flank-results.md)
   consolidates the matched three-point learning-rate controls, pooled
   endpoints, caveats, and next engineering gate.
10. [`10-s1-b1-fixed-gate-engineering-results.md`](10-s1-b1-fixed-gate-engineering-results.md)
   records the fixed positive-threshold plumbing pilots, complete-selection
   endpoints, acceptance decision, and B1 handoff.
11. [`validation-partitions.yaml`](validation-partitions.yaml) freezes the
   document-disjoint selection and confirmation source-document lists.
12. [`config-registry.yaml`](config-registry.yaml) is the config-level source of
   truth for materialized campaign cells.
13. [`run-registry.yaml`](run-registry.yaml) records every run attempt. It is
   intentionally separate because one immutable config can have more than one
   infrastructure attempt.

## Non-Negotiable Controls

- All Pythia runs use the architecture named by the config with
  `model.initialization: random`. Released Pythia checkpoint weights are not
  loaded.
- The broad screen uses 2,048 optimizer steps, 65,536 tokens per step, and
  134,217,728 training tokens unless a row is explicitly labeled as a
  fixed-token batch-size control.
- Architecture, gate, pressure, learning-rate, seed, and token-budget effects
  are not silently mixed. Every comparison has a matched control id.
- RN and OR use identical Ricker `weight`, `c`, and `sigma`; L1N and OL1 use
  identical L1 weight. A method family is promoted as a complete
  `{AdamW, RN, OR, L1N, OL1}` panel, not as an isolated winner.
- Primary OR and OL1 cells always use `step_budget: 0.5`. Alternate budgets are
  diagnostic stability controls and are never eligible as tuned paper cells.
- Pressure scope is explicit. `QKV-only` and `all-active-gates` are different
  interventions.
- Screen selection uses a frozen selection-validation partition. Confirmation
  seeds and the disjoint campaign-confirmation partition are not inspected
  until the architecture and hyperparameters are frozen. This partition is new
  to the campaign, but not historically untouched: Reports 04--06 evaluated the
  complete cache that contains it.
- Exact zero means the integer comparison `x == 0`; no tolerance is used.
- `R_block` and `R_model` are logical scalar-product opportunities, not measured
  speedups. Runtime claims require sparse-kernel measurements.

## Resume Checklist

An agent resuming this campaign should:

1. Read this folder in numerical order, including the cloud plan, then read
   `configs/README.md` and `docs/methods.md`.
2. Inspect `git status --short`, the two registries, and terminal manifests. A
   manifest always outranks a registry status.
3. Confirm that no row with the same `design_id` is configured, running, or
   complete before allocating a prefix.
4. Complete the engineering blockers in Section 8 of
   `01-screening-matrix.md` before materializing affected cells.
5. Allocate the next unused sequential prefix, add a config-registry record,
   commit the config and registry from a clean tree, and only then launch.
6. Record every attempt in the run registry, including failures and retries.
7. Run the terminal envelope and diagnostic checks in `04-runbook.md` before
   marking evidence valid or considering promotion.

## Status Vocabulary

Config planning and execution are distinct from scientific disposition:

- Config status: `planned`, `blocked`, `ready`, `active`, `closed`, or
  `cancelled`.
- Run lifecycle: `running`, `completed`, `failed`, `statusless_complete`,
  `event_stream`, `partial`, or
  `inconsistent`.
- Evidence status: `unreviewed`, `valid`, `provisional`, or `invalid`.
- Decision status: `pending`, `control`, `screened_out`, `promote_tokens`,
  `promote_seed`, `promote_model`, `paper_candidate`, or `superseded`.

A scientifically poor result is not a failed run. A failed run can still have a
provisional endpoint, but its failed manifest is never rewritten as completed.

## Historical Boundary

The campaign registries begin with config `121`. Existing configs `1--120` stay
indexed by `docs/experiment_log.md` and `docs/paper_map.md`. Before a public
release, their local attempts should be backfilled into a frozen historical
inventory; that migration must not mutate their artifacts. Config `119` remains
`failed + provisional` because its training and validation outputs are durable
but the terminal atomic predictions write exceeded the Windows path envelope.
