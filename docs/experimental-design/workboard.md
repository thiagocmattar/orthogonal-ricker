# Experimental Workboard

This is the single tracker for design/implementation TODOs. It is not an
attempt log. Change a state only with a linked commit, test, decision, or
review artifact.

Work-item states are `open`, `in_progress`, `resolved`, and `not_applicable`.
Plan review status and attempt status have their own authorities; they are not
copied into this table.

## Blocking Work

| ID | State | Blocks | Requirement | Acceptance evidence |
| --- | --- | --- | --- | --- |
| `ID-01` | resolved | A1+ | Verify proposed 14M dataset/model/tokenizer pins, split/text column, and licenses | Realized immutable revisions, field identities, and license review recorded in `protocol.md`; training/selection/confirmation cache hashes recomputed on 2026-08-25 |
| `ID-02` | open | C1+ | Pin 70M/410M architecture/tokenizer revisions, licenses, and cache compatibility | Reviewed protocol values and cache evidence |
| `TRAIN-01` | resolved | A1+ | Implement seeded complete-block permutation plus 74-block wrap and schedule hash | Commit `e7d3b68`; exact permutation/wrap/cardinality tests and frozen A1 schedule SHA-256 |
| `TRAIN-02` | resolved | A1+ | Implement 1% warmup and cosine decay to 0.1 of peak | Commit `e7d3b68`; endpoint/off-by-one tests for both budgets |
| `TRAIN-03` | resolved | A1+ | Implement global gradient clipping at 1.0 with specified L1/OL1 ordering | Commit `1d71698`; 59 focused tests covering none/L1/OL1 ordering, nonfinite rejection, telemetry, and manifests |
| `METHOD-01` | open | B1+ | Support one-sided FFN and independently one-sided/symmetric attention thresholds under one kappa | Reviewed methods amendment; schema/model/checkpoint tests; numerical and diagnostic-sufficiency proof for the kappa-zero functional alias |
| `METHOD-02` | open | A3+ | Freeze OL1 `step_budget` without using smoke calibration as scientific selection | Recorded `ol1_step_budget` decision and rationale |
| `DIAG-01` | open | B1+ | Define and implement `R_model^max` numerator/denominator/coverage contract | Reviewed `diagnostics.md` amendment and exact-count tests |
| `DIAG-02` | not_applicable | A2 pretraining | Add pooled-RMS schema fields or new training-time spillover machinery | Config `018-a2-activation-histograms`, run `001-20260828-082044-a031175f`, used the existing schema-v3 implementation over the complete selection partition; artifact SHA-256 `5f33739088a3cdc69e7cec28a7fe0061497c96d12f5b73996422324d76be850b`. Existing integer counts and RMS fields supply every reviewed A2 reduction. |
| `DIAG-03` | open | A3+ | Aggregate OL1 conflict/projection/eligibility counters over every optimizer boundary | Coverage and serialization tests |
| `VAL-01` | resolved | A1+ | Implement exact selection cadence and a caller-bound saved-checkpoint confirmation evaluator | Commit `e7d3b68`; exact 9/31-point cadence plus source/checkpoint/cache identity, complete-block coverage, loss/perplexity, completeness, and provenance tests |
| `VAL-02` | open | headline confirmation/paper release | Publish a durable confirmation-validation workflow around the reusable evaluator | Exact source-run resolution, cache-byte verification, lifecycle publication, and CLI tests; not required by the A1 screen |
| `OPS-01` | resolved | A1 config allocation | Validate catalog counts, reviewed design SHA/blobs, exact A1 group membership, config/manifest identity, and duplicate fingerprints | Commit `99c2d03`; all 18 catalog expressions checked; A1-only materialization, strict check, launch preflight, canonical fingerprints, and manifest SHA tests |
| `OPS-08` | resolved | A2 config allocation | Admit exactly the confirmed six-cell A2 cohort while preserving completed A1 history outside active scope without extending launch authority | Commit `6ca61ff4a2542093b59de5080af51bb711a0c00a`; 61 focused passes, 541 full-suite passes with 3 skips, strict check at 0 errors/warnings, smoke `025-20260827-110207-15e78835`, and exact-history acceptance for all 11 A1 configs |
| `OPS-16` | resolved | A2 definitive launch | Bind exact configs `012`-`017` to two distinct homogeneous A40 workers under one coordinator and lock, using production-shaped A2 timing rather than the provisional ETC | Accepted concurrent calibration and definitive execution; all six runs completed and were imported from archive SHA-256 `147213d9616502a189f9b6a06cee2e7f36c1384d2b05fea89941aec659359914`; unchanged-archive deep verification and import acceptance SHA-256 `90a973806d637f938643f4060900cc57f0971af6df19d6fe31fe76e00af1d83a`; teardown verified zero Pods, zero network volumes, and `$0/hour`. |
| `OPS-15` | open | A3+ config allocation | Add exact physical-cell materialization contracts for later non-A2 groups | Per-group field/grid/decision/reuse tests; current implementation deliberately fails closed for unknown later groups |
| `DESIGN-02` | resolved | A1 boundary extension | Review the exact four-cell A1 design at a committed SHA while preserving configs/runs `001`–`003` and allocating only the proposed `4e-3` cell afterward | User approval on 2026-08-25 of design commit `e8214a411afebf0cec5658f0f1ccdd3e6bcd5585`; reviewed manifest for `[A1-lr-screen]` |
| `DESIGN-03` | resolved | A1 final boundary extension | Review the exact five-cell A1 design while preserving configs/runs `001`–`004` and allocating only `8e-3` afterward | User direction on 2026-08-26; reviewed design commit `3710dfdd416ac3484516c4d8c2162692346fe7e9`; manifest scope `[A1-lr-screen]` |
| `DESIGN-04` | resolved | A1 three-cell high-LR extension | Review the exact eight-cell A1 design while preserving configs/runs `001`–`005` and allocating only `1.6e-2`, `3.2e-2`, and `6.4e-2` afterward | Direct user instruction on 2026-08-26; reviewed design commit `d80f6a9b6c99bcaec7ddc52e73c1a407a5020a8e`; manifest scope `[A1-lr-screen]` |
| `DESIGN-05` | resolved | A1 very-high-LR extension | Review the exact eleven-cell A1 design while preserving configs/runs `001`–`008` and allocating only `1.28e-1`, `2.56e-1`, and `5.12e-1` afterward | Direct user instruction on 2026-08-26; reviewed design commit `2320d542b14926315a17e873afac2d41a40d6814`; manifest scope `[A1-lr-screen]` |
| `DESIGN-06` | resolved | A2 screen | Refine A2 to one seed-0 ReLU control plus h-only L1 at lambda `{0.1, 0.5, 1, 2, 5}`; do not add seeds for the full response | User approval on 2026-08-27 of exact design commit `8d0a750f8f687041370037fa25553c13c9e4c081`; reviewed manifest scope `[A2-relu-control, A2-l1-screen]` |
| `DESIGN-01` | open | B1/B2 | Resolve the Phase B baseline contradiction: B1/B2 use `full-pass-wrap`, but their catalogued A0 control reuses the selected 400M-token A1 run even though the fingerprint contract forbids reuse across budgets | Reviewed catalog/phase amendment that either allocates a matching full-pass A0 control or changes the relevant Phase B budget, with corrected run counts and reuse tests |
| `OPS-02` | resolved | recovery | Scope resume to pretraining, skip completed configs, require explicit failed-retry authorization, and stop on unsafe state | Commits `8de8324`, `5263dba`, `be0c9a4`; 301 tests, strict check, and infrastructure smoke passed |
| `OPS-03` | resolved | calibration/launch | Add a 600-second calibration-only limit that cannot truncate definitive pretraining; record setup, train, validation, diagnostic, and checkpoint timing separately | Commit `242c439`, local lifecycle tests, and the accepted [A1 same-hardware calibration](a1-calibration-packet.md); acceptance digest `d6b12d230e1b82a5b57f75857283b1b84690cadfe8264742e4e2b7456a051216` |
| `OPS-04` | resolved | A1+ | Freeze the Pythia-14M physical microbatch and gradient-accumulation decomposition on A40 48GB while preserving 128 sequences per optimizer update | Commits `0852a2e`, `037a705`, `131a142`; live idle-GPU profile selected microbatch 16 / accumulation 8; value explicitly approved on 2026-08-25 |
| `OPS-07` | in_progress | C1+ | Review/freeze the proposed Pythia-70M and Pythia-410M physical-batch decompositions on their pinned RunPod GPU class | Live idle-A40 profiles exist and proposals are recorded in `protocol.md`; scale-up value review remains outside the A1 packet |
| `CLOUD-01` | resolved | cloud smoke/launch | Set up RunPod for agent operation from the official `agent-setup.md`: install the skill bundle and Codex marketplace plugin, complete user OAuth, and define secure storage, environment, provenance, and teardown practice | Commits `8fc134b`, `82f67d3`; authenticated MCP and pinned `runpodctl` v2.8.0; [operator procedure](../runbook.md#runpod-operations); Pod `zv71bv4m85nvhu` in `CA-MTL-1` passed live SSH/GPU proof on 2026-08-25, artifacts were retrieved, and MCP plus CLI inventory checks after deletion returned no Pods or volumes; artifact-pack SHA-256 `65ac472668b347ff74aab7886160cbc4674a8bd533cc037d92e171635e81623c` |
| `OPS-05` | resolved | concurrent calibration | Implement bounded execution under one authoritative one-host coordinator and lock; definitive pretraining was serial at this calibration milestone | Prior infrastructure commits through `b62f03b` plus commit `99c2d03`; live two-GPU isolation proof; calibration accepts distinct configs on distinct homogeneous GPUs. The later exact-A1 exception is owned by `OPS-09`; same-GPU packing and multi-Pod mode still fail closed |
| `OPS-06` | resolved | concurrency infrastructure | Validate the bounded RunPod worker workflow with infrastructure-only smoke tests before any scientific use | Commit `b62f03b`; live report SHA-256 `2c356f51bd0068c7fd2d39ac5bbd70c3c4b3a43e53d101e7a3595f08c16bde17` proves overlap, injected failure/drain, same-coordinator-invocation recovery, completed reuse, BF16, and distinct GPUs; artifact retrieved and both RunPod inventories verified empty |
| `OPS-09` | resolved | definitive A1 launch | Expose bounded config-level concurrency only to the exact `A1-lr-screen` case runner: one coordinator and lock, one Pod, exactly two distinct homogeneous A40 slots, stable admission, failure stop-and-drain, and unchanged per-config reuse/retry semantics | Commit `a23c56d`; authorization is bound to the ordered A1 config IDs, two workers, and `NVIDIA A40`; 74 focused runner/scheduler tests and the full 480-pass suite passed, with strict check at 0 errors/warnings and infrastructure smoke completed |
| `OPS-10` | resolved | A1 boundary-cell launch | Preserve cross-revision A1 training semantics and make reuse of configs `001`–`003` fail closed before config `004` can run | Materialization commit `e7e63a4`; exact evidence below; 81 focused tests and full suite of 491 passed / 3 skipped; config `004` subsequently completed with the frozen initialization, schedule, cache, validation, budget, batch, precision, and checkpoint identities |
| `OPS-11` | resolved | A1 `8e-3` boundary-cell launch | Preserve the active A1 path and make reuse of configs `001`–`004` fail closed before config `005` can run | Launch commit `a4ddaa5c9897224a9285afae09d2d9c6b07b3cec`; config `005` run `001-20260826-135546-928279bb` completed all 1,526 steps and passed remote/local checkpoint acceptance; retrieved archive SHA-256 `04e1ef3684152e86836a535f9f6b82c95193646f313bb7beae5aa89030c399f0`; teardown verified zero Pods, zero network volumes, and $0/hour |
| `OPS-12` | resolved | A1 three-cell high-LR preparation | Bind a definitive exception to exact configs `001`–`008`, require completed reuse of `001`–`005`, and admit only pending `006`–`008` on three distinct homogeneous A40 GPUs under one coordinator and lock | Materialization commit `b586500d28bd9ee6e15319ceb4180008a0b63082`; 104 focused tests and full suite 494 passed / 3 skipped; strict integrity 0 errors/warnings; infrastructure smoke `021-20260826-162207-c1b4c4b9`. Pod `9l8jns1uwarkfp` passed base preflight but was stopped before transfer or training; resume did not reacquire three A40s, it was deleted, permitted replacement calls allocated no Pod, and inventories were reconciled. This path was superseded before scientific execution by `OPS-13` |
| `OPS-13` | resolved | A1 three-cell high-LR serial launch | Execute exact configs `006`–`008` serially on one Secure A40 under the existing coordinator, lock, scientific contract, and required reuse of `001`–`005` | User-approved one-A40 amendment; clean commit `d4105722516958df6e9c3cc43b20d6bfd4619d0f`; Pod `bq45s1hj2262ak` in `CA-MTL-1`; runs `001-20260826-174611-04b42898`, `001-20260826-182559-bb05a50c`, and `001-20260826-190546-4df1c441` each completed all 1,526 steps and are eligible, valid evidence; all remote/local per-file hashes matched; coordinator-log SHA-256 `01f23203e8995b8b60818dc1bb3353b4319211018b2586d551fc6b43ab08beea`; verified retrieval preceded deletion; zero Pods and zero network volumes confirmed |
| `OPS-14` | resolved | A1 very-high-LR serial launch | Execute exact configs `009`–`011` serially on one Secure A40 under one coordinator and lock while requiring immutable completed reuse of `001`–`008` | Packet approved at commit `2f8f55cd57b25828fea10f015992b0824b28f3a6`; execution commit `4e5e93e64d979004f2fd2e2a5b7aab275b088e0d`; Pod `bgh0jufzmn180f` in `CA-MTL-1`; runs `001-20260826-221407-812e78f4`, `001-20260826-225355-07a74682`, and `001-20260826-233349-87400e7d` completed all 1,526 steps and are eligible, valid evidence; accepted archive SHA-256 `69e3166607635d891607888b92dab6cdad7e27ed451ac65ef735b78894322a98`; acceptance SHA-256 `41917bad40e0b6bf8a085378967f57b81be03937088ad6aeb764815a6ba33035`; verified retrieval preceded deletion; zero Pods, network volumes, endpoints, and `$0/hour` confirmed |
| `PLOT-01` | open | paper release | Implement the three declared figure families from pinned artifacts | Deterministic plot tests, PDF/PNG, provenance sidecars |
| `MAN-01` | open | manuscript release | Remove or separately validate out-of-scope 12B/long-context claims and enforce claim wording | Reviewed introduction consistent with `outputs.md` |

## Stage Readiness

Stage states are `blocked`, `ready`, `running`, `complete`, and `stopped`.
Readiness is derived from this table, the reviewed case-group scope, and
`decisions.md`; it is not copied into `cases.yaml`.

| Stage | State | Required closure/decision |
| --- | --- | --- |
| A1 | complete | All eleven reviewed cells completed and are eligible, valid evidence; the predeclared lowest-final-selection-loss rule selects config `008-a1-lr-6p4e-2` for dependent experiments. |
| A2 | complete | Configs `012`–`017` and diagnostic `018` completed, passed independent acceptance, and provide the reviewed seed-0 loss and spillover evidence. |
| A3 | blocked | Review group `A3-ol1-screen`; A2 complete; `METHOD-02`, `DIAG-03`, `OPS-15` |
| B1 | blocked | Review `B1-threshold-screen`; A2 terminal interpretation; resolve `DESIGN-01`, `METHOD-01`, `DIAG-01`, `OPS-15` |
| B2 | blocked | Review the B2 groups; resolve `DESIGN-01`; decisions `lambda_B2`, `b1_family`; valid prerequisites; `OPS-15` |
| C1 | blocked | Review group `C1-lr-screens`; `ID-02`, `OPS-07`, `OPS-15`, plus shared training/validation blockers |
| C2 | blocked | Review C2 groups; C1 per-model LRs and A2 design; `OPS-15` |
| C3 | blocked | Review C3 groups; C2 plus decisions `b2_frontier` and `b2_winner`; `OPS-15` |

`blocked` is descriptive here; it does not replace the goal-status semantics of
an interactive agent. Agents should take the highest upstream unresolved item
whose scope is authorized, verify it, and commit that coherent step before
moving to the next item.

## Recurring Prelaunch Checks

After a case group is reviewed and its configs are committed, every tranche
still needs the runbook preflight, a production-shaped same-hardware timing
sample, first-run/full-tranche ETCs, and explicit launch approval. These are
launch checks, not plan-review blockers. A1 has completed its timing sample;
later tranches must supply their own.

`CLOUD-01`, `OPS-04`, `OPS-05`, `OPS-06`, and `OPS-09` preserve the operational
record for the completed original A1 launch. `OPS-12` preserves the
superseded three-A40 preparation, `OPS-13` records the completed one-A40
serial high-LR launch, and `OPS-14` records the completed one-A40 serial
extension at configs `009`–`011`. Future definitive tranches remain serial by
default unless a newly reviewed bounded exception says otherwise. Multiple
runners, same-GPU packing, heterogeneous slots, and multi-Pod execution remain
forbidden.

## A1 Boundary-Extension Compatibility Evidence

Baseline execution commit
`276da7cd8e9142da48b95e12b46a99d61367ca8f` and materialization commit
`e7e63a4ae7f7def56d344b69adc426636dd7e0fb` retain identical Git blobs for
`training.py`, `modeling.py`, `optimization.py`, `data.py`,
`reproducibility.py`, `config.py`, `design.py`, `activation_pressure.py`,
`activations.py`, `topology.py`, `run.py`, `utils.py`, `launch.py`,
`parallel.py`, `pyproject.toml`, and `constraints/requirements-ci.txt`.

The only changed library files are:

- `integrity.py`: recognizes exact indexed completed historical evidence while
  a plan is placeholder or while its group is outside active reviewed scope;
  this preservation grants no launch authority, and `classify_run_directory`
  is unchanged.
- `runner.py`: adds `required_completed_config_ids` checks before and under the
  launch lock. `_run_one` and the training call are unchanged. This affects
  admission/reuse only and refuses to rerun a required historical config.

Normalized-LF source-block SHA-256 values match across both commits:

| Function | SHA-256 |
| --- | --- |
| `classify_run_directory` | `9597d2ee32c4a4bb2101a25e25d51db82f4c70ee859d9cb50630df359560fb16` |
| `_run_one` | `f929344adafd0000b978add058532d7a09e59eb8974b5eb1c166f6e86ff6b2c8` |
| `run_training` | `9d4efbcc602426119514420f11b6a1513b8409129accc57c0008f16f1c5f9137` |

Configs `001`–`003` retain Git blobs
`5da71c95ce2c291708286432abb4b889da00508a`,
`be850d7405d3b81366762740fa367263b25657d6`, and
`79870c132d3d4a4a9301f13ef8fcbfce3c43ad6`.
Config `004` differs from `003` only in experiment label, derived fingerprint,
and peak LR; its fingerprint is
`7742e7219fb40ee55adc4a42d87c00de6790eb7a5b3f5ff9643f85a137b9dd01`
and normalized complete-config SHA-256 is
`701afdbdcade83bdd878a30b65683825fb27c15038e24e4f6426f30658f1680d`.
The local runner-state classifier resolves exactly one accepted completion for
each of `001`–`003` and no attempt for `004`.

Verification at materialization: 81 focused runner/design tests passed; the
full suite passed 491 with 3 expected platform skips; strict integrity reported
0 errors and 0 warnings; infrastructure smoke attempt
`018-20260825-233716-c9e29f60` passed. Config `004` subsequently reproduced the
frozen runtime contract in run `001-20260826-123606-46e7454f`. Config `005`
changed only its label, derived fingerprint, and peak LR, then reproduced the
same initialization, schedule, cache, validation, budget, physical batch,
precision, and checkpoint identities in run
`001-20260826-135546-928279bb`.
