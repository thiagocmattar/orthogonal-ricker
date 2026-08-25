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
| `DIAG-02` | open | A2+ | Add pooled RMS and freeze histogram bins/range | Diagnostic schema/reduction tests and recorded decision |
| `DIAG-03` | open | A3+ | Aggregate OL1 conflict/projection/eligibility counters over every optimizer boundary | Coverage and serialization tests |
| `VAL-01` | resolved | A1+ | Implement exact selection cadence and a caller-bound saved-checkpoint confirmation evaluator | Commit `e7d3b68`; exact 9/31-point cadence plus source/checkpoint/cache identity, complete-block coverage, loss/perplexity, completeness, and provenance tests |
| `VAL-02` | open | headline confirmation/paper release | Publish a durable confirmation-validation workflow around the reusable evaluator | Exact source-run resolution, cache-byte verification, lifecycle publication, and CLI tests; not required by the A1 screen |
| `OPS-01` | resolved | A1 config allocation | Validate catalog counts, reviewed design SHA/blobs, exact A1 group membership, config/manifest identity, and duplicate fingerprints | Commit `99c2d03`; all 18 catalog expressions checked; A1-only materialization, strict check, launch preflight, canonical fingerprints, and manifest SHA tests |
| `OPS-08` | open | A2+ config allocation | Add exact physical-cell materialization contracts for non-A1 groups | Per-group field/grid/decision/reuse tests; current implementation fails closed for every non-A1 group |
| `OPS-02` | resolved | recovery | Scope resume to pretraining, skip completed configs, require explicit failed-retry authorization, and stop on unsafe state | Commits `8de8324`, `5263dba`, `be0c9a4`; 301 tests, strict check, and infrastructure smoke passed |
| `OPS-03` | in_progress | calibration/launch | Add a 600-second calibration-only limit that cannot truncate definitive pretraining; record setup, train, validation, diagnostic, and checkpoint timing separately | Commit `242c439` and local lifecycle tests; production-shaped same-hardware timing artifact still required |
| `OPS-04` | resolved | A1+ | Freeze the Pythia-14M physical microbatch and gradient-accumulation decomposition on A40 48GB while preserving 128 sequences per optimizer update | Commits `0852a2e`, `037a705`, `131a142`; live idle-GPU profile selected microbatch 16 / accumulation 8; value explicitly approved on 2026-08-25 |
| `OPS-07` | in_progress | C1+ | Review/freeze the proposed Pythia-70M and Pythia-410M physical-batch decompositions on their pinned RunPod GPU class | Live idle-A40 profiles exist and proposals are recorded in `protocol.md`; scale-up value review remains outside the A1 packet |
| `CLOUD-01` | resolved | cloud smoke/launch | Set up RunPod for agent operation from the official `agent-setup.md`: install the skill bundle and Codex marketplace plugin, complete user OAuth, and define secure storage, environment, provenance, and teardown practice | Commits `8fc134b`, `82f67d3`; authenticated MCP and pinned `runpodctl` v2.8.0; [operator procedure](../runbook.md#runpod-operations); Pod `zv71bv4m85nvhu` in `CA-MTL-1` passed live SSH/GPU proof on 2026-08-25, artifacts were retrieved, and MCP plus CLI inventory checks after deletion returned no Pods or volumes; artifact-pack SHA-256 `65ac472668b347ff74aab7886160cbc4674a8bd533cc037d92e171635e81623c` |
| `OPS-05` | resolved | concurrent calibration | Implement bounded execution under one authoritative one-host coordinator and lock; keep definitive pretraining serial | Prior infrastructure commits through `b62f03b` plus commit `99c2d03`; live two-GPU isolation proof; calibration accepts distinct configs on distinct homogeneous GPUs, while case-runner worker slots, same-GPU packing, and multi-Pod mode fail closed |
| `OPS-06` | resolved | concurrency infrastructure | Validate the bounded RunPod worker workflow with infrastructure-only smoke tests before any scientific use | Commit `b62f03b`; live report SHA-256 `2c356f51bd0068c7fd2d39ac5bbd70c3c4b3a43e53d101e7a3595f08c16bde17` proves overlap, injected failure/drain, same-coordinator-invocation recovery, completed reuse, BF16, and distinct GPUs; artifact retrieved and both RunPod inventories verified empty |
| `PLOT-01` | open | paper release | Implement the three declared figure families from pinned artifacts | Deterministic plot tests, PDF/PNG, provenance sidecars |
| `MAN-01` | open | manuscript release | Remove or separately validate out-of-scope 12B/long-context claims and enforce claim wording | Reviewed introduction consistent with `outputs.md` |

## Stage Readiness

Stage states are `blocked`, `ready`, `running`, `complete`, and `stopped`.
Readiness is derived from this table, the reviewed case-group scope, and
`decisions.md`; it is not copied into `cases.yaml`.

| Stage | State | Required closure/decision |
| --- | --- | --- |
| A1 | blocked | Formal review of group `A1-lr-screen`; every input, implementation, and physical-batch blocker is resolved |
| A2 | blocked | Reviewed A2 groups; A1 decision `lr_14m`; `DIAG-02`, `OPS-08` |
| A3 | blocked | Reviewed group `A3-ol1-screen`; A2 complete; `METHOD-02`, `DIAG-03`, `OPS-08` |
| B1 | blocked | Reviewed group `B1-threshold-screen`; A2 terminal interpretation; `METHOD-01`, `DIAG-01`, `OPS-08` |
| B2 | blocked | Reviewed B2 groups; decisions `lambda_B2`, `b1_family`; valid prerequisites; `OPS-08` |
| C1 | blocked | Reviewed group `C1-lr-screens`; `ID-02`, `OPS-07`, `OPS-08`, plus shared training/validation blockers |
| C2 | blocked | Reviewed C2 groups; C1 per-model LRs and A2 design |
| C3 | blocked | Reviewed C3 groups; C2 plus decisions `b2_frontier` and `b2_winner` |

`blocked` is descriptive here; it does not replace the goal-status semantics of
an interactive agent. Agents should take the highest upstream unresolved item
whose scope is authorized, verify it, and commit that coherent step before
moving to the next item.

## Recurring Prelaunch Checks

After a case group is reviewed and its configs are committed, every tranche
still needs the runbook preflight, the `OPS-03` calibration facility, a
production-shaped same-hardware timing sample, first-run/full-tranche ETCs,
and explicit launch approval. These are launch checks, not plan-review
blockers.

`CLOUD-01`, `OPS-04`, `OPS-05`, and `OPS-06` support the approved A1
calibration shape, but their resolution is not scientific-launch or spending
authorization. `AGENTS.md` requires serial definitive pretraining. The only
additional concurrency currently authorized is bounded calibration under one
coordinator and lock, with one process per distinct homogeneous physical GPU;
multiple runners, same-GPU packing, and multi-Pod execution remain forbidden.
