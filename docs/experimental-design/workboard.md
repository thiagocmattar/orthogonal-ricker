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
| `ID-01` | open | A1+ | Verify proposed 14M dataset/model/tokenizer pins, split/text column, licenses, and physical batch decomposition | Reviewed identities and memory-fit evidence in `protocol.md`; cache integrity check |
| `ID-02` | open | C1+ | Pin 70M/410M architecture/tokenizer revisions, licenses, cache compatibility, and physical batches | Reviewed protocol values and memory evidence |
| `TRAIN-01` | open | A1+ | Implement seeded complete-block permutation plus 74-block wrap and schedule hash | Focused reproducibility tests and serialized hash |
| `TRAIN-02` | open | A1+ | Implement 1% warmup and cosine decay to 0.1 of peak | Endpoint/off-by-one tests for both budgets |
| `TRAIN-03` | open | A1+ | Implement global gradient clipping at 1.0 with specified L1/OL1 ordering | Numerical tests and pre/post norm telemetry |
| `METHOD-01` | open | B1+ | Support one-sided FFN and independently one-sided/symmetric attention thresholds under one kappa | Reviewed methods amendment; schema/model/checkpoint tests; numerical and diagnostic-sufficiency proof for the kappa-zero functional alias |
| `METHOD-02` | open | A3+ | Freeze OL1 `step_budget` without using smoke calibration as scientific selection | Recorded `ol1_step_budget` decision and rationale |
| `DIAG-01` | open | B1+ | Define and implement `R_model^max` numerator/denominator/coverage contract | Reviewed `diagnostics.md` amendment and exact-count tests |
| `DIAG-02` | open | A2+ | Add pooled RMS and freeze histogram bins/range | Diagnostic schema/reduction tests and recorded decision |
| `DIAG-03` | open | A3+ | Aggregate OL1 conflict/projection/eligibility counters over every optimizer boundary | Coverage and serialization tests |
| `VAL-01` | open | A1+ | Implement exact 191-update cadence and generic saved-checkpoint confirmation validation | Cadence tests; source/partition/coverage/provenance tests |
| `OPS-01` | open | config allocation | Validate catalog expansion/count expressions; full reviewed-design SHA and unchanged normative blobs; reviewed group membership; functional aliases; config/manifest identity fields; and duplicate fingerprints | Catalog/fingerprint tests integrated with strict check and launch preflight |
| `OPS-02` | resolved | recovery | Scope resume to pretraining, skip completed configs, require explicit failed-retry authorization, and stop on unsafe state | Commits `8de8324`, `5263dba`, `be0c9a4`; 301 tests, strict check, and infrastructure smoke passed |
| `OPS-03` | open | calibration/launch | Add a 600-second calibration-only limit that cannot truncate definitive pretraining; record setup, train, validation, diagnostic, and checkpoint timing separately | CLI/config separation tests and a production-shaped same-hardware timing artifact |
| `PLOT-01` | open | paper release | Implement the three declared figure families from pinned artifacts | Deterministic plot tests, PDF/PNG, provenance sidecars |
| `MAN-01` | open | manuscript release | Remove or separately validate out-of-scope 12B/long-context claims and enforce claim wording | Reviewed introduction consistent with `outputs.md` |

## Stage Readiness

Stage states are `blocked`, `ready`, `running`, `complete`, and `stopped`.
Readiness is derived from this table, the reviewed case-group scope, and
`decisions.md`; it is not copied into `cases.yaml`.

| Stage | State | Required closure/decision |
| --- | --- | --- |
| A1 | blocked | Reviewed group `A1-lr-screen`; `ID-01`, `TRAIN-01..03`, `VAL-01`, `OPS-01` |
| A2 | blocked | Reviewed A2 groups; A1 decision `lr_14m`; `DIAG-02` |
| A3 | blocked | Reviewed group `A3-ol1-screen`; A2 complete; `METHOD-02`, `DIAG-03` |
| B1 | blocked | Reviewed group `B1-threshold-screen`; A2 terminal interpretation; `METHOD-01`, `DIAG-01` |
| B2 | blocked | Reviewed B2 groups; decisions `lambda_B2`, `b1_family`; valid prerequisites |
| C1 | blocked | Reviewed group `C1-lr-screens`; `ID-02` plus shared training/validation blockers |
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
