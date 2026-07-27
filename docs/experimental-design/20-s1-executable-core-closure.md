# S1 Executable-Core Closure and Handoff

## Terminal state

The applied registries close the Pythia-14M 2,048-step executable S1 core on
2026-07-27 at 132/132 scientific cells. A dated local closure audit reported
no census errors or gaps, all 36 mandatory scientific diagnostics
`closed_valid`, and repository-integrity return code 0. The reproducible
repository check is `python -m paper_exp.cli check`; at closure it returned
zero errors with 23 retained historical warnings. The registry closure commit
is
`5ddc0ca1e650f90c1035e14a3b7d6f69384d0205`.

| Block | Executable scientific cells | Required pooled diagnostics | Status |
| --- | ---: | --- | --- |
| S1-B0 | 20/20 | 128, 134, 145 | complete; two declared post-PV context cells remain dependency-gated |
| S1-B1 | 36/36 | 150--196 even, 203, 210 | complete |
| S1-B2 | 26/26 | 247 | complete |
| S1-B3 | 40/40 | 256, 265, 274, 283, 292 | complete |
| S1-B4 | 10/10 | 303 | complete |
| **Executable S1** | **132/132** | **36 diagnostics** | **complete** |

The primary design declared 134 cells. `A4-C` and `A7-POST-C`, the two post-PV
context-gate cells, remain dependency-gated and are outside the registered
132-cell executable denominator. They were not silently replaced.

## Evidence map

- B0 architecture and LR results: documents
  [`07`](07-s1-b0-anchor-results.md) through
  [`09`](09-s1-b0-learning-rate-flank-results.md).
- B1 fixed-threshold results:
  [`10`](10-s1-b1-fixed-gate-engineering-results.md) and
  [`11`](11-s1-b1-fixed-threshold-results.md).
- B2 learned-ATG results:
  [`12`](12-s1-b2-learned-atg-engineering-results.md) and
  [`13`](13-s1-b2-learned-atg-results.md).
- B3 pressure results: documents
  [`14`](14-s1-b3-t1-central-pressure-results.md) through
  [`18`](18-s1-b3-t5-ricker-shape-results.md).
- B4 paired-seed results:
  [`19-s1-b4-seed-sentinel-results.md`](19-s1-b4-seed-sentinel-results.md).
- Canonical config identities and block accounting:
  [`config-registry.yaml`](config-registry.yaml).
- Canonical attempts, runtimes, safety flags, and pooled endpoints:
  [`run-registry.yaml`](run-registry.yaml).

The last scientific training is config `302`, run
`001-20260727-134130-86a95dc4`. The last mandatory diagnostic is config `303`,
run `001-20260727-190455-ba5f3286`; its pooled artifact SHA-256 is
`2f2c24b31f953f58e1406aa2013992bff5526713f848ccc0c5da9e8c6393b4d5`.
No campaign process remains active.

## Interpretation boundary

S1 is a discovery screen. It supports feasibility, collapse and stability
checks, and predeclared within-stratum contrasts. It does not establish a
global winner, population variance, confirmatory effect, realized sparse
kernel speedup, or long-budget ranking.

Exact-zero endpoints use direct numeric equality `x == 0` and integer-pooled
counts over the named frozen validation partition. `R_block` and `R_model`
measure logical scalar-product opportunities; they are not measured wall-clock
savings. B4 changes model-initialization and data-order seeds together and has
only two observations per sentinel.

## Next registered decision

The next unused config prefix is `304`. No scientific or conditional run is
authorized by S1 closure alone.

Before materializing prefix `304`:

1. review every predeclared conditional trigger in
   [`01-screening-matrix.md`](01-screening-matrix.md);
2. register the activated subset and the reason for each activation or
   non-activation;
3. run only that subset;
4. freeze complete contrast-preserving panels for the 8,192-step S2 rung;
5. pass the RunPod qualification gates before cloud scientific execution.

`C4-BUDGET` is already triggered because the primary `step_budget=.5` cap
bound in multiple B3 OR/OL1 rows; B4 OR config `302` also bound on all logged
updates. It is a stability control, not a pressure-tuning axis. Other
conditional blocks require their own registered trigger review.

Preserve all negative cells, invalid attempts, canonical checkpoints, and
diagnostic artifacts. Do not rank heterogeneous architectures or pressure
settings with a new post-hoc scalar score.
