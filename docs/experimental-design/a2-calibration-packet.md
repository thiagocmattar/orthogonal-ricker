# A2 RunPod Calibration and Launch Packet

> **Timing evidence only; no current launch authority.** The calibration
> validates the physical execution shape and ETC. Its losses and checkpoints
> are not A2 scientific evidence and must not seed definitive runs.

This packet covers reviewed groups `[A2-relu-control, A2-l1-screen]`, design
commit `8d0a750f8f687041370037fa25553c13c9e4c081`, and calibration execution
commit `80d41feb738873552a99da5be14dc21d161a9275`.

## Acceptance

| Check | Result |
| --- | --- |
| Runtime | Secure RunPod Pod `4ugiu0bayis25z`, EU-SE-1, 2 x NVIDIA A40 48 GB, compute capability 8.6, Python `3.12.3`, Torch `2.11.0+cu128`, CUDA `12.8`, Transformers `5.12.1` |
| Image | `runpod/pytorch@sha256:4d1721e62b56d345c83b4fd6090664be6daf9312caab5b2e76f23d8231941851` |
| Workload | Exactly configs `012-a2-relu-control` and `015-a2-l1-1`; 600 completed optimizer-step seconds each; one process per distinct A40 |
| Identity | Exact clean Git SHA, frozen cache hashes, schedule hashes, and shared initial-parameter SHA-256 |
| Lifecycle | Both attempts completed with metrics, events, predictions, and final checkpoints; strict repository check reports 0 errors and 0 warnings |
| Concurrency | Workers overlapped for 612.875 seconds on distinct GPU UUIDs |
| Retrieval | Archive size 104,748,909 bytes; SHA-256 `25b11ccf6080e22ccf258e4751807cd81905f5b8304f4993d15f73893b0d25d5` |
| Teardown | Zero Pods, zero network volumes, and `$0/hour` after verified retrieval |

The accepted package is retained in ignored local storage under
`tmp/a2-cal-retrieval-4ugiu0bayis25z/`. The billing ledger had not finalized
at teardown; the approximately 21.7-minute Pod lifecycle implies about `$0.32`
of GPU time, safely below the approved `$0.70` ceiling.

## Measurements

| Config | Steps | Tokens | Training time | Seconds/update | Tokens/s | Peak allocated / reserved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `012`, ReLU control | 390 | 102,236,160 | 601.045 s | 1.5411 | 170,097 | 23,101 / 29,342 MB |
| `015`, L1 lambda 1 | 213 | 55,836,672 | 600.251 s | 2.8181 | 93,022 | 23,568 / 29,356 MB |

L1 throughput is 54.69% of the control, or 1.8286 times slower. All five L1
cells use the same graph and differ only in the scalar lambda, so config `015`
is the timing proxy for the full L1 grid. Approximately 29.4 GB reserved per
worker validates one process per A40 and rules out same-GPU packing.

Calibration losses are deliberately excluded: the runs ended at different
optimizer steps and therefore cannot support a scientific comparison. The
primary spillover measurements remain the plan-defined post-hoc activation
histograms from definitive checkpoints.

## Calibrated definitive ETC and cost

Each definitive config has 5,691 optimizer updates and 1,491,861,504 input
tokens. Projections include the plan-defined 31 validations and finalization.

| Scope | Point ETC |
| --- | ---: |
| ReLU control | 2h 26m 47s |
| Each L1 cell | 4h 27m 55s |
| Six runs, one A40 serial reference | 24h 46m |
| Six runs, two bounded A40 workers | 13h 24m |
| Pod creation through verified retrieval | about 13h 40m |

Use 13-15 hours as the planning range and a 17-hour backend termination guard.
At no more than `$0.44` per A40-hour, expected total cost is about `$12.1`,
including running storage. The conservative 17-hour total-cost ceiling is
`$15.25`. These are operational projections, not confidence intervals.

## Definitive execution recommendation

Use one Secure Pod with exactly two homogeneous A40s, one coordinator, one
repository lock, and worker slots `gpu-0=0` and `gpu-1=1`. The tracked
authorization is bound to ordered configs `012`-`017`; no other config or
worker count is admitted. On an escaping failure, stop new admissions and
drain already-admitted workers.

Preserve the calibrated storage path: checkout, cache, and raw attempts run on
container-local storage under `/root/orthogonal-sparsity-pressure`; operations,
logs, and the retrieval package live under `/workspace`. Verify free space
before launch and retrieve every artifact before deleting the ephemeral Pod.

Definitive launch still requires explicit approval at the clean commit that
contains this packet and the exact two-worker case-runner authorization. That
approval must separately authorize RunPod resources, transfer, spending,
deadline, failure behavior, and teardown. Post-hoc diagnostics remain a later,
separately pinned operation.
