# A1 RunPod Calibration Packet

**Status:** accepted operational evidence. **Scientific evidence:** no.
Definitive A1 pretraining has not been launched and still requires explicit
approval.

This packet closes the production-shaped A1 timing check for the three
immutable configs approved at design commit
`54be534f383001b4af3d3b43597e135d4ca6653d` and materialized at execution
commit `7791242677e196d4721fb7a99cfd8f837ec877e2`.

## Acceptance

| Check | Result |
| --- | --- |
| Runtime | Secure RunPod Pod `rdd7d5acfl9au3`, CA-MTL-1, 2 x NVIDIA A40 48 GB, CUDA 12.8, Torch `2.11.0+cu128`, Transformers `5.12.1` |
| Image | `runpod/pytorch@sha256:4d1721e62b56d345c83b4fd6090664be6daf9312caab5b2e76f23d8231941851` |
| Physical batch | 16 sequences/GPU process x 8 accumulation steps = 128 sequences/update; BF16; 29,500 MB peak reserved VRAM |
| Identity | All attempts use the committed configs, exact execution Git SHA, frozen cache hashes, schedule hashes, and initial-parameter SHA-256 `778955b0319dc27e39201153e55c491350f90e0317e0a7b6ae6c7590fa7cfd17` |
| Lifecycle | Four completed calibration attempts; required metrics, events, predictions, and final checkpoints present; strict repository check reports 0 errors and 0 warnings |
| Concurrency | Two distinct GPU UUIDs, stable slots `gpu-0=0` and `gpu-1=1`, maximum concurrency 2, then the third config admitted after a slot became free |
| Teardown | 0 Pods and 0 network volumes; current spend rate $0/hour |
| Acceptance digest | SHA-256 `d6b12d230e1b82a5b57f75857283b1b84690cadfe8264742e4e2b7456a051216` |

The ignored evidence pack is stored at
`experiments/01-a1-lr-screen/raw/_calibration-20260825/`. Its archive SHA-256
is `0f8b79f81dcd3dc9393a1f415618ea919884a5303b2b4537e902e8162d4e8666`;
the control-plane record SHA-256 is
`8c0a4dca3e9ae4a6c3ee537a4e0484fe8480b3ebd2235a3cc8a9dc4363893400`.

## Measurements

| Mode | Config / peak LR | Steps | Tokens | Setup | Timed training | Total | Tokens/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Solo | `002` / `1e-3` | 390 | 102,236,160 | 11.48 s | 600.66 s | 619.44 s | 170,207 |
| Concurrent | `001` / `5e-4` | 390 | 102,236,160 | 86.13 s | 600.50 s | 692.46 s | 170,253 |
| Concurrent | `002` / `1e-3` | 390 | 102,236,160 | 86.23 s | 600.55 s | 692.55 s | 170,238 |
| Concurrent | `003` / `2e-3` | 389 | 101,974,016 | 20.73 s | 600.08 s | 626.17 s | 169,935 |

The matched `002` concurrent/solo throughput ratio is `1.00018`: the measured
difference is +0.018%, so this sample shows no concurrency slowdown. The
initial pair overlapped for 693.34 seconds by lifecycle and approximately
604.63 seconds during their optimizer samples. Concurrent setup took about 86
seconds because both workers initialized the model and cache together; this
did not affect steady-state training throughput.

These measurements validate the coordinator and hardware decomposition. They
do not authorize concurrent definitive training and cannot select a learning
rate.

## Calibrated ETC

The definitive/calibration workload ratio is `1,526 / 390 = 3.9128205`.
The solo projection is:

| Component | Definitive estimate |
| --- | ---: |
| Training | 2,350.26 s |
| Setup | 11.48 s |
| Nine validations | 7.87 s |
| Diagnostics | 9.96 s |
| Final checkpoint | 1.26 s |
| **One run** | **2,380.83 s = 39m 41s** |

| Scope from runner start | Point ETC | Planning range | Conservative cap |
| --- | ---: | ---: | ---: |
| First run | 39m 41s | 37m 43s-43m 36s | 45m 34s |
| Three-run serial tranche | 1h 59m 03s | 1h 53m-2h 11m | 2h 17m |

The planning range applies -5%/+10% to training time; the cap applies +15%.
Provisioning, transfer, dependency setup, retrieval, and teardown are outside
the runner ETC. The execution Pod took about 12 minutes from creation to the
solo wrapper start, so a three-hour backend termination window leaves useful
margin around the 2h 17m runner cap.

## Cost and Definitive-Launch Recommendation

The approved calibration consumed **$0.7623164963**, measured by account
balance delta across two short unavailable allocations and the execution Pod;
this is below the $3.57 ceiling. The detailed billing ledger had not posted at
teardown, so the balance delta is the cost authority for this packet.

For definitive A1, use **one Secure A40 and the serial tranche runner**. At the
2026-08-25 catalog price of $0.44/A40-hour, the three-run point compute cost is
$0.87, the planning range is $0.83-$0.96, and the conservative runner cap is
$1.00. A second GPU would remain idle under the reviewed serial workflow.

The proposed resource ceiling is one Pod for at most three hours: 1 x A40,
30 GB container disk, 50 GB volume disk at `/workspace`, no network volume,
and the pinned image above. Compute is capped at $1.32. At RunPod's
[published running storage rate](https://docs.runpod.io/pods/pricing) of
$0.10/GB/month, the combined 80 GB adds about $0.034 for three hours; round the
**total Pod ceiling to $1.36**. Recheck capacity, price, balance, clean Git SHA,
and an exact absolute termination timestamp immediately before requesting
spending approval and creating the Pod.

## Limitation and Required Operator Fix

RunPod did not inject `RUNPOD_POD_ID`, so the concurrent calibration manifests
record `worker_assignment.runpod_pod_id: null`. The immutable artifacts were
not rewritten; the independently saved control-plane record binds them to Pod
`rdd7d5acfl9au3`. Before definitive launch, export the exact control-plane Pod
ID into `RUNPOD_POD_ID` and verify that the running manifest records it. This
is an operational provenance fix, not a scientific failure of the calibration.

## Next Sequence

1. Commit this packet and the RunPod provenance procedure; do not change the
   A1 configs.
2. Obtain explicit definitive A1 approval at that exact Git SHA.
3. Recheck one-A40 capacity and price, calculate the absolute three-hour
   termination deadline, and obtain the separate spending approval.
4. Transfer a clean bundle and frozen cache, export the Pod ID, run
   `experiments/01-a1-lr-screen/run/runner.py` serially, and monitor read-only.
5. Retrieve and verify all three definitive attempts before deleting the Pod;
   then classify the cells and select the A1 learning rate under the reviewed
   rule.
