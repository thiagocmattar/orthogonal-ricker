# A1 Three-Cell Very-High-LR Definitive Launch Packet

> **Status:** prepared on 2026-08-26 and awaiting explicit definitive-launch
> and RunPod-spending approval. No Pod, transfer, or scientific attempt has
> been created under this packet.

## Scientific and Execution Identity

This launch executes only the reviewed factor-two A1 extension above
`6.4e-2`. The design commit is
`2320d542b14926315a17e873afac2d41a40d6814`, the activation commit is
`b7b75fdbe104e22b2554cdc40a387108ea1dede5`, the materialization commit is
`1fd0914068531d1c05e047f95352fabee3e3b04a`, and the clean execution commit is
`4e5e93e64d979004f2fd2e2a5b7aab275b088e0d`.

| Config | Peak LR | Condition fingerprint | Complete-config SHA-256 |
| --- | ---: | --- | --- |
| `009-a1-lr-1p28e-1` | `1.28e-1` | `0b5c10665bc83355fe365e91b1aeae156f970a96dc995a4cffbd80581426a001` | `8a7da2c097435968e03704fa09921a8345dea0c14a20a9e9ed717c980b05f259` |
| `010-a1-lr-2p56e-1` | `2.56e-1` | `9a541bb3b14196b96a9d130c7eb2d3c360beaa94ff4d828bb13aee93b10fec18` | `4edfcfbca86a72e0ef0809e743ae62157b4759518dd2b7705a8ba8d5f8805cfc` |
| `011-a1-lr-5p12e-1` | `5.12e-1` | `9782a13261d41f817ad99b72ac4efd2addc43bfe51de452ac9673cdfcf15faaa` | `7424beae00383dc0396712992cc38d372f2b7bf6de2226bf30b574a8fb7a7ae1` |

Each new config differs from config `008` only in experiment label, derived
fingerprint, and peak learning rate. Relative to the accepted serial execution
commit `d4105722516958df6e9c3cc43b20d6bfd4619d0f`, no active training, model,
optimizer, data, reproducibility, config, design, runner, launch, dependency,
or constraint source changed. The only added file under `src/` is the
post-run A1 plotting module. Local verification at the execution commit passed
503 tests with 3 expected platform skips, strict integrity with 0 errors and
0 warnings, and infrastructure smoke attempt
`023-20260826-213338-c8137578`.

## Required Completed Reuse

Before admitting config `009`, the runner must classify exactly one coherent
completed pretraining attempt for each immutable source below. None may be
rerun.

| Config | Required accepted run |
| --- | --- |
| `001-a1-lr-5e-4` | `001-20260825-191155-6b7376de` |
| `002-a1-lr-1e-3` | `001-20260825-191154-b9299c46` |
| `003-a1-lr-2e-3` | `001-20260825-195141-f842c400` |
| `004-a1-lr-4e-3` | `001-20260826-123606-46e7454f` |
| `005-a1-lr-8e-3` | `001-20260826-135546-928279bb` |
| `006-a1-lr-1p6e-2` | `001-20260826-174611-04b42898` |
| `007-a1-lr-3p2e-2` | `001-20260826-182559-bb05a50c` |
| `008-a1-lr-6p4e-2` | `001-20260826-190546-4df1c441` |

The local preflight classifies these eight runs as completed and configs
`009`–`011` as pending. The tracked runner lists all eleven configs and binds
the eight completed IDs as required reuse.

## Verified Transfer Set

| Payload | Bytes | SHA-256 |
| --- | ---: | --- |
| `tmp/osp-a1-very-high-lr-4e5e93e.bundle` | 9,277,631 | `d571190e06ad5f99b9729bc6570393f1f778975f54de540e7c1b47959f072fcc` |
| `tmp/a1-reuse-001-008-4e5e93e.tar.gz` | 418,155,314 | `a9301455fd24368dc4811df6e0af38ccc9f1e0eda41562fc3c98561494c4beea` |
| Frozen MiniPile cache | 5,972,403,826 | training tokens: `da82a2ea2e0080c7fd681c7a93b07d3d9ff3d5357a8640895a82d536a1eaf97c`; selection tokens: `22bb7c27864f0e5941548c572d6c75b1b5ba6a4c13e4cd26f40f4de546c5cc19` |

The bundle has complete history, no prerequisites, and exactly one advertised
head: the execution commit above. The reuse archive has an exact 88-member
allowlist, eight completed/pretrain manifests, and no links. Bundle history
and archive membership contain no `docs/humans/main.pdf`, credentials, key or
`.env` files, token cache, unrelated run, or unrelated untracked file. POSIX
path normalization is permitted only in the copied cache metadata.

## RunPod Envelope

A read-only inventory and capacity snapshot at `2026-08-26T21:34Z` found zero
Pods, zero network volumes, zero endpoints, and therefore `$0/hour` recurring
spend. Secure A40 CUDA 12.8 capacity was `LOW` at both permitted data centers;
allocation is not guaranteed.

| Field | Exact envelope |
| --- | --- |
| Pod | exactly one Secure Pod named `osp-a1-vhigh-4e5e93e` |
| GPU | exactly 1× NVIDIA A40 48GB; compute capability 8.6 |
| Region | `CA-MTL-1` preferred; `EU-SE-1` permitted |
| Price | at most `$0.44/GPU-hour`; stop if the live price is higher |
| Image | `runpod/pytorch@sha256:4d1721e62b56d345c83b4fd6090664be6daf9312caab5b2e76f23d8231941851` |
| Storage | 30GB container disk; 50GB Pod volume at `/workspace`; no network volume |
| Access | SSH with the previously validated RunPod public key; never transfer a private key or credential |
| Provisioning | one capacity-triggered creation attempt, no later than `2026-08-27T12:00:00Z` |
| Termination | backend deletion exactly 165 minutes after creation and never later than `2026-08-27T14:45:00Z` |
| Cost | expected `$1.10`–`$1.16`; maximum projected resource charge approximately `$1.23`; hard ceiling `$1.35` |

If the creation call allocates no Pod, stop without retry or replacement. A
successful allocation may use bounded SSH reconnection and resumable transfer
on that same Pod; it may not create another resource.

## Calibrated ETC

The matched configs `006`–`008` used the same A40, runtime, 1,526-update
schedule, microbatch 16, accumulation 8, validation/checkpoint policy, and
serial runner shape. Their durations were 2,388.336, 2,387.123, and 2,387.004
seconds: mean 39m47.5s and sample standard deviation 0.74s. Their complete
serial runner span was 1h59m22.8s. This direct evidence is stronger than a new
short calibration.

| Milestone | Point ETC | Planning range |
| --- | ---: | ---: |
| First new run after coordinator start | 39m48s | approximately 38–44m |
| All three new runs after coordinator start | 1h59m23s | 1h53m–2h11m |
| First result after Pod creation | approximately 60m | 60–64m |
| Scientific completion after Pod creation | approximately 2h18m | 2h20m–2h25m |
| Retrieval, verification, and teardown | approximately 2h32m | 2h30m–2h35m |

The range is an operational planning bound, not a confidence interval. It
assumes the exact hardware/runtime, local frozen-cache transfer, no competing
GPU process, and prior 18m31s creation-to-science setup performance. The
165-minute backend guard leaves roughly 10–15 minutes beyond expected verified
teardown.

## Launch, Failure, and Teardown Contract

1. Reconcile inventory, perform the single permitted creation call, record the
   selected data center and Pod ID, and arm backend termination.
2. Transfer and verify the small Git bundle first. Check out exactly the
   execution commit and install the pinned dependency snapshot.
3. Require one A40, Python 3.12.3, Torch `2.11.0+cu128`, CUDA 12.8,
   Transformers 5.12.1, BF16 support, a compatible host driver, and a bounded
   CUDA forward/backward smoke. Otherwise do not transfer the large payloads
   or train.
4. Transfer and verify only the reuse archive and frozen cache. Require strict
   integrity, the exact cache hashes, eight completed reuse classifications,
   and three pending classifications. Export the exact RunPod Pod ID before
   launch.
5. Run one serial coordinator and one lock, with no worker-slot arguments:

   ```bash
   CUDA_VISIBLE_DEVICES=0 python3 experiments/01-a1-lr-screen/run/runner.py
   ```

6. Monitor manifests, events, process state, disk, and GPU state read-only.
   Retrieve every available attempt, checkpoint, coordinator/setup log,
   checksum inventory, and control-plane record before deletion.
7. Verify retrieved artifacts locally, delete the Pod, and confirm zero Pods,
   zero network volumes, zero endpoints, and `$0/hour` recurring spend.

A nonfinite or divergent scientific outcome is terminal and ineligible, is
retained as evidence, and is never retried. The serial runner stops later
admission on the first escaping failure; any unattempted higher cell remains
unclassified for review. An infrastructure or preflight failure preserves and
retrieves available evidence, then stops without scientific retry, replacement
Pod, additional LR condition, or ad hoc command. Teardown remains authorized
after success or any failure.

After all three cells have reviewed terminal classifications, regenerate the
complete eleven-cell table, curve, and provenance and apply the predeclared
lowest-final-selection-loss rule. No further automatic A1 extension is
authorized.
