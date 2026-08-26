# A1 Three-Cell High-LR Definitive Launch Record

> **Status:** completed under the user-approved one-A40 serial amendment on
> 2026-08-26. Configs `006`–`008` are completed, eligible, and valid; their
> artifacts were retrieved and verified before teardown. The subsequent
> eight-cell decision checkpoint froze `lr_14m=6.4e-2` from config
> `008-a1-lr-6p4e-2`, run `001-20260826-190546-4df1c441`. It is the upper
> tested boundary, and no further A1 LR extension is authorized.

## Authority and Identity

The user's 2026-08-26 instruction to run three additional learning rates above
`8e-3` authorized the exact reviewed cells `1.6e-2`, `3.2e-2`, and `6.4e-2`
and their one definitive execution. The reviewed design commit is
`d80f6a9b6c99bcaec7ddc52e73c1a407a5020a8e`, activated at
`b4ec541a9bfb5dc7e1e22b3ff137fc4facf4b0f9`. The materialized execution recipe
is commit `b586500d28bd9ee6e15319ceb4180008a0b63082`; the definitive runs used the
clean execution commit `d4105722516958df6e9c3cc43b20d6bfd4619d0f`.

| Config | Peak LR | Condition fingerprint | Complete-config SHA-256 |
| --- | ---: | --- | --- |
| `006-a1-lr-1p6e-2` | `1.6e-2` | `088d78d630a4c292211ae4038a9f7ea8be8a824778758c15a161149e4b9891dc` | `ca66f1c07c009a6801ac5ccf038d9d0e1f9cfc320affb40517a744e04ec8b751` |
| `007-a1-lr-3p2e-2` | `3.2e-2` | `97e04832ee14e7929778e7e38237a8005b6028ef4435212714aa9aa7546eed29` | `24b93ac2435717c653f4849eea597698020a4b9abd8a71fcd5ca2fc600ea543d` |
| `008-a1-lr-6p4e-2` | `6.4e-2` | `336adaf43bde4494c90d1f6a7f11d5a4aca7804a60ee75fded15a670a9f6f89e` | `9e6f56a930024b5bf67fa34a9bd94ce3a1903f99a149249cf1b5946a5a2d597e` |

Each new config differs from config `005` only in experiment label, derived
fingerprint, and peak learning rate. There is no change under `src/`,
`constraints/`, or `pyproject.toml` between config `005` execution commit
`a4ddaa5c9897224a9285afae09d2d9c6b07b3cec` and materialization. Random
initialization, forward/loss, task gradients, optimizer/schedule, data order,
validation, checkpoint semantics, and `a1_pretraining_v1` therefore remain on
the accepted active path.

## Required Reuse

The runner classified exactly these accepted attempts as complete before it
created a new attempt:

| Config | Accepted run |
| --- | --- |
| `001-a1-lr-5e-4` | `001-20260825-191155-6b7376de` |
| `002-a1-lr-1e-3` | `001-20260825-191154-b9299c46` |
| `003-a1-lr-2e-3` | `001-20260825-195141-f842c400` |
| `004-a1-lr-4e-3` | `001-20260826-123606-46e7454f` |
| `005-a1-lr-8e-3` | `001-20260826-135546-928279bb` |

No accepted cell was rerun. Before launch, configs `006`–`008` were pending;
the runner used one coordinator, one repository lock, and its existing serial
default path.

## Historical Three-A40 Preparation (Superseded)

The original launch contract prepared one Secure Cloud Pod with exactly three
distinct homogeneous NVIDIA A40 48GB GPUs, one isolated worker per pending
config, and concurrent admission of configs `006`–`008`. It permitted
`CA-MTL-1` or `EU-SE-1`, pinned image
`runpod/pytorch@sha256:4d1721e62b56d345c83b4fd6090664be6daf9312caab5b2e76f23d8231941851`,
a 50GB container disk, no Pod volume, no network volume, a `$2.25` cost cap,
and termination 100 minutes after creation. The point ETC was 57–65 minutes
from creation for the first result and full concurrent tranche.

Pod `9l8jns1uwarkfp` (`osp-a1-hi-lr-d410572`) was allocated under that
contract and passed the base GPU/runtime preflight, then was stopped before Git
or cache transfer and before scientific training. Resume did not reacquire
three co-resident A40s, so the Pod was deleted. Permitted replacement creation
calls allocated no Pod, and inventories were reconciled. This is operational
history, not a scientific attempt or scientific failure. The three-A40 path
was then superseded; its tracked authorization does not authorize another
launch.

## Executed One-A40 Serial Amendment

The user amended only the execution shape: exactly one Secure A40 48GB Pod,
configs `006`–`008` executed serially under one coordinator and lock, maximum
cost `$1.35`, and termination 150 minutes after creation. All scientific,
runtime, transfer, failure, and teardown terms remained unchanged. The
existing runner was invoked without worker-slot arguments, so no concurrent
scientific path was used.

| Field | Executed value |
| --- | --- |
| Pod | `bq45s1hj2262ak` |
| Region and hardware | `CA-MTL-1`; 1× NVIDIA A40 48GB |
| Execution commit | `d4105722516958df6e9c3cc43b20d6bfd4619d0f` |
| Pod created | `2026-08-26T17:27:39.972Z` |
| Scientific launch | `2026-08-26T17:46:11Z` |
| Final completion | `2026-08-26T19:45:33.876Z` |
| Teardown | Pod deleted after verified retrieval; zero Pods and zero network volumes confirmed |

The pinned runtime was Python 3.12.3, Torch 2.11.0+cu128, CUDA 12.8,
Transformers 5.12.1, and a compatible host driver. The transfer used only the
verified clean Git bundle, the five accepted reuse attempts, and the frozen
MiniPile cache; credentials, `docs/humans/main.pdf`, calibration attempts,
nonaccepted attempts, and unrelated untracked files remained excluded. POSIX
path normalization was confined to copied cache metadata.

The unchanged failure contract made a nonfinite scientific outcome terminal
and ineligible without retry, and required an infrastructure interruption to
preserve its attempt and stop for classification. No retry or replacement was
used. Terminal attempts, the coordinator log, and control-plane evidence were
retrieved before Pod deletion.

## Results and Artifact Acceptance

| Config | Run | Final selection loss | Classification |
| --- | --- | ---: | --- |
| `006-a1-lr-1p6e-2` | `001-20260826-174611-04b42898` | `4.112285005418878` | completed; eligible; valid |
| `007-a1-lr-3p2e-2` | `001-20260826-182559-bb05a50c` | `4.082745991255107` | completed; eligible; valid |
| `008-a1-lr-6p4e-2` | `001-20260826-190546-4df1c441` | `4.0587728086270785` | completed; eligible; valid |

All three attempts completed 1,526 optimizer updates / 400,031,744 training
tokens, produced the required eight files, and remained finite. Config `008`
logged exactly two clipped optimizer steps, at steps 1 and 10; this did not
trigger a failure rule.

| Config | Retrieved files | Retrieved bytes | Final checkpoint SHA-256 |
| --- | ---: | ---: | --- |
| `006-a1-lr-1p6e-2` | 8 | 56,502,513 | `2aab9dcb7b9f22a4bf6a808ca15fda9cb2d65800710cb0b640bd061fe7f3692a` |
| `007-a1-lr-3p2e-2` | 8 | 56,502,491 | `73026efc16a27ffa58fab667ae60e13fc11df916a6a24b8999fa2ebe12db91d7` |
| `008-a1-lr-6p4e-2` | 8 | 56,502,344 | `5205d7cfa3c8bf47a481d8ede10df6585aaaecb75498e755734a856cdc307849` |

All remote/local per-file hashes matched. The retrieved coordinator log is
`tmp/a1-hi-lr-runner-bq45s1hj2262ak.log` (1,617 bytes; SHA-256
`01f23203e8995b8b60818dc1bb3353b4319211018b2586d551fc6b43ab08beea`).
RunPod billing was still ingesting the closed `19:00–20:00Z` hour in
near-exact five-minute slices at the `2026-08-26T20:15:08Z` cutoff. For Pod
`bq45s1hj2262ak`, the API subtotal of `$0.77430647413712` is therefore a lower
bound, not a final charge. The independently observed lifecycle through about
`19:45–19:46Z` implies an estimated final successful-Pod cost of
`$0.998–$1.005`; this remains an inference pending billing convergence and is
below the `$1.35` cap. The earlier preparation Pod `9l8jns1uwarkfp` is separate
and has an exact posted charge of `$0.030331766232848167`. Final inventories
show zero Pods and zero network volumes, so recurring spend is `$0/hour`.

The separate decision checkpoint applied the predeclared
lowest-final-selection-loss rule to all eight completed, eligible, valid
cells. It selected config `008-a1-lr-6p4e-2`, run
`001-20260826-190546-4df1c441`, checkpoint
[`checkpoints/final`](../../experiments/01-a1-lr-screen/raw/008-a1-lr-6p4e-2/001-20260826-190546-4df1c441/checkpoints/final/), and froze
`lr_14m=6.4e-2`. The [curve](../../experiments/01-a1-lr-screen/figs/01-a1-learning-rate-screen.pdf),
[table](../../experiments/01-a1-lr-screen/figs/01-a1-learning-rate-screen.md),
and [provenance](../../experiments/01-a1-lr-screen/figs/01-a1-learning-rate-screen.provenance.json)
use the deterministic recipe committed at
`56d7771a84ea378be09e66b7fc270cab29e17b0c`; the operational record is commit
`be365472ff775493984b0c5e69b6250e03d1392e`. Because the selected point is the
upper tested boundary, it is the best tested LR at the fixed A1 horizon, not a
global optimum. No further A1 LR extension is authorized.
