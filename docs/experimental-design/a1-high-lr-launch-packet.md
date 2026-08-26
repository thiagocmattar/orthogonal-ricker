# A1 Three-Cell High-LR Definitive Launch Packet

## Authority and Identity

The user's 2026-08-26 instruction to run three additional learning rates above
`8e-3` is treated as authorization for the exact reviewed cells `1.6e-2`,
`3.2e-2`, and `6.4e-2`, their one definitive execution, and the bounded
RunPod envelope below. The reviewed design commit is
`d80f6a9b6c99bcaec7ddc52e73c1a407a5020a8e`, activated at
`b4ec541a9bfb5dc7e1e22b3ff137fc4facf4b0f9`. The materialized execution recipe
is commit `b586500d28bd9ee6e15319ceb4180008a0b63082`.

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

## Required Reuse and Admission

The runner must classify exactly these accepted attempts as complete before
creating any new attempt:

| Config | Accepted run |
| --- | --- |
| `001-a1-lr-5e-4` | `001-20260825-191155-6b7376de` |
| `002-a1-lr-1e-3` | `001-20260825-191154-b9299c46` |
| `003-a1-lr-2e-3` | `001-20260825-195141-f842c400` |
| `004-a1-lr-4e-3` | `001-20260826-123606-46e7454f` |
| `005-a1-lr-8e-3` | `001-20260826-135546-928279bb` |

Configs `006`–`008` must be pending. Use one coordinator, one repository lock,
one writable checkout, and exactly three worker slots mapped one-to-one to
three distinct homogeneous `NVIDIA A40` GPUs on one machine. All three pending
configs are initially admitted, so an escaping failure drains the other
workers to terminal state and leaves no planned cell unadmitted. Do not rerun
configs `001`–`005`, start another runner, pack workers on one GPU, mix GPU
types, or dispatch across Pods.

## RunPod Envelope and ETC

Provision at most one live Secure Cloud Pod named `osp-a1-hi-lr-<short-sha>`
with exactly 3× NVIDIA A40 48GB on one machine. Permit `CA-MTL-1` and
`EU-SE-1`, use image
`runpod/pytorch@sha256:4d1721e62b56d345c83b4fd6090664be6daf9312caab5b2e76f23d8231941851`,
50GB container disk, no Pod volume, no network volume, and SSH with the
validated RunPod public key. Failed allocation calls before any Pod exists may
be repeated at most three times within 30 minutes; after one Pod is created,
no replacement Pod is authorized.

The current Secure A40 price is `$0.44/GPU-hour`; prior 50GB container-disk
billing adds approximately `$0.007/hour` per Pod. Accepted A1 runs took
39m45s–41m45s end to end, and config `005` took 56m57s from Pod creation
through teardown. With shared setup and three concurrent workers, the first
result and full tranche are both expected 57–65 minutes after creation, with
approximately ±15 minutes uncertainty for allocation, setup, transfer, and
retrieval. Expected spend is approximately `$1.27`; maximum total spend is
`$2.25`. Set backend termination exactly 100 minutes after successful Pod
creation.

## Runtime, Transfer, and Failure Contract

Before transferring the large cache, require all of the following on the
allocated host:

- three `NVIDIA A40` GPUs with distinct UUIDs, 48GB each, compute capability
  8.6, native BF16 support, and no competing GPU process;
- Python 3.12.3, Torch 2.11.0+cu128, CUDA runtime 12.8, Transformers 5.12.1,
  and a compatible host driver;
- a bounded CUDA BF16 forward/backward smoke on every worker GPU.

Transfer only a verified clean Git bundle for the execution commit, the five
accepted reuse attempts, and the frozen MiniPile cache. Exclude credentials,
`docs/humans/main.pdf`, calibration attempts, nonaccepted attempts, and
unrelated untracked files. POSIX path normalization is permitted only in the
copied cache metadata. Run from container-local `/root` to avoid the prior
mounted-filesystem stalls; the backend deadline compensates for ephemeral
storage.

Scientific divergence or another nonfinite outcome from the frozen recipe is
a resolved ineligible cell and is not retried. An infrastructure interruption
preserves the exact failed attempt and stops for classification; no automatic
scientific or infrastructure retry is authorized. Retrieve all terminal
attempts, logs, control-plane records, and checksums before deleting the Pod.
Finally verify zero Pods, zero network volumes, and `$0/hour`.
