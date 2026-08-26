# A1 `8e-3` Boundary Cell — Definitive Launch Packet

## Authority and Identity

The user's 2026-08-26 direction to extend the factor-two grid and run the new
cell is treated as authorization for this exact scientific change and its
single execution. The reviewed design commit is
`3710dfdd416ac3484516c4d8c2162692346fe7e9`, activated at
`3cd5317f29543e94b922b5d6e0ba00c18bf2033a`. The materialized execution
commit is `6df2206a215579853436d43c5fe0b5b6aa7620a9`.

| Item | Exact value |
| --- | --- |
| Config | `005-a1-lr-8e-3` |
| Condition fingerprint | `9cc6a74400386deee36fe706aac1967fd79facd156d7e2ca7cacee56a2a22167` |
| Complete-config SHA-256 | `9383ccc371a86e64fff2beb02057f15ff84256bf505ce1449e0ff3df5d2945ea` |
| Scientific condition | seed 0; peak LR `8e-3`; 1,526 updates; 400,031,744 input tokens |
| Physical batch | A40 48GB; BF16; microbatch 16; accumulation 8 |

Config `005` differs from config `004` only in the experiment label, derived
condition fingerprint, and peak learning rate. No `src/paper_exp`, dependency
constraint, or package-definition file changed between config `004` execution
commit `f235081239cd67831684e4174531992af4253e9c` and materialization. The active
training implementation remains `a1_pretraining_v1`.

## Required Reuse and Admission

The serial runner must classify exactly these accepted attempts as complete:

| Config | Accepted run |
| --- | --- |
| `001-a1-lr-5e-4` | `001-20260825-191155-6b7376de` |
| `002-a1-lr-1e-3` | `001-20260825-191154-b9299c46` |
| `003-a1-lr-2e-3` | `001-20260825-195141-f842c400` |
| `004-a1-lr-4e-3` | `001-20260826-123606-46e7454f` |

Config `005` must be pending. Any missing, duplicate, failed, running,
statusless, or inconsistent reuse attempt stops before mutation. Run exactly
one serial coordinator and lock; do not rerun configs `001`–`004`, launch a
second case runner, pack another worker onto the GPU, or retry a scientific
outcome.

## RunPod Envelope

Use exactly one Secure NVIDIA A40 48GB Pod, CA-MTL-1 preferred or EU-SE-1
fallback, with pinned image
`runpod/pytorch@sha256:4d1721e62b56d345c83b4fd6090664be6daf9312caab5b2e76f23d8231941851`,
50GB container disk, no Pod volume, and no network volume. Require A40 compute
capability 8.6, Python 3.12.3, Torch 2.11.0+cu128, CUDA runtime 12.8,
Transformers 5.12.1, BF16 support, a compatible host driver, and a bounded CUDA
forward/backward smoke before transferring the large cache.

The Pod rate is $0.44/GPU-hour. The expected create-to-retrieval duration is
55–65 minutes; set a $0.75 maximum envelope and backend deletion 90 minutes
after creation. Never keep more than one Pod. A failed allocation may fall
through from CA-MTL-1 to EU-SE-1, but no replacement is created after a Pod
successfully allocates.

## Container-Local Operational Exception

Run the bundle, checkout, cache, exact reuse set, logs, raw output, checkpoint,
and artifact packaging from container-local `/root`. This explicit exception
avoids the `/workspace` MFS stalls observed during config `004`; it changes no
scientific input. It is compensated by the backend deadline, read-only
monitoring, early retrieval of small records when practical, and complete
local artifact/hash verification before teardown.

Transfer only the clean Git bundle, the frozen MiniPile cache, and the four
accepted reuse attempts. Exclude credentials, `docs/humans/main.pdf`, unrelated
untracked files, calibration attempts, and nonaccepted raw attempts. POSIX path
normalization is permitted only in copied cache metadata.

After terminal completion or failure, retrieve the complete config `005`
attempt plus setup/runner logs and checksums, verify it locally, then delete the
Pod and confirm zero Pods, zero network volumes, and $0/hour.
