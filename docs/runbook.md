# Experiment Runbook

This is the operational contract for definitive experiments. The reviewed
[`experiment_plan.md`](experiment_plan.md) is the scientific authority.
The modular proposal and physical-case reuse rules are indexed in
[`experimental-design/README.md`](experimental-design/README.md).

## 1. Launch Gate

Scientific work remains blocked while the definitive plan says:

```text
Plan status: placeholder
```

The infrastructure-only
`experiments/00-infrastructure-smoke/run/00-smoke.yaml` is exempt. A
scientific config requires its case group in the reviewed scope. A launch also
requires a committed scaffold recipe, clean checkout, calibrated ETC, and
explicit launch approval.
After A1 closeout, the plan is at placeholder status; the next scientific
config or run requires newly reviewed scope.
Once attempted, a config is immutable. A scientific change gets a new config;
an infrastructure retry gets a new run attempt under the same config.

## 2. Turn the Plan Into Launches

The plan's phases do not map one-to-one to processes. A screen, a
review-dependent selection, and a promotion are separate launches because the
later config list does not exist until earlier evidence is reviewed.

Before allocating a config, resolve its case group and condition fingerprint
under [`experimental-design/cases.yaml`](experimental-design/cases.yaml) and
[`experimental-design/run-reuse.md`](experimental-design/run-reuse.md). Reuse
an existing matching config/seed; a later stage is a consumer, not a duplicate
owner. The group must appear on `experiment_plan.md`'s raw
`Reviewed case groups:` line.

Use one numeric scaffold ID and keep the phase/tranche in the name:

```text
experiments/NN-<phase>-<tranche>/
  run/
    runner.py
    CCC-<case>.yaml
  raw/
  figs/
```

Scaffold numbers define launch order. Config numbers are globally unique and
define case order across scaffolds. Do not reserve or create later scaffold
recipes before their inputs and promotion rules are known.

Each case runner contains an ordered `CONFIGS` tuple and a call to
`paper_exp.runner.run_launch`. A reviewed bounded execution exception may add
tracked operational authorization bound to exact config IDs, worker count,
and GPU identity. A1's dormant authorization metadata is historical and
authorizes no rerun or new work. Shared mechanics belong in the parent;
science belongs in configs and the reviewed plan.

## 3. Preflight and ETC

Before a launch, verify:

- exact tranche and dependencies in the reviewed plan;
- catalog membership, one config per condition fingerprint/seed, and every
  declared exact or functional-equivalence alias;
- ordered config list, with no duplicates or unresolved `TODO:` values;
- matched seeds and data ordering where the plan requires them;
- model, data, tokenizer, optimizer, schedule, budget, and validation fields;
- `model.initialization: random` for Pythia pretraining;
- cache metadata and available disk/GPU memory;
- current branch, Git SHA, clean checkout, and absence of another launch;
- focused tests, `make test`, and `make check`.

Before starting, report the estimated duration and local completion time for
the first run and the complete tranche. State the throughput evidence,
assumptions, and uncertainty, then wait for explicit launch approval. If no
defensible same-hardware estimate exists, run the plan-approved calibration
first.

`calibrate` means exactly 600 accumulated seconds inside completed
production-shaped optimizer steps on the launch hardware. Setup, validation,
diagnostics, and checkpoint timing are measured and reported separately, so
total elapsed calibration time is longer than 600 seconds. The operational
timer is checked only at completed optimizer-step boundaries and cannot enter
or truncate definitive pretraining. A scientific learning-rate screen is a
normal training tranche and must use a case runner. `OPS-03` closes only after
the local implementation is paired with a same-hardware timing artifact.

The historical A1 calibration permitted solo or bounded concurrent execution.
Concurrent calibration accepted distinct committed configs from one scaffold
under one coordinator and repository lock. It required at least two explicit
worker slots, one process per distinct homogeneous BF16-capable physical GPU,
and recorded the GPU, Torch, CUDA runtime, launch, slot, and config identity in
every calibration manifest. It was not pretraining evidence and does not
authorize future work.

## 4. Launch

Single-config data preparation and throughput calibration commands remain
available:

```bash
make prepare-data CONFIG=experiments/<scaffold>/run/<config>.yaml
make calibrate CONFIG=experiments/<scaffold>/run/<config>.yaml
```

The historical approved two-GPU A1 calibration shape was:

```bash
python -m paper_exp.cli calibrate \
  --config experiments/<a1-scaffold>/run/<first>.yaml \
  --config experiments/<a1-scaffold>/run/<second>.yaml \
  --config experiments/<a1-scaffold>/run/<third>.yaml \
  --worker-slot gpu-0=0 \
  --worker-slot gpu-1=1
```

The coordinator admits two configs first and the third when one slot becomes
free. Run the predeclared solo calibration separately before this command when
measuring concurrency overhead. Calibration concurrency does not authorize a
definitive training launch or cloud spend.

Run every definitive training tranche through its case runner, including a
tranche with only one config:

```bash
python experiments/NN-phase-tranche/run/runner.py
```

Serial execution is the default. The completed original three-cell A1 launch
historically used the same runner with two explicit A40 worker slots. Configs
`004` and `005` later completed serially. High-LR configs `006`–`008` completed
serially on one A40 from clean commit `d410572` with this plain invocation:

```bash
CUDA_VISIBLE_DEVICES=0 python3 experiments/01-a1-lr-screen/run/runner.py
```

No `--worker-slot` argument was passed, so the runner used its serial-default
path under one coordinator and lock. The superseded three-A40 authorization
remains dormant historical metadata only. All exact A1 configs `001`–`008` are
completed evidence and must never be rerun.

The parent runner:

- requires the runner and configs to be tracked;
- requires every config to be a direct sibling of the scaffold runner;
- requires each config's `output.dir` to name that scaffold's exact `raw/`;
- validates the full ordered config list before starting;
- holds one repository launch lock;
- rechecks all existing attempt states under that lock;
- scopes resume decisions to `mode: pretrain`; preparation, calibration, and
  diagnostic attempts cannot satisfy or block a training case;
- reuses exactly one coherent completed pretraining attempt for an unchanged
  config;
- retries coherent failed pretraining attempts only with the explicit
  `--retry-failed` recovery flag;
- aborts before mutation on running, statusless, inconsistent, or ambiguous
  pretraining state;
- executes one config at a time by default;
- stops admitting new configs on the first escaping failure and drains every
  already-admitted worker to terminal state.

Never start two case runners in parallel. Data preparation and diagnostics run
one config at a time unless the final plan explicitly adds a serial runner for
that workflow.

The isolated worker engine is implemented and live-validated for
infrastructure smoke, bounded calibration, and the completed original A1
launch. No current scientific scope authorizes bounded-worker execution; new
work remains serial by default unless a newly reviewed plan says otherwise.
Multiple case runners, same-GPU packing, heterogeneous worker slots, and
multi-Pod dispatch remain unsupported.

### RunPod Operations

This subsection is the operator procedure for RunPod. It does not relax the
launch gate above. Only case groups in the reviewed `experiment_plan.md` scope
may run scientifically, and every definitive launch and billable envelope
requires its own explicit approval. Other RunPod use is limited to explicitly
identified infrastructure profiling and smoke work.

#### Skill and tool routing

For every multi-step RunPod task, load the installed `runpod` router skill
first. Then use the smallest applicable lane:

- `runpod-usage` for Pod-versus-serverless, GPU, storage, and batch-training
  design;
- `runpod-mcp` for catalog and capacity reads and for Pod, volume, log,
  billing, and lifecycle operations;
- `runpodctl` for SSH-key registration, fresh SSH connection details, file
  transfer, and reproducible terminal workflows;
- `companion-clis` only when GitHub, Hugging Face, Docker, or RunPod S3 is
  actually required.

Pretraining and its infrastructure smoke use Pods, not Serverless or Flash.
The connected MCP schemas and `runpodctl ... --help` are authoritative when a
skill snapshot and the installed tool differ. MCP OAuth authenticates the MCP
control plane only; it does not authenticate `runpodctl`.

The repository-ignored Windows client used by this workflow is
`tmp/runpodctl-v2.8.0.exe`, pinned to official release `v2.8.0` with SHA-256
`85c8e7daac6f11a1e7047a06113dfbed0d778cc0aac6e57560ce116aca09ada7`.
Before its first mutating use, the operator must run the following locally and
enter the API key only in that interactive flow:

```powershell
.\tmp\runpodctl-v2.8.0.exe doctor
```

Do not paste the key into chat. Recheck the binary digest and `--help` schema
after replacement or upgrade. This version supports backend
`--terminate-after <absolute-UTC-time>` and `--stop-after`; use termination for
the disposable smoke so cleanup does not depend on the agent remaining alive.

Never print, copy into chat, or commit an API key, private SSH key, registry
credential, or model/dataset token. Test only whether a credential resolves.

#### SSH identity and validation

The expected Windows operator identity is
`$env:USERPROFILE\.ssh\id_ed25519_runpod`; only its public half or fingerprint
may be inspected or registered. Checks and the approved repair on 2026-08-24,
followed by the disposable live verification on 2026-08-25, established:

- the private and public files parse as the same ED25519 key pair;
- fingerprint
  `SHA256:ew2/c7ja1vPEtMZEP+sAHT2NzJhW+Np7HdNlWdRXcc8`;
- the private key is owned by `TCML\thima`, has protected inheritance, and has
  exactly one allow rule granting that operator full control;
- Windows OpenSSH reads it without an interactive passphrase and derives the
  exact stored public key;
- the Windows `ssh-agent` remains disabled/stopped but is not required when the
  key is supplied explicitly with `-i`;
- the pinned workspace-local `runpodctl` binary is verified and its one-time
  local `doctor` authentication completed without exposing the API key; and
- direct SSH to a Secure Cloud Pod accepted this exact key and returned
  `RUNPOD_SSH_OK` non-interactively.

That result validates the identity and procedure, not a future allocation's
ephemeral endpoint. Every new or restarted Pod still requires the live
assertion below using fresh connection details. The connected MCP `create_pod`
tool accepts `sshPublicKey`; pass the contents of the matching `.pub` file
directly so the official RunPod image authorizes it and exposes `22/tcp`. This
avoids depending on account-level key registration. If a different creation
lane does not inject the key explicitly, confirm that the exact public
fingerprint is registered before Pod boot.

Do not loosen or replace the repaired private-key ACL automatically. If a later
live test reports `Permission denied (publickey)`, verify explicit injection or
registration of the fingerprint above before changing the Pod or key.

The definitive check is the first assertion of the approved combined
GPU/concurrency smoke, not a standalone SSH exercise. The disposable Pod must
have SSH enabled and a backend hard-termination deadline. After retrieving its
current host and port, require this non-interactive command to return exactly
`RUNPOD_SSH_OK` with exit code zero:

```powershell
$key = "$env:USERPROFILE\.ssh\id_ed25519_runpod"
ssh -o BatchMode=yes -o IdentitiesOnly=yes `
  -o StrictHostKeyChecking=accept-new `
  -o UserKnownHostsFile=tmp/runpod_known_hosts `
  -o ConnectTimeout=15 -i $key -p <port> root@<host> `
  "printf RUNPOD_SSH_OK"
```

Use a repository-ignored known-hosts file as shown; do not disable host-key
checking globally. Registration and local integrity alone are not substitutes
for this live check.

#### Discover and resume before creating

At the start of every agentic session, use MCP to list Pods and network
volumes before any creation. Empty lists are valid. For every existing project
resource, retrieve its current state and match its name, Pod/volume ID, data
center, GPU, image, mount path, automatic-termination deadline, and intended
Git SHA. Use names beginning `osp-` and including purpose plus a short Git SHA
so discovery is unambiguous.

After every Pod start or restart, retrieve fresh SSH connection details; host
and port may change. On connection, verify the checkout path, clean Git SHA,
environment identity, active process, exact attempt directory, terminal or
running manifest, and latest event before acting. An existing runner or an
ambiguous `running` attempt means monitor only: never start another runner or
infer state from chat history.

Do not assume the Pod ID is injected into the container. Before launching a
worker, compare the MCP/control-plane Pod ID with any existing
`RUNPOD_POD_ID`; if the variable is absent, export the exact control-plane ID:

```bash
test -n "${RUNPOD_POD_ID:-}" || export RUNPOD_POD_ID="<exact-pod-id>"
test "$RUNPOD_POD_ID" = "<exact-pod-id>"
```

The running and terminal manifests must then record that ID. A separate
control-plane record was accepted as an explicit limitation for the 2026-08-25
A1 calibration, but it is not a substitute for manifest-level Pod identity in
definitive runs.

#### Persistence and concurrency

Mount the Pod volume at `/workspace`. Keep the execution checkout, dependency
and model caches, detached-process logs, ignored raw artifacts, and checkpoints
below that mount; container-only paths such as `/root` are ephemeral. A Pod
volume survives stop/restart but is deleted with the Pod, so retrieve required
artifacts before termination. Pin the container image, dependency constraints,
and clean Git SHA. Long-running commands must survive SSH loss and write their
log to `/workspace`.

One multi-GPU Pod with one authoritative coordinator and one writable checkout
is the supported parallel shape when a reviewed scope explicitly authorizes
it. A1's bounded-worker metadata is completed history and authorizes no rerun;
the actual high-LR cells ran serially on one A40. No current scientific scope
authorizes parallel execution. Do not run independent case runners or allow
concurrent writable checkouts on one volume. Multi-Pod scientific execution
requires a future reviewed contract rather than an ad hoc extension of the
local lock.

#### Combined GPU smoke and hardware profile

Run the local fault-injection proof before requesting cloud resources:

```bash
python -m paper_exp.cli smoke \
  --config experiments/00-infrastructure-smoke/run/00-smoke.yaml \
  --worker-slot smoke-0=0 --worker-slot smoke-1=1
```

The approved live procedure is one combined operation:

1. Re-list Pods and network volumes through MCP; reconcile every existing
   `osp-` resource before creation.
2. Create one Secure Cloud two-GPU Pod named `osp-smoke-<short-git-sha>` with
   SSH, `22/tcp`, an ephemeral Pod volume at `/workspace`, and an absolute
   backend termination time no more than four hours later. Do not create a
   network volume for this disposable operation.
3. Make the non-interactive `RUNPOD_SSH_OK` assertion above. If it fails, retain
   the deadline, collect the resource state, and either correct explicit key
   injection or terminate; do not turn it into an open-ended SSH investigation.
4. Transfer a Git bundle for the approved SHA, verify the bundle, create one
   checkout under `/workspace`, install the exact CUDA 12.8 Torch wheel and
   then the constraint snapshot using the verified block below, and require a
   clean checkout at that SHA. Record the Pod image, driver, CUDA, Python,
   Torch, Transformers, and GPU identities.
5. Run the two-GPU smoke below. It proves concurrent overlap, one injected
   failure with complete draining of admitted work, explicit unchanged
   recovery, completed-work reuse, distinct stable physical GPU UUIDs, BF16
   execution, and disjoint durable attempt roots. The deterministic scheduler
   test separately proves that no work is admitted after the coordinator
   observes a failure; work admitted before that observation remains valid and
   is drained.
6. With both GPUs otherwise idle, run the 14M, 70M, and 410M hardware profiles
   sequentially on one physical GPU. Warm each pinned Hugging Face config cache
   before profiling; do not overlap profiles or any other GPU process.
7. Retrieve the smoke and profile artifacts plus checksums and setup log,
   verify them locally, terminate the Pod, and re-list Pods and volumes until
   the project inventory is empty.

The pinned base image for this smoke is:

```text
runpod/pytorch@sha256:4d1721e62b56d345c83b4fd6090664be6daf9312caab5b2e76f23d8231941851
```

The constraints file pins Torch's public version, but not its CUDA wheel
variant or package index. On the pinned CUDA 12.8 image, a one-step editable
install was observed to select the CUDA 13.0 (`cu130`) wheel, which could not
initialize against the allocated host driver. Install and verify the CUDA 12.8
wheel first, then apply the repository constraints:

```bash
python3 -m pip install --break-system-packages --force-reinstall \
  --index-url https://download.pytorch.org/whl/cu128 \
  'torch==2.11.0+cu128'
python3 -m pip uninstall --break-system-packages -y torchvision torchaudio
python3 -m pip install --break-system-packages \
  -c constraints/requirements-ci.txt -e '.[dev]'
python3 -m pip check
python3 - <<'PY'
import torch

assert torch.__version__ == "2.11.0+cu128"
assert torch.version.cuda == "12.8"
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 2
print(torch.__version__, torch.version.cuda)
PY
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version \
  --format=csv,noheader
```

Use `--index-url`, not `--extra-index-url`, for the Torch step. The local
`+cu128` build satisfies the constraints file's public `torch==2.11.0` pin, so
do not make the cross-platform constraints file CUDA-specific. The image digest
pins the container, not RunPod's host driver; record the observed driver on
every allocation. A changed driver/runtime identity requires fresh smoke and
profile evidence under new work roots. The live environment record, not the
image tag alone, proves the package versions actually used.

The Pod creation pattern is:

```powershell
.\tmp\runpodctl-v2.8.0.exe pod create `
  --name osp-smoke-<short-git-sha> --cloud-type SECURE `
  --gpu-id "<current-catalog-gpu-id>" --gpu-count 2 `
  --image "<immutable-image-above>" `
  --container-disk-in-gb 30 --volume-in-gb 50 `
  --volume-mount-path /workspace --ports "22/tcp" `
  --terminate-after "<absolute-UTC-deadline>"
```

Use the current catalog result rather than copying an old GPU ID, price, or
data center from this document. The remote concurrency command is:

```bash
python -m paper_exp.cli smoke \
  --config experiments/00-infrastructure-smoke/run/00-smoke.yaml \
  --worker-slot gpu-0=0 --worker-slot gpu-1=1 --require-cuda
```

The infrastructure-only profile identities are operational sizing inputs, not
approved paper pins:

| Architecture | Revision |
| --- | --- |
| `EleutherAI/pythia-14m-deduped` | `7386d9a4ae45aef494a6e704910394def3037fc5` |
| `EleutherAI/pythia-70m-deduped` | `f289af01c98892bc173f73d2075d1b9ee19af190` |
| `EleutherAI/pythia-410m-deduped` | `b5e8535141902c0e985cea61fd02afe7fe86af32` |

For profiling, derive `--gpu-class` from the exact Torch runtime name seen by
the isolated worker; do not reuse the RunPod catalog GPU ID or append catalog
or VRAM display text. Device memory is recorded separately. Create only the
shared profile parent before the first run: the profiler creates each leaf work
root but deliberately does not create missing ancestors. A leaf must be absent
for its first run or already be the same profile-owned root for exact resume.

```bash
GPU_CLASS="$(CUDA_VISIBLE_DEVICES=0 python3 -c \
  'import torch; print(torch.cuda.get_device_name(0))')"
PROFILE_PARENT=/workspace/osp-profile
mkdir -p "$PROFILE_PARENT"

python3 -m paper_exp.cli profile-hardware \
  --architecture <architecture> --revision <revision> \
  --gpu-class "$GPU_CLASS" \
  --candidate-microbatches 1,2,4,8,16,32,64,128 --repeats 2 \
  --cuda-device 0 --worker-timeout-seconds 1200 \
  --container-image "<immutable-image-above>" \
  --work-root "$PROFILE_PARENT/<model>"
```

Selection is only over the supplied grid: every listed candidate must run.
All repeats must fit within 90% reserved VRAM; candidates within 2% of the
fastest median core-update throughput prefer lower worst-repeat reserved
memory. No loss, sparsity, or other scientific result is retained. These
measurements select the physical-batch decomposition and estimate smoke
duration only. They are not end-to-end ETC evidence; the separate 600-second
same-hardware calibration supplies the launch ETC after scientific configs are
reviewed.

#### Approval, cost guard, and teardown

Before creating any billable resource, report the exact purpose, cloud tier,
GPU SKU and count, current hourly price, data center, image, volume and storage
price, maximum duration, maximum projected cost, automatic-termination
deadline, and cleanup intent. If the capacity API does not expose placement
before allocation, state the permitted data-center scope explicitly and record
the selected center immediately after creation. Obtain explicit approval for
that resource envelope. Infrastructure-smoke approval is separate from
scientific-launch approval.

Before removing compute, verify that required logs and artifacts are durable.
Delete the Pod first, confirm that it is absent, and then re-list Pods and
volumes. Delete a temporary volume only after its exact contents have been
verified as copied or disposable; volume deletion is destructive, while a
retained volume continues to bill. Report every retained resource and its
recurring cost. A timeout or failed setup does not imply that creation failed:
always discover and clean up by resource ID.

## 5. Monitor Without Mutation

Inspect the active run's `manifest.json` and `events.jsonl`, plus the exact
runner process, GPU state, and free disk space. Report:

- completed, active, failed, and remaining configs in the case runner;
- active scaffold, config, run directory, step, tokens seen, and latest event
  time;
- latest finite task and method-specific metrics;
- recent throughput;
- updated current-run and full-tranche ETCs;
- stale events, nonfinite values, low disk, or other risks.

Monitoring must not rewrite artifacts or terminate a process by name, guessed
PID, GPU ownership, or elapsed time.

## 6. Completion, Failure, and Retry

After each run:

1. Read the terminal manifest.
2. Verify scaffold/config/run identity, launch provenance, completed budget,
   and finite required metrics.
3. Verify `config.yaml`, `manifest.json`, `metrics.json`, `predictions.jsonl`,
   `events.jsonl`, and required checkpoints.
4. Run method-specific checks before using the run as diagnostic or paper input.
5. Record terminal status, reviewed case class, evidence status, and
   limitations in `docs/experiment_log.md`.

Preserve failed attempts and valid negative results. Never overwrite an
attempt or rewrite it to manufacture completion. A retry creates a new run
directory. A changed scientific input creates a new config.

After recording a reviewed infrastructure failure and its recovery action,
restart the same unchanged case runner explicitly:

```bash
python experiments/NN-phase-tranche/run/runner.py --retry-failed
```

Without that flag, a coherent failed attempt stops before the lock or any new
attempt. With it, the parent skips coherent completed configs and resumes in
config order. The flag is the operator's attestation that the exact failed
attempt was reviewed and recorded as infrastructure; the runner does not infer
that classification from the experiment log. Never use it for a terminal
scientific failure, edit the runner list to remove earlier cases, rerun a
completed config manually, or create a replacement config/seed. Multiple
completed attempts, a running or statusless attempt, changed config snapshot,
symlink/non-directory state, or inconsistent artifacts stop recovery for
read-only review.

## 7. Handoff

A live handoff states the Git SHA, exact scaffold and case runner, ordered
configs, completed/active/remaining counts, active run and process, latest
metrics, current-run and full-tranche ETCs, failures, and next plan-authorized
action. For RunPod it also states Pod and volume IDs and names, data center,
GPU, pinned image, mount and checkout paths, active command/process, exact
attempt directory, automatic-termination deadline, and retain/delete intent.
The receiving agent re-lists resources and retrieves fresh SSH details rather
than trusting copied host/port values. It verifies files and processes; chat
text is not evidence.
