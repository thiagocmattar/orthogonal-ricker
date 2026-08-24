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

Each case runner contains only an ordered `CONFIGS` tuple and a call to
`paper_exp.runner.run_launch`. Shared mechanics belong in the parent; science
belongs in configs and the reviewed plan.

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

`calibrate` means a 600-second production-shaped timing sample on the launch
hardware, with setup, training, validation, diagnostics, and checkpoint costs
reported separately. Its operational duration cap must not change the
immutable scientific config or truncate definitive pretraining; this remains
blocked until workboard item `OPS-03` is implemented. A scientific
learning-rate screen is a normal training tranche and must use a case runner.

## 4. Launch

Single-config data preparation and throughput calibration commands remain
available:

```bash
make prepare-data CONFIG=experiments/<scaffold>/run/<config>.yaml
make calibrate CONFIG=experiments/<scaffold>/run/<config>.yaml
```

Run every definitive training tranche through its case runner, including a
tranche with only one config:

```bash
python experiments/NN-phase-tranche/run/runner.py
```

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
- executes one config at a time;
- stops immediately on the first escaping failure.

Never start two case runners in parallel. Data preparation and diagnostics run
one config at a time unless the final plan explicitly adds a serial runner for
that workflow.

Concurrent execution is planned but remains blocked by workboard items
`CLOUD-01`, `OPS-05`, and `OPS-06`. The reviewed concurrent design must retain
one authoritative case-runner coordinator and one complete-tranche preflight,
then dispatch distinct immutable configs to bounded isolated subprocess/Pod/GPU
slots with exact-once claims and disjoint attempt roots. Multiple independent
case runners remain forbidden. On a worker failure, the coordinator stops
admitting new configs, lets already-running siblings publish terminal state,
and exits nonzero; it never kills workers in a way that strands `running`
manifests. A multi-Pod coordinator cannot rely on this repository's local lock.

Before enabling that mode, `OPS-06` must demonstrate actual concurrent GPU
workers, device/output isolation, durable artifact collection, completion
reuse, failure draining, and resource teardown using infrastructure-only smoke
inputs. Same-GPU process packing is allowed only if the `OPS-04` measurements
show a useful aggregate-throughput gain for the frozen physical batch.

### RunPod Operations

This subsection is the operator procedure for RunPod. It does not relax the
launch gate above: while `experiment_plan.md` is a placeholder, RunPod may be
used only for explicitly identified infrastructure profiling and smoke work.

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

Never print, copy into chat, or commit an API key, private SSH key, registry
credential, or model/dataset token. Test only whether a credential resolves.

#### SSH identity and validation

The expected Windows operator identity is
`$env:USERPROFILE\.ssh\id_ed25519_runpod`; only its public half or fingerprint
may be inspected or registered. Checks and the approved repair on 2026-08-24
established:

- the private and public files parse as the same ED25519 key pair;
- fingerprint
  `SHA256:ew2/c7ja1vPEtMZEP+sAHT2NzJhW+Np7HdNlWdRXcc8`;
- the private key is owned by `TCML\thima`, has protected inheritance, and has
  exactly one allow rule granting that operator full control;
- Windows OpenSSH reads it without an interactive passphrase and derives the
  exact stored public key;
- the Windows `ssh-agent` remains disabled/stopped but is not required when the
  key is supplied explicitly with `-i`;
- `runpodctl` has no local installation or authentication; and
- a live Pod login remains unverified.

Do not describe this as end-to-end SSH validation until a live Pod accepts the
key. The connected MCP `create_pod` tool accepts `sshPublicKey`; pass the
contents of the matching `.pub` file directly so the official RunPod image
authorizes it and exposes `22/tcp`. This avoids depending on account-level key
registration. If a different creation lane does not inject the key explicitly,
confirm that the exact public fingerprint is registered before Pod boot.

Do not loosen or replace the repaired private-key ACL automatically. If a later
live test reports `Permission denied (publickey)`, verify explicit injection or
registration of the fingerprint above before changing the Pod or key.

The definitive check is an approved disposable Pod with SSH enabled and a hard
termination deadline. After retrieving its current host and port, require this
non-interactive command to return exactly `RUNPOD_SSH_OK` with exit code zero:

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

#### Persistence and concurrency

Mount persistent storage at `/workspace`. Keep the execution checkout,
dependency and dataset caches, detached-process logs, ignored raw artifacts,
and checkpoints below that mount; container-only paths such as `/root` are
ephemeral. Pin the container image, dependency constraints, and clean Git SHA.
Long-running commands must survive SSH loss and write their log to
`/workspace`.

One multi-GPU Pod with one authoritative coordinator is the first supported
parallel shape. Do not run independent case runners or allow concurrent
writable checkouts on one network volume. Multi-Pod execution remains blocked
until `OPS-05` defines external exact-once claims, per-worker writable roots,
durable collection, and shared read-only caches, and `OPS-06` verifies them.

#### Approval, cost guard, and teardown

Before creating any billable resource, report the exact purpose, cloud tier,
GPU SKU and count, current hourly price, data center, image, volume and storage
price, maximum duration, maximum projected cost, automatic-termination
deadline, and cleanup intent. Obtain explicit approval for that resource
envelope. Infrastructure-smoke approval is separate from scientific-launch
approval.

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
