# A1 `4e-3` Boundary Cell — Definitive Launch Review Packet

> **Status:** prepared for exact-SHA scientific launch review. This packet does
> not authorize RunPod provisioning, spending, transfer, or execution.

## Exact Recipe and Authority

| Item | Value |
| --- | --- |
| Approved design | `e8214a411afebf0cec5658f0f1ccdd3e6bcd5585` |
| Reviewed manifest | `[A1-lr-screen]`, activated at `ab4c5b12449316ca1b92caebe4793af7d9bf1bcd` |
| Materialized recipe | `e7e63a4ae7f7def56d344b69adc426636dd7e0fb` |
| Runner | `experiments/01-a1-lr-screen/run/runner.py`; ordered configs `001`–`004` |
| Execution shape | Serial: one coordinator, one lock, one A40, no worker slots or parallel authorization |
| Only pending condition | `004-a1-lr-4e-3`, seed 0, 1,526 updates / 400,031,744 input tokens |
| Config identity | Fingerprint `7742e7219fb40ee55adc4a42d87c00de6790eb7a5b3f5ff9643f85a137b9dd01`; complete-config SHA-256 `701afdbdcade83bdd878a30b65683825fb27c15038e24e4f6426f30658f1680d` |

Config `004` differs from config `003` only in peak LR and the corresponding
label/fingerprint. Configs `001`–`003` are byte-identical to their accepted
recipes. The cross-revision active-path proof is recorded under `OPS-10` in
[`workboard.md`](workboard.md#a1-boundary-extension-compatibility-evidence).

## Required Reuse and Pending Proof

| Required config | Exact accepted run |
| --- | --- |
| `001-a1-lr-5e-4` | `001-20260825-191155-6b7376de` |
| `002-a1-lr-1e-3` | `001-20260825-191154-b9299c46` |
| `003-a1-lr-2e-3` | `001-20260825-195141-f842c400` |

The runner binds these IDs as `required_completed_config_ids` and checks them
before and after acquiring the launch lock. Missing, failed, ambiguous,
statusless, or mismatched evidence stops before mutation, even with
`--retry-failed`; it cannot admit an old config as pending.

The local artifact classification is:

```text
completed/reused: 001-a1-lr-5e-4, 002-a1-lr-1e-3, 003-a1-lr-2e-3
pending: 004-a1-lr-4e-3
```

The accepted three-run archive is
`a1-definitive-artifacts-276da7c.tar.gz`, 156,676,741 bytes, SHA-256
`acfb77783792dd6785c6cd81f4b0b0ff4f599c336a15660ff5e2a154ea7bc4af`;
its canonical acceptance digest is
`269f3ac7b59a947d3a1c7cf4d3cc5f63e806d0728654d83feb54442941270722`.
On the execution host, extract only its accepted raw attempts into the current
scaffold, then require the same classification before launch.

## ETC and Runtime Boundary

Prior exact-hardware total wall times were 2,379.7, 2,384.5, and 2,415.3
seconds. For the one pending cell:

| Scope | Point ETC | Planning range | Conservative bound |
| --- | ---: | ---: | ---: |
| First completion | 40m15s | 38–44m | 47m |
| Incremental full tranche | 40m15s | 38–44m | 47m |
| Whole Pod lifecycle | 61–62m | — | 90m guard |

No new calibration is required only if preflight observes one NVIDIA A40 48GB
and a materially matching runtime: Python 3.12.3, Torch `2.11.0+cu128`, CUDA
12.8, Transformers 5.12.1, the pinned image/dependencies, and a compatible
driver (prior: 570.195.03). A new GPU UUID is expected; a different GPU class
or material runtime identity stops launch and requires a refreshed
calibration/ETC review. Prior peak reserved memory was 29.5GB.

Historical pricing gives a planning estimate only: $0.451/hour including
storage, about $0.46 expected and $0.68 for 90 minutes. Live capacity, price,
balance, projected completion timestamps, provision cutoff, total cap, and
absolute termination deadline must be reported and approved separately.

## Terminal and Recovery Rules

- Exact-budget completion with finite required outputs is eligible.
- Scientific divergence/nonfinite values are preserved and classified as a
  scientific failure; no retry or replacement occurs.
- Infrastructure failure is preserved, retrieved, and classified; recovery
  requires separate approval for the unchanged config and `--retry-failed`.
- Running, statusless, inconsistent, or ambiguous state is monitored only.
- No automatic retry, replacement Pod, new config, or `8e-3` extension.

After completion, retrieve and verify config `004`, rerun the four-cell
classification and selection rule, regenerate the four-point figure under the
same scaffold, and only then delete the Pod under an approved teardown clause.

## Approval Boundary and Next Step

Scientific launch approval must name the exact Git SHA containing this packet.
That approval authorizes the definitive config `004` run under this recipe but
does not authorize billable resources or transfers. After scientific approval,
perform a read-only live RunPod query and return with one exact Secure A40 Pod
envelope, clean-bundle/cache/reuse-archive hashes, exclusions, cost ceiling,
cutoff, deadline, retrieval, and teardown terms for separate approval.
