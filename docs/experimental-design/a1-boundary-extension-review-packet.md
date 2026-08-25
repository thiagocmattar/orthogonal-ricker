# A1 Learning-Rate Screen — Boundary-Extension Review Packet

> **Status:** proposed design amendment; no current config-allocation, launch,
> or RunPod-spending authority.

## Why One More Cell

The original three A1 cells completed as eligible, valid runs under the same
400M-token contract:

| Peak LR | Final selection loss | Exact accepted run |
| ---: | ---: | --- |
| `5e-4` | 5.191815 | `001-a1-lr-5e-4` / `001-20260825-191155-6b7376de` |
| `1e-3` | 4.840335 | `002-a1-lr-1e-3` / `001-20260825-191154-b9299c46` |
| `2e-3` | 4.292333 | `003-a1-lr-2e-3` / `001-20260825-195141-f842c400` |

The best observed value was the upper tested boundary. This amendment adds one
factor-two point, `4e-3`, to determine whether the result continues to improve
or turns within the tested range. It does not add or imply an `8e-3` run.

## Exact Scientific Amendment

| Item | Reviewed value |
| --- | --- |
| Group and scaffold | `A1-lr-screen`; keep all recipes, raw attempts, and figures under `experiments/01-a1-lr-screen/` |
| Complete peak-LR grid | `{5e-4, 1e-3, 2e-3, 4e-3}` |
| New physical condition | One A0/no-pressure run at peak LR `4e-3`, seed 0 |
| Reused conditions | The exact accepted runs for configs `001`, `002`, and `003`; do not rerun them |
| Model and initialization | Pythia-14M architecture pinned in `protocol.md`, randomly initialized; released weights are not loaded |
| Budget and data order | `lr-400m`: 1,526 updates, 400,031,744 input tokens, frozen seed-0 schedule hash; identical to the completed cells |
| Batch and precision | 262,144 input tokens/update = microbatch 16 × accumulation 8 at sequence length 2,048; BF16 autocast with FP32 parameters/state |
| Optimization | Same AdamW recipe and 1%-warmup/cosine-decay schedule; only peak LR changes |
| Validation and checkpoint | Same nine complete selection evaluations and final-only model checkpoint |
| Implementation identity | `a1_pretraining_v1`; every active training, data, and measurement field remains unchanged |

The complete normative values and pins remain in `protocol.md`; this packet
changes only the 14M peak-LR grid, A1 cell count, and four-cell decision rule.

## Decision, Failure, and Claim Rules

After all four cells have a reviewed terminal classification, select the
eligible cell with the lowest final selection loss; an exact tie selects the
lower LR. A scientific failure remains a resolved, ineligible grid cell. An
infrastructure failure remains unresolved until an explicitly approved retry
of the unchanged config succeeds or the tranche is stopped. If no cell is
eligible, stop downstream 14M stages for review.

Report the complete four-cell table and updated LR-versus-final-validation-loss
figure. A1 selects the best tested LR for Pythia-14M at this exact 400M-token
horizon. It makes no sparsity, convergence, full-pass, or horizon-independent
optimality claim. If `4e-3` wins, state explicitly that it is still the upper
tested boundary; do not extend the grid automatically.

## Post-Review Materialization and Execution Shape

After exact-SHA design approval, append the immutable config
`experiments/01-a1-lr-screen/run/004-a1-lr-4e-3.yaml` and list all four configs
in that scaffold's runner. Remove the historical parallel authorization, which
is bound to exactly the original three config IDs. Run the amended case runner
serially under one coordinator and lock: it must reuse the three completed
attempts and execute only config `004`.

Before launch, record the unchanged-active-training-path check required by
`run-reuse.md`. Stage the three exact accepted raw attempts and the frozen
MiniPile cache on the execution host; without those raw attempts, the runner
must stop rather than rerun the original cells.

Accepted same-hardware evidence gives prior end-to-end run times of 39m40s,
39m45s, and 40m15s. The incremental runner ETC is therefore about 40 minutes
(38–44 minutes planning range; 47 minutes conservative). One Secure A40 48GB
is sufficient; a second GPU cannot shorten this single pending run under the
reviewed single-device training implementation. No new calibration is needed
unless the hardware or runtime identity changes. Live capacity, price, and the
billable ceiling must still be checked immediately before provisioning.
Using the historical $0.44/A40-hour plus $0.011/hour storage rate only as a
planning basis, the expected 61–62 minute Pod lifecycle is about $0.46 and a
90-minute envelope is about $0.68. These are not a live quote or spending cap.

## Effect of Formal Approval

Formal approval must name the exact 40-character Git SHA containing this
packet and its normative amendments. It authorizes only the next design step:
set `docs/experiment_plan.md` to `reviewed` for `[A1-lr-screen]`, pin that
design SHA, and materialize and commit the single new config plus the serial
four-config runner in the existing scaffold.

It does **not** authorize scientific execution, RunPod provisioning or
spending, transfer, retry/replacement, or teardown. After materialization and
verification, definitive launch requires approval at the exact execution SHA;
any billable RunPod work separately requires an exact resource, cost, and
termination envelope.
