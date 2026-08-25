# A1 Learning-Rate Screen — Formal Review Packet

**Status:** formally approved at design commit
`54be534f383001b4af3d3b43597e135d4ca6653d`; the manifest reviews only
`A1-lr-screen`. Its three immutable configs are materialized under
`experiments/01-a1-lr-screen/`. The separately approved calibration is now
accepted in [`a1-calibration-packet.md`](a1-calibration-packet.md); definitive
pretraining and any new RunPod spending remain unapproved.

This packet summarizes the exact `A1-lr-screen` scope for review at one Git
commit. If any summary conflicts with a normative component listed in
[`../experiment_plan.md`](../experiment_plan.md), the normative component wins.

This revision changes A1 only from `full-pass-wrap` to `lr-400m`. It reduces
each A1 run from 5,691 to 1,526 updates (73.19%). No downstream group budget is
changed.

## Scientific Contract

| Item | Frozen A1 value |
| --- | --- |
| Question | Which peak learning rate should be used for later Pythia-14M comparisons? |
| Physical cases | Three A0 runs at peak LR `5e-4`, `1e-3`, and `2e-3`; seed 0 |
| Model | `EleutherAI/pythia-14m-deduped` config at `7386d9a4ae45aef494a6e704910394def3037fc5`; random initialization; released weights are not loaded |
| Architecture intervention | Topology A0; no threshold; activation pressure disabled with method `none` |
| Training-cache source | All 1,000,000 MiniPile training documents at `18ad1b0c701eaa0de03d3cecfdd769cbc70ffbd0`; cache token SHA-256 `da82a2ea2e0080c7fd681c7a93b07d3d9ff3d5357a8640895a82d536a1eaf97c` |
| Tokenizer | Pythia-14M tokenizer at the model revision; append EOS; 2,048-token blocks |
| Budget | `lr-400m`: 1,526 optimizer updates; 400,031,744 input tokens |
| Data order | First 195,328 blocks of the seed-0 complete-block permutation; no wrap; 533,046 complete blocks unused and the 1,464-token tail excluded; schedule SHA-256 `5feffe55fe37c764e86c6709500f1b0afad85be652de127f5fc7c958a7eb481c`; identical across all three cases |
| Batch | 262,144 input tokens/update = 128 sequences; A40 48GB physical batch 16 sequences × accumulation 8 |
| Optimization | AdamW, betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay `0.1`, global gradient clipping `1.0`, zero hidden/attention dropout |
| Precision | BF16 CUDA autocast with FP32 parameters and optimizer state |
| LR schedule | 16-update linear warmup to the case peak, then cosine decay to `0.1 × peak` at update 1,526 |
| Selection validation | Frozen selection cache token SHA-256 `22bb7c27864f0e5941548c572d6c75b1b5ba6a4c13e4cd26f40f4de546c5cc19`; all 152 complete sequences; batch 4; update 1, every 191 updates, and final = 9 evaluations |
| Checkpoint | Final model checkpoint only; no intermediate or optimizer-state checkpoint |
| Eligibility | Exact completed budget, finite final selection loss, complete required metrics and artifact envelope; infrastructure failures retry only unchanged, scientific failures remain grid cells and are not replaced |
| Decision | Among eligible cases, lowest final selection loss; exact tie selects lower LR; if none is eligible, stop downstream 14M stages for review |
| A1 report | Complete three-cell tuning table, training/validation curves, final selection loss, throughput/timing/resource telemetry, terminal classification, and selected LR; no sparsity claim and no full-pass, convergence, or horizon-independent optimum claim |

All three cases must record the same initial-parameter hash and schedule hash.
The immutable config includes `identity.training_implementation_id:
a1_pretraining_v1`, its canonical condition fingerprint, and both cache token
digests. The manifest also retains the complete config SHA-256 and exact Git,
package, device, and attempt provenance.

## Readiness Evidence

| Evidence | Result |
| --- | --- |
| Input/license and physical-batch freeze | Commit `c5aef6d`; three cache digests reverified locally on 2026-08-25 |
| Deterministic schedule, LR, validation, and confirmation evaluator | Commit `e7d3b68` |
| Catalog, exact A1 membership, fingerprints, manifests, and calibration-only concurrency | Commit `99c2d03` |
| Fixed 400M-token A1 contract | This packet's reviewed Git commit; exact budget, schedule, and validation-cadence tests |
| Automated tests after final A1 cache pins | 467 passed, 3 expected platform skips |
| Strict repository check | 0 errors, 0 warnings |
| Local infrastructure smoke | Completed at `experiments/00-infrastructure-smoke/raw/00-smoke/009-20260825-130526-1a34bf17/`; not scientific evidence |

Every A1 input, implementation, identity, physical-batch, and calibration
workboard item is resolved. The calibration is operational ETC evidence only,
not an A1 learning-rate result.

## Effect of Formal Approval

Formal approval must name the exact Git commit containing this packet and may
authorize only the following next step: change the manifest to `reviewed` for
`[A1-lr-screen]`, pin that design commit, then materialize and commit exactly
three immutable configs after fingerprint validation.

It does **not** authorize RunPod spending, definitive pretraining, concurrent
definitive runs, multiple runners, same-GPU packing, or multi-Pod execution.
The existing calibration-only authorization becomes actionable only after the
plan and configs are committed; any billable RunPod calibration still requires
separate approval of its exact resource and cost envelope.

## Recommended Post-Approval Sequence

1. Commit the reviewed manifest and the three fingerprinted A1 configs; rerun
   the full preflight while the plan components remain byte-identical to the
   approved design commit.
2. Query current RunPod capacity and price, present one two-A40 Pod envelope
   with an automatic termination deadline and maximum cost, and request
   separate spending approval.
3. On the approved Pod, run a 600-second solo calibration of the middle-LR
   config, then one coordinator over all three configs on two GPUs. The middle
   LR therefore has matched solo and concurrent timing; the concurrent run
   schedules two configs first and one after a slot frees.
4. Verify timing boundaries, GPU/runtime identities, cost, concurrency overlap,
   and artifact isolation; retrieve artifacts and terminate the Pod.
5. Re-estimate per-run and full-tranche ETC, completion time, and cost with
   uncertainty. Return for explicit definitive-launch approval. Definitive A1
   pretraining remains serial unless a later policy and plan review explicitly
   authorizes otherwise.

That later scheduling-only review is now recorded in
[`experiment_plan.md`](../experiment_plan.md) and
[`a1-calibration-packet.md`](a1-calibration-packet.md). It does not change this
packet's scientific contract or its reviewed-design identity.

The calibrated ETC must be based on the 1,526-update production schedule. No
ETC or cost estimate derived from the superseded 5,691-update horizon is valid
for launch approval.
