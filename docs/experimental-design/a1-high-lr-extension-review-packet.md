# A1 Three-Cell High-LR Extension Review Packet

> **Status:** reviewed by direct user instruction on 2026-08-26 at exact
> design commit `d80f6a9b6c99bcaec7ddc52e73c1a407a5020a8e`; activated for
> `[A1-lr-screen]` in `docs/experiment_plan.md`.

## Why These Cells Exist

The completed five-cell screen continued to improve at its upper boundary:
`8e-3` reached final selection loss `4.1769665165951375`, versus
`4.224078962677403` at `4e-3`. The user directed three additional values above
`8e-3`. Continuing the established factor-two progression gives exactly
`1.6e-2`, `3.2e-2`, and `6.4e-2`.

## Exact Scientific Delta

The complete A1 grid becomes
`{5e-4, 1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2, 3.2e-2, 6.4e-2}`. Materialize exactly
three immutable configs:

- `006-a1-lr-1p6e-2`;
- `007-a1-lr-3p2e-2`;
- `008-a1-lr-6p4e-2`.

Relative to config `005`, each new config changes only its experiment label,
derived condition fingerprint, and peak learning rate. Reuse the accepted
attempts for configs `001`–`005` exactly.

Every new cell retains seed 0, random Pythia-14M initialization, topology
`A0`, no activation pressure, the frozen MiniPile caches and data order, 1,526
updates / 400,031,744 tokens, 262,144 tokens per update, microbatch 16 with
accumulation 8, BF16 autocast, the fixed AdamW schedule, nine full-coverage
selection evaluations, and one final model checkpoint.

## Decision and Stop Rules

After all eight cells have reviewed terminal classifications, select the
eligible cell with the lowest final selection loss; an exact tie favors the
lower learning rate. Scientific divergence is a resolved ineligible cell and
is never retried. Infrastructure interruption requires explicit recovery of
the unchanged config. No further automatic learning-rate extension is
authorized.

Report the complete eight-cell table and curve. If `6.4e-2` wins, label it as
the upper tested boundary and make no convergence, full-pass, global-optimum,
or horizon-independent optimality claim.

## Execution Shape

Use one Secure RunPod machine with exactly three distinct homogeneous NVIDIA
A40 48GB GPUs, one coordinator, one repository lock, and one isolated worker
per pending config. Require exact completed reuse of configs `001`–`005`
before admitting configs `006`–`008`. Because all three pending cells are
admitted at launch, a scientific failure in one cell does not suppress the
other two; the coordinator drains every admitted worker before returning.

Multiple runners, same-GPU packing, heterogeneous GPUs, serial fail-stop, and
multi-Pod execution are forbidden for this tranche. The accepted config `005`
create-to-retrieval duration was approximately 57 minutes and its training
phase was approximately 40 minutes. Use these same-hardware observations for
the launch ETC and cost envelope; verify the current RunPod price and capacity
immediately before provisioning.
