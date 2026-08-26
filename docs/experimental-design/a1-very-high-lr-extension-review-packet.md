# A1 Three-Cell Very-High-LR Extension Review Packet

> **Status:** reviewed by direct user instruction on 2026-08-26 at exact design
> commit `2320d542b14926315a17e873afac2d41a40d6814`. This approval permits
> activation and config allocation only; it does not authorize RunPod
> provisioning, spending, transfer, or a scientific launch.

## Question and Evidence

The A1 question remains: which tested peak learning rate minimizes final
selection loss for randomly initialized Pythia-14M after exactly 400,031,744
training tokens? The completed upper cells continued to improve:

| Peak LR | Final selection loss |
| ---: | ---: |
| `1.6e-2` | `4.112285005418878` |
| `3.2e-2` | `4.082745991255107` |
| `6.4e-2` | `4.0587728086270785` |

Because the lowest observed loss is still at the upper boundary, the user
directed three larger values. Continuing the established factor-two spacing
gives exactly `1.28e-1`, `2.56e-1`, and `5.12e-1`.

## Exact Scientific Delta

The complete A1 grid becomes
`{5e-4, 1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2, 3.2e-2, 6.4e-2, 1.28e-1, 2.56e-1, 5.12e-1}`.
After review, allocate exactly these immutable configs:

- `009-a1-lr-1p28e-1`;
- `010-a1-lr-2p56e-1`;
- `011-a1-lr-5p12e-1`.

Relative to config `008`, each changes only its experiment label, derived
condition fingerprint, and peak learning rate. Reuse the exact accepted
attempts for configs `001`–`008`; they must never be rerun.

Every new cell retains seed 0, random Pythia-14M initialization, topology
`A0`, no activation pressure, the frozen MiniPile caches and data order, 1,526
updates / 400,031,744 tokens, 262,144 tokens per update, microbatch 16 with
accumulation 8, BF16 autocast, the fixed AdamW schedule, nine full-coverage
selection evaluations, and one final model checkpoint.

## Decision and Failure Rules

After all eleven cells have reviewed terminal classifications, select the
eligible cell with the lowest final selection loss; an exact tie favors the
lower learning rate. A finite but worse point remains eligible and reported.
Scientific divergence or a nonfinite task value is a resolved ineligible cell
and is never retried. An infrastructure interruption preserves its attempt and
requires explicit recovery of the unchanged config.

The definitive runner orders the new cells from lower to higher LR and retains
the repository rule that the first escaping failure stops later admission. A
failure before config `011` therefore stops the tranche for review rather than
silently classifying unattempted cells. No replacement seed, automatic retry,
or further automatic LR extension is permitted.

## Required Outputs and Claim Boundary

Update the A1 table, curve, and deterministic provenance to all eleven exact
config/run identities. Report terminal classification and eligibility for
every planned cell, including adverse or failed outcomes. If `5.12e-1` wins,
label it as the upper tested boundary.

A1 selects only the best tested LR at this fixed 400M-token horizon. It does
not establish convergence, a full-pass result, a global optimum, a
horizon-independent optimum, or any sparsity claim.

## Review and Launch Boundary

Approval of this packet at its exact committed design SHA permits activating
only `[A1-lr-screen]` and then materializing configs `009`–`011`. It does not
itself authorize cloud spending or launch. After materialization, the launch
packet must pin the clean execution SHA, complete-config hashes, required reuse
of configs `001`–`008`, transferred artifact identities, current RunPod price
and capacity, first-run/full-tranche ETC, cost cap, termination deadline,
failure policy, artifact retrieval, and teardown.
