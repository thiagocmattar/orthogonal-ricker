# A1 `8e-3` Boundary Extension Review Packet

> **Status:** directed by the user on 2026-08-26; awaiting exact-design-SHA
> activation in `docs/experiment_plan.md` before config allocation.

## Why This Cell Exists

The completed four-cell screen continued to improve at its upper boundary:
`4e-3` reached final selection loss `4.224078962677403`, versus
`4.292332950391267` at `2e-3`. The user directed one additional value with the
same factor-two spacing. The resulting peak learning rate is exactly `8e-3`.

## Exact Scientific Delta

The complete A1 grid becomes `{5e-4, 1e-3, 2e-3, 4e-3, 8e-3}`. Materialize
one new immutable config, `005-a1-lr-8e-3`, and change only its experiment
label, derived condition fingerprint, and peak learning rate relative to
config `004`. Reuse the accepted attempts for configs `001`–`004` exactly.

The new cell retains seed 0, random Pythia-14M initialization, topology `A0`,
no activation pressure, the frozen MiniPile caches and data order, 1,526
updates / 400,031,744 tokens, 262,144 tokens per update, microbatch 16 with
accumulation 8, BF16 autocast, the fixed AdamW schedule, nine full-coverage
selection evaluations, and one final model checkpoint.

## Decision and Stop Rules

After all five cells have terminal classifications, select the eligible cell
with the lowest final selection loss; an exact tie favors the lower learning
rate. Scientific divergence is retained as a resolved dominated cell and is
not retried. Infrastructure interruption requires explicit unchanged-config
recovery. No automatic `1.6e-2` extension is authorized.

Report the complete five-cell table and curve. If `8e-3` wins, label it as the
upper tested boundary and make no horizon-independent optimality claim.

## Execution Shape

Use the existing serial A1 runner under one coordinator and lock. Require exact
completed reuse of configs `001`–`004`, then run only config `005` on one A40
48GB. Retrieve and verify the complete artifact envelope before deleting the
Pod. The prior measured end-to-end duration for config `004` was 41m44.9s; use
a conservative 55–65 minute provisioning-to-retrieval envelope.
