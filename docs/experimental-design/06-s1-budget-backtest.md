# 06. S1 Budget Backtest and Launch Gate

## Status

The predeclared rank-survival check failed for global selection at approximately
2,048 steps. No S1 scientific cell should launch until the campaign records one
of the budget decisions below. E0 engineering pilots may proceed because they
are not ranked by validation loss.

## Evidence

Historical full-pass runs log validation at step 2,000 rather than 2,048. They
share the full 692,224-token validation cache, seed 0, 65,536 tokens/update,
LR `3e-5`, warmup 100, and a 22,762-step endpoint.

The strict nine-run fixed-architecture pressure cohort gives:

- Spearman rank correlation `rho = 0.7500`;
- Kendall `tau = 0.5556`;
- top-three membership overlap `3/3`;
- top-five membership overlap `4/5`.

The broader 17-run architecture/method stress test gives:

- Spearman `rho = 0.2059` and Kendall `tau = 0.1912`;
- top-one overlap `0/1`, top-three `1/3`, top-five `1/5`, top-eight `5/8`;
- One-ReLU L1N changes rank `14 -> 1`; One-ReLU RN changes `15 -> 4`;
- rank correlation versus the final endpoint rises from `0.559` at 3,000 steps
  to `0.824` at 4,000, `0.900` at 5,000, and `0.975` at 8,000.

| Cohort member | Validation at 2,000 | Final validation | Rank change |
| --- | ---: | ---: | ---: |
| Stock AdamW, config 50 | 7.081452 | 4.831727 | 12 -> 5 |
| One-ReLU AdamW, 77 | 7.015604 | 4.840446 | 1 -> 6 |
| One-ReLU RN, 78 | 7.168204 | 4.813355 | 15 -> 4 |
| One-ReLU OR, 79 | 7.060251 | 4.808733 | 7 -> 3 |
| One-ReLU L1N, 80 | 7.167561 | 4.796697 | 14 -> 1 |
| One-ReLU OL1, 81 | 7.027924 | 4.798111 | 2 -> 2 |
| Three-ReLU AdamW, 98 | 7.036153 | 4.956438 | 3 -> 7 |
| Three-ReLU RN, 105 | 7.258286 | 5.026340 | 17 -> 10 |
| Three-ReLU OR, 103 | 7.071855 | 5.009055 | 11 -> 9 |
| Three-ReLU L1N, 104 | 7.214872 | 5.188236 | 16 -> 15 |
| Three-ReLU OL1, 99 | 7.052708 | 5.170647 | 5 -> 13 |
| Six-ReLU PRE AdamW, 107 | 7.045142 | 4.972047 | 4 -> 8 |
| Six-ReLU PRE OR, 108 | 7.068353 | 5.457653 | 10 -> 16 |
| Six-ReLU PRE OL1, 109 | 7.083382 | 5.496211 | 13 -> 17 |
| Six-ReLU POST AdamW, 110 | 7.059338 | 5.099146 | 6 -> 11 |
| Six-ReLU POST OR, 111 | 7.062046 | 5.135765 | 8 -> 12 |
| Six-ReLU POST OL1, 112 | 7.064821 | 5.173141 | 9 -> 14 |

The exploratory within-architecture validation-loss/logged-zero Pareto sets had
full final-frontier recall, but this is not sufficient validation: logged zeros
cover only the final 8,192-token training microbatch, site definitions differ
by architecture, and no step-2,000 checkpoint exists for full-validation
`R_block`, `R_model`, or `U_arch`.

## Decision Required Before S1

Choose and register exactly one:

1. Keep 2,048 steps as a feasibility/collapse and within-stratum screen only.
   Never use a global rank cutoff; preserve AdamW and matched RN/OR and
   L1N/OL1 pairs for every viable family into the 8,192-step rung.
2. Raise S1 to 4,096 or 5,120 steps and recompute every config count, token
   budget, runtime, and promotion rule before launch. Historical rank
   correlations at nearby logged steps are `0.824` and `0.900`.
3. Merge discovery with the 8,192-step rung. This gives the strongest historical
   rank proxy (`rho = 0.975`) but approximately quadruples the planned S1
   training work.

The historical analysis is one-seed and spans multiple repository commits.
Its purpose is to veto unsupported selection behavior, not to estimate final
paper uncertainty.

## Pinned Historical Runs

The strict pressure cohort uses configs `50--54` and `56--59`. The architecture
stress cohort adds the pinned complete runs for configs `77--81`, `98--99`,
`103--105`, and `107--112`. Exact paths remain in `docs/experiment_log.md` and
the Report 04/05 pinned cohorts; do not replace them with "latest completed"
runs when repeating this check.
