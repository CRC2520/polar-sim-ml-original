# Study 2: disclosed post-run decision adjudication

Adjudicated at 2026-09-06T02:32:51.882708+00:00.

Frozen evaluator output: **pivot_to_conditional_structural_prior**.

Literal-protocol adjudication: **suspend_exclusive_polar_advantage_claim**.

Frozen study2/evaluation.py selects pivot whenever aligned practical benefit is present after broader criteria fail; the frozen protocol says ONLY aligned regimes. The code omitted the exclusion of practical benefit in diagonal and misaligned regimes.

Broader criteria fail and the aligned-only condition is false; the protocol's otherwise clause applies.

| Regime | Upper 95% interval bound | Practical benefit |
|---|---:|---|
| paired | -0.003813979 | True |
| diagonal | -0.001467560 | False |
| misaligned | -0.002376336 | True |

The practical-benefit rule is upper interval bound < -0.002000. Broader continuation criteria: False. Aligned-only condition: False.

No numeric result was changed. The raw decision and frozen evaluator remain available with their original label so that regeneration reproduces the historical output. This document corrects the subsequent interpretation transparently; it is not a newly registered hypothesis test. The publication's decision should use the adjudicated label and disclose the discrepancy.

This decision concerns the exclusive polar-advantage claim. It does not negate the measured benefit of including learned coupling relative to its lesion or prohibit a new, separately specified study of useful generic coupling.

Reproduce with `python scripts/adjudicate_study2.py`. The JSON records exact input hashes, the frozen rule, every decision predicate, unchanged supporting estimates, sample-size qualification, and adjudicator source hash.
