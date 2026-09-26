# POLAR P2-Critical v2 — frozen confirmatory protocol

This v2 exists solely because v1 failed while serializing a NumPy integer after the
confirmatory computations had already been performed. No v1 seed metrics were persisted,
printed or inspected. Seeds 995001–995012 are permanently retired.

Scientific experiment engine: unchanged from `155adc84b56dfc8465b3c914883f1440b92247da`.
Thresholds: unchanged from P2-Critical v1.
New confirmatory seeds: **996001–996012**.
Global criterion per gap: **>=9/12** complete seed-level conjunctions.

## E1 — POLAR vs GENERIC isomorphic relational learner
Identical data, training, base terms, eight interaction slots and ridge fit.
A seed resolves the gap if both models recover >=6/8 true relations, both beat
a random eight-relation control by >=0.10 MSE, and either POLAR beats GENERIC
by >=0.10 MSE or they are practically equivalent within |ΔMSE|<=0.02.
Equivalence is interpreted as: a generic learner can match POLAR when it
reconstructs the same effective relational organization.

## E2 — integrated relational discovery
The agent receives only observation, own action and factual consequence.
PASS requires >=2/3 relation recovery before and after the switch, a changed
relation set, >=0.05 post-switch reward gain over frozen-relations control,
and >=0.05 gain over no-relation lesion.

## E3 — theory-discriminating functional lesions
Full guardrails: primary >=0.70, workspace >=0.70, source BA >=0.55.
Relational lesion: primary damage <=0.03, workspace damage >=0.10,
meta-Brier damage >=0.02, source damage >=0.15.
Broadcast lesion: primary damage >=0.10.
Higher-order lesion: primary <=0.03, workspace <=0.03, meta-Brier damage >=0.05.
The three signatures must satisfy the frozen ordering/uniqueness check.
This is a functional discrimination test, not evidence of phenomenal consciousness.

## E4 — standard-equation OOD
Logistic harvesting, RC thermal and queue-service tasks use the same generic
quadratic outcome learner after a fixed 96-step own-action identification prefix.
Each task requires oracle alive >=0.95, agent alive >=0.90, and agent/oracle
return ratio >=0.75. This is standard-equation OOD validation, not independent-team replication.
