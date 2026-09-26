# POLAR P2-Critical — frozen confirmatory protocol

Date: 2026-09-20/21 UTC boundary. Development seeds 991001–991004 are excluded from confirmation.
Scientific engine fixed at commit `155adc84b56dfc8465b3c914883f1440b92247da`.
Confirmatory seeds: **995001–995012**. Seed is the inferential unit.
Each experiment passes globally at **>=9/12** complete seed-level conjunctions.

## E1 — POLAR vs GENERIC isomorphic relational learner
Both learners receive identical 16-dimensional data, the same train/test split, base linear terms,
exactly eight interaction slots and identical ridge fitting. POLAR requires a disjoint matching;
GENERIC may choose any eight residual interactions. RANDOM has eight random disjoint pairs.

A seed resolves the gap if:
- POLAR and GENERIC each recover >=6/8 true relations;
- each beats RANDOM by >=0.10 MSE; and
- either POLAR beats GENERIC by >=0.10 MSE (**POLAR-specific**) or
  |POLAR-GENERIC| <=0.02 MSE (**practical equivalence after reconstructing the same relations**).

Thus the protocol can resolve the specificity question negatively; it does not require POLAR superiority.

## E2 — relational discovery inside one integrated closed-loop agent
A single agent receives only current observations, own action and factual consequence. It maintains
associative memory, prediction error/uncertainty, a local/cross model and a relation set learned from
its own sliding factual buffer. Hidden pair labels are evaluator-only. The true relation set switches
at step 320.

Per-seed PASS:
- >=2/3 true relations recovered before the switch;
- >=2/3 recovered after the switch;
- learned relation set changes;
- post-switch reward advantage over a controller whose relation set is frozen before the switch >=0.05;
- post-switch reward advantage over a no-relation lesion >=0.05.

## E3 — theory-discriminating functional lesions
The benchmark contains separable local first-order processing, global broadcast/workspace,
higher-order confidence, and a relational/coherence pathway. It tests functional lesion signatures,
not phenomenal consciousness and not literal neuroscience theories.

Full system guardrails:
- primary accuracy >=0.70;
- workspace recall >=0.70;
- source balanced accuracy >=0.55.

POLAR/relational lesion:
- primary accuracy damage <=0.03;
- workspace damage >=0.10;
- metacognitive Brier damage >=0.02;
- source-attribution damage >=0.15.

Broadcast/workspace lesion:
- primary accuracy damage >=0.10.

Higher-order confidence lesion:
- primary damage <=0.03;
- workspace damage <=0.03;
- metacognitive Brier damage >=0.05.

The three lesion signatures must remain ordered/distinct by the implementation's frozen uniqueness check.

## E4 — standard-equation OOD validation
Three target families are defined from standard mathematical forms rather than POLAR-specific latent
variables: logistic harvesting, first-order RC thermal dynamics and queue-service dynamics.
The same quadratic outcome learner receives three normalized observations and four actions in each task.
A fixed 96-step own-action identification prefix is allowed; scoring is only over the remaining window.

For every task in a seed:
- oracle alive fraction >=0.95;
- agent alive fraction >=0.90;
- agent/oracle mean-return ratio >=0.75.

This is OOD validation on independently specified equation families, **not an independent-team replication**.

## Interpretation
PASS/FAIL applies only to these frozen operational criteria. E3 can discriminate computational lesion
signatures but cannot establish subjective experience. E4 cannot substitute for independent replication
by another team or implementation.
