# R36-R1 DEV1 amendment — planning exposure regression

R36-R1 DEV1 improved the metacognitive calibration/gating target but regressed the independent planning-capacity metric.

DEV1 medians:
- metacognitive calibration gap: +0.00032250
- metacognitive gate gain: +0.00021945
- planning gain: -0.00023858

Per-seed inspection showed the stabilized gate correctly avoided plan candidates that were factually worse than the myopic action, so the functional gate gain was positive. However the reliability threshold was too restrictive, reducing planning exposure across the lifetime and changing the visited-state distribution enough to make the independent raw planning candidate worse.

Before confirmatory seeds were opened, DEV2 changes only:
- META_PLAN_CONF_MIN: 0.55 -> 0.30

Unchanged:
- action-conditional factual-error estimator;
- empirical confidence percentile;
- positive predicted-plan-advantage requirement;
- environment;
- D/C/R lesions;
- memory, own-history, source/time tests;
- planning algorithm itself;
- development/confirmatory seed split;
- requirement that planning and all non-metacognitive R36 components retain their frozen thresholds.

DEV1 remains a development failure and is not evidence for the final R36-R1 hypothesis.
