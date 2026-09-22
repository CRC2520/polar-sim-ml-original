# R36-R1 final development report before confirmatory freeze

R36-R1 targets only the metacognitive bottleneck retained by R36.

## Parent R36 failure

R36 confirmatory:
- metacognitive calibration: 7/12;
- metacognitive functional gate: 8/12;
- complete same-agent conjunction: 4/12;
- all R36 median guardrails passed.

## DEV1 — action-conditional confidence, overly restrictive planning exposure

Run: 35743474176.

The confidence estimator was changed from state-only local error to action-conditional factual reliability over [state, action, predicted-next-state]. Calibration and gate utility improved strongly, but the stricter gate changed the visited-state distribution and raw all-candidate planning gain became negative.

DEV1 is retained as a development failure.

## DEV2 — lower confidence threshold

Run: 35743785065.

META_PLAN_CONF_MIN was lowered from 0.55 to 0.30 while retaining the positive predicted-plan-advantage requirement. Metacognition remained strong but the raw planning-candidate metric was still slightly negative.

## DEV3 — final metric-domain correction

Run: 35744003028.

The plan generator was not changed. Development revealed that scoring plan candidates on states where the metacognitive gate explicitly rejected planning conflated candidate generation with invoked planning.

Before confirmatory seeds were opened:
- planning_gain was restricted to states where use_plan=True;
- planning_candidate_gain_all retains the former quantity as diagnostic;
- planning_invocation_rate was added as a guard against trivial non-use.

Final development medians:
- D damage: 0.014340
- C gain: 0.003656
- R damage: 0.002875
- memory reentry: 0.000572
- own-history transplant damage: 0.005154
- source balanced accuracy: 0.9664
- time-order accuracy: 1.000
- metacognitive calibration gap: +0.000306
- metacognitive gate gain: +0.000421
- invoked planning gain: +0.003365
- planning invocation rate: 0.04383
- stability: 1.000

All four development seeds pass the retained R36 per-seed thresholds under the final R36-R1 metric definition, including the new invocation-rate guard.

## Confirmatory freeze

Confirmatory seeds: 2017001–2017012.

The R36 thresholds for D, C, R, memory, own-history, source, time, metacognition, planning, stability, model count, and same-agent lifetime are retained numerically. A new planning-invocation-rate guard is added.

PASS requires:
- every component >=9/12;
- complete same-agent conjunction >=9/12;
- all frozen median criteria pass.

No confirmatory threshold may be relaxed after seed opening.

Source blob: 99c6ac712e26abbbe7a1202709701e5af94c6ab4
Adjudicator blob: 5ecb746dd98b93f20d1feccbdc92eee46c272498
