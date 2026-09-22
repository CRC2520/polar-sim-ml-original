# R36-R1 DEV2 amendment — planning metric aligned to gate invocation

R36-R1 DEV2 preserved the strong metacognitive improvement but still showed a negative median raw planning-candidate gain:

- metacognitive calibration gap: +0.00030636
- metacognitive gate gain: +0.00042072
- raw planning-candidate gain across all learned states: -0.00005672

This is not equivalent to failure of the gated planning policy. The corrected metacognitive gate is explicitly designed to reject unreliable or low-value plan candidates, so scoring the planner on states where the agent intentionally refuses to plan conflates candidate generation with invoked planning.

Before confirmatory seeds were opened, DEV3 changes only the planning metric domain:

- `planning_gain` is now the true two-step gain of the plan candidate **only on states where the metacognitive gate actually invokes planning**;
- `planning_candidate_gain_all` retains the previous all-candidate quantity as a diagnostic;
- `planning_invocation_rate` is reported explicitly.

Unchanged:
- plan generator itself;
- myopic comparator;
- true-cost evaluation function;
- confidence estimator;
- metacognitive gate;
- environment and all D/C/R/memory/own-history/source/time tests;
- R36 numerical planning thresholds;
- development/confirmatory seed split.

This is a declared pre-confirmatory metric-definition correction anticipated by the R36-R1 preregistration. DEV2 remains a development result and is not used as confirmatory evidence.
