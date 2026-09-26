# POLAR R10 — Five discriminating experiments

Frozen design branch: `research/r10-discriminating-five-tests-20260920`.
Base implementation: merged R9 main at `a0696148e568c94365a26311953430b082a3725b`.

## General rules
- Pilot seeds: 961001–961003. They may select only the CSD state subset; no final threshold changes.
- Confirmatory seeds: 963001–963012 (n=12).
- The seed is the inferential unit; within-trajectory steps are not independent replicates.
- R9 source is imported unchanged. New code creates bounded experimental wrappers and new held-out dynamics.
- All adverse outcomes, deaths, selected-off gates and failed criteria are retained.
- A PASS means the frozen operational criterion passed in this benchmark. It does not establish consciousness or a universal polar principle.

## E1 — Causal-State Discovery (CSD)
Dynamic state blocks: HIST (16-step own history + previous action), PRED (previous observation/pole prediction/tension), ERR (recent prediction-error history), MEM (associative workspace memory/pending/recalled history), GOAL (workspace goal state).
Pilot enumerates all 32 subsets and selects the smallest subset that passes all pilot seeds; ties use fixed lexical order. If none passes, FULL is selected.

For each candidate, use the same trained R9 full agent and held-out ecology-delay9 tape. After every factual transition, omitted state blocks are reset to canonical observation-conditioned defaults before the next decision. The selected subset is then frozen for final seeds.

Per final seed PASS requires:
1. same-observation shadow action agreement >= 0.90,
2. memory forecast MAE <= 0.08,
3. uncertainty MAE <= 0.05,
4. gate-state agreement >= 0.90,
5. closed-loop return loss <= 0.03,
6. closed-loop alive-fraction loss <= 0.02.
Global PASS: >= 9/12 final seeds.

## E2 — POLAR-vs-GENERIC Isomorphic Control
Generic agent receives exactly the same 48 real-valued features, model dimensions, penalties, Q/workspace, acquisition budget and selection budget, but a deterministic fixed permutation of the 48 predictive coordinates is applied before every predictor call. The same local/cross mask positions therefore no longer correspond to the declared polar grouping. No information is removed and parameter counts are identical.

Both agents acquire their own trajectories under matched exogenous tapes. Per seed PASS requires FULL-GENERIC reward >= 0.01 and alive difference >= -0.005 in both ecology_train and ecology_delay9. Global superiority PASS: >= 9/12. Failure does not establish equivalence unless the descriptive confidence interval also supports a prespecified +/-0.01 equivalence margin.

## E3 — Adaptive-Gate Necessity
A mechanism-specific synthetic challenge supplies observable six-dimensional context. In half-spaces determined by context, a cross-route correction is beneficial; in the complementary region it is harmful. The gate uses the same ridge-on-predictive-gain logic and exact-off binary decision class as R9.

Per seed PASS requires:
- held-out gate classification accuracy >= 0.85,
- MSE improvement over the better of always-off/always-on >= 0.02,
- context permutation worsens MSE by >= 0.02.
Global PASS: >= 9/12.

## E4 — Cross-Domain Transfer
No model, Q, calibration or gate parameters are fitted on the two new domains.
Two independently coded dynamics share only the public three-field observation/action interface:
A. cyclic_buffer: oscillatory replenishment, nonlinear throughput cost and delayed deficit;
B. repair_queue: stochastic repair pressure, queue/resource coupling and nonlinear damage.
The R9 full and dense agents are trained only on ecology_train.

Per seed PASS requires on BOTH domains:
- FULL alive fraction >= 0.80,
- FULL normalized return >= 0.35,
- FULL-DENSE return >= -0.01,
- FULL-DENSE alive fraction >= -0.02.
Global PASS: >= 9/12.

## E5 — Integrated Consciousness Precursors
A single frozen R9 full agent traverses one 640-step attribution environment. The same trajectory is used to measure eight operational precursors from agent-accessible state:
1. self/world binary attribution,
2. three-way source attribution (self/world/mixed),
3. time-since-world-event bins,
4. workspace/content utility,
5. metacognitive error discrimination,
6. episodic-memory utility,
7. counterfactual action-ranking accuracy,
8. polar/cross-route utility.

Linear ridge probes are trained on the first half only and tested on the second half. Evaluator-only source/time labels never enter the native policy. Counterfactual outcomes are obtained from copied evaluator environment states and never fed to the policy.

Per seed base conjunction:
- self/world balanced accuracy >= .70,
- source balanced accuracy >= .55,
- time-bin balanced accuracy >= .45,
- uncertainty-vs-error AUC >= .65,
- counterfactual best-action accuracy >= .45.
Lesion conjunction on matched tapes:
- noMemory worsens memory forecast MAE >= .02,
- permuted_content lowers return >= .01,
- noCross lowers return >= .01,
while every lesion changes alive fraction by no more than .10 (to exclude indiscriminate collapse).
Global PASS: base + lesion conjunction in >= 9/12.

These are functional precursor tests only. They neither measure nor infer phenomenal experience.
