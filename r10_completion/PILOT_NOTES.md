# R10 pilot closure before confirmation

Pilot seeds 970101–970104 are development-only and excluded from confirmatory inference.

The preregistered thresholds were not changed after pilot observation. Two source-level methodological issues were corrected before the final freeze:

1. E5 episodic memory originally stored evaluator source truth. It now stores the agent's own inferred source and decision; evaluator labels are retained only for scoring.
2. E3 originally fitted the gate to the hidden switching-regime label. It now fits from factual observed prediction-gain labels on exploratory training actions; the hidden regime is evaluator-only during training and final policy execution.

The pilot was repeated after both corrections. Final development medians/counts:
- E1: 4/4 development seeds meet the frozen strong criterion; selected dimension median 4.
- E2: exact isomorphism 4/4; structural specificity 0/4; median return advantage +0.010086.
- E3: 4/4; median gate balanced accuracy 0.95894 and reward advantage over best constant +0.04924.
- E4: 0/4; queue transfer is strong but inertia transfer is adverse (median POLAR return 0.58154).
- E5: 0/4; median primary balanced accuracy 0.75661, meta-Brier gain 0.00843, and four lesion signatures; counterfactual gain is below threshold.

These pilot outcomes are not confirmatory results and are not used to modify the frozen criteria. The final scientific freeze is `bf34f2f7cac13127dd909ed5267b43123b15971dbc581721ddb077dd5e6380ed`.
