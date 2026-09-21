# E1 PREREG — Emergent Reconsolidation

Freeze: 2026-09-21 America/Lima.

Development seeds used and excluded: 1301001–1301803 and all intermediate development variants.
Untouched confirmatory seeds: 1312001–1312012.

Architecture: one GRU (hidden=128) plus ordinary action and factual-prediction heads. There is no explicit latent store, reconsolidation gate, memory-specific update rule, salience variable, or trauma variable.

Task: an outcome association is acquired, followed by a long neutral interval. Four conditions differ only in whether the target cue is reactivated before a subsequent factual outcome and whether that outcome contradicts the acquired association:
1. target reactivation + contradiction;
2. no target reactivation + same contradictory scalar outcome;
3. target reactivation + non-contradictory outcome;
4. distractor reactivation + contradictory outcome.
The corrective outcome itself carries no cue identity.

Frozen confirmatory median criteria:
- overall accuracy >=0.90
- reactivation+contradiction accuracy >=0.90
- no-reactivation preservation >=0.88
- reactivation+same preservation >=0.95
- distractor preservation >=0.82
- updated-memory hidden-state AUC >=0.94

At least 9/12 seeds must jointly satisfy:
react+contradiction >=0.80; no-reactivation >=0.78; react+same >=0.90; distractor >=0.70; updated-memory AUC >=0.88.

A confirmatory failure is retained. Any corrective R1 must use a new code/hash commitment and new untouched seeds.
