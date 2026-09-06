# Study 3A: separately registered active-mechanism panel

Timing: written AFTER inspecting Study 3 development selection and BEFORE generating
any Study 3 or Study 3A final seed. This is a prospective mechanistic follow-on,
not a rewrite of the original protocol and not evidence independent of its design
motivation. The original Study 3 freeze at e44f783e096aafa9822ce517994780d5a89c3ae0
and all five original decision rules remain unchanged.

## Reason
All four development-selected networks chose a=b=0, eta=1. The primary final study
will therefore evaluate a selection procedure that chose to remove the network.
Its matched lesions/topology/feature comparisons cannot identify an active route.
That finding must be preserved, not replaced with a forced-active winner.
This separate panel asks what happens when the K route is actually present.

## Selection fixed before panel outcomes
Reuse the ALREADY archived development scores, not new final outcomes. For full,
rewired, reverse and generic_nonlinear, restrict the original candidate set to b!=0
(18 original candidates per architecture) and select the minimum mean loss over
all six development seeds and twelve cells, with the same deterministic tie break.
Use original retuned_zero settings. no_K removes K from the chosen active full
without changing eta or W. No free edge weights, extra training, task-specific
selection, or new hyperparameter values are introduced. Both active configurations
and selected development means are frozen publicly before the panel begins.

## Independent final realizations and six conditions
Use NEW seeds 902001..902040, not the primary study's final seeds. Reuse the FROZEN
core task generators and controller implementations without modification.
Conditions: full, no_K, retuned_zero, rewired, reverse, generic_nonlinear.
Forty seeds * twelve factorial cells * six conditions = 2880 runs, 184320 transitions.
Within-panel cells/conditions are paired by seed; primary and panel seeds are separate.
Synthetic families and the shared development dataset are not independent replications.

## Fixed analysis
Same external MSE, costs, post-switch loss, return loss and hard constraints as
Study 3. Same five contrasts and engineering thresholds: full-minus-no_K and
full-minus-retuned_zero upper < -0.001; full-minus-rewired/reverse/generic upper
< -0.0005. Bootstrap 20000 complete-seed resamples with RNG 917371, one-sided 99%
upper bounds (Bonferroni .05/5 WITHIN this separately identified panel); also
report descriptive two-sided 95% intervals. Do not claim a single familywise .05
bound for all claims across BOTH studies combined.
Same engineering gates: cost difference upper<=0.05; post-switch-loss difference
upper<=0.005; zero hard violations. Separate harm flag when full-minus-retuned_zero
99% lower>+0.001. No positive panel finding overturns the original selection result.

Require b!=0 in each intact network, correct selected settings, complete unique
trial set, finite full traces, action feasibility, and exact per-record outcome and
environment checks. Record maximum action difference full versus no_K; a zero
behavioral difference would be nonidentification, not evidence of useful tension.
Existing nonzero-graph equivalence regression tests remain applicable. Additionally
re-execute 24 preselected records (2 first/last seeds * 2 first/last cells * 6
conditions) from archived settings and compare actions and outcomes. This replay
is verification, not new evidence.

The panel is labelled separately everywhere. It tests the best active configuration
WITHIN a small predeclared gain grid and fixed templates. A failure does not show
that every active network or every continuously optimized gain would fail. The
full C/E/U/IACL architecture, memory recall and consciousness are not evaluated.
