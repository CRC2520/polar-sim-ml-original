# R34 result summary — three contextual-access hypotheses

R34 tested the three explanations left open by R33's mixed A-like result.

## Frozen resolutions

- H1 — A is not necessary under the original R33 task: **R34_H1_INCONCLUSIVE**
- H2 — the R33 A probe is too narrow / broader contextual-access capacity: **R34_H2_BROADER_A_CAPACITY_SUPPORTED**
- H3 — A is a task-demand-contingent pattern that should replicate across architectures: **R34_H3_A_TASK_CONTINGENCY_NOT_SUPPORTED**

The combined result favors H2. It does not justify removing A from POLAR Core v1.0, and it does not establish universal A necessity.

## H1 — original R33 task

Fresh confirmatory seeds 1807001–1807012 on random-DAG + skew:

- GRU is successful and passes D+C+R+A:
  - control gain +0.17618
  - prediction gain +0.12772
  - D 0.015712
  - C 0.002942
  - R 0.000956
  - A 0.000298
- LSTM passes D+C+R+A on mechanism medians but narrowly misses the frozen R33 architecture-success gate:
  - control gain +0.04642 < +0.05
  - prediction gain -0.03737

The preregistered D+C+R / A-fail counterexample is therefore not reproduced. Per-seed DCR-success/A-fail counts are only 2/12 for GRU and 1/12 for LSTM. Because only one noncurrent architecture is successful under the original gate, H1 also cannot establish A necessity. Resolution: INCONCLUSIVE.

## H2 — direct contextual history access

In the selective-query task the same history/current physical token is probed while only the query cue changes whether history is relevant.

GRU:
- A_switch +0.00067325
- q=1 history damage +0.00207695
- q=1 gain vs CURRENT_MLP -0.09752, within the frozen non-inferiority band
- q=0 ratio 0.67636
- seed guard 8/12

Strict-heldout LSTM:
- A_switch +0.00225783
- q=1 history damage +0.00455587
- q=1 gain -0.14886, within the frozen non-inferiority band
- q=0 ratio 0.28056
- seed guard 8/12

CURRENT_MLP has zero history effects.

Thus GRU and LSTM both demonstrate direct context-sensitive history use under the broader intervention. This establishes architecture capacity for an A-like operation. It does not prove that the original R33 A metric was a false negative on the original task.

## H3 — selective versus uniform task demand

All four memory-capable confirmatory architectures are capable under both frozen task-capability gates.

Median interactions:
- GRU: +0.00083298 — contingency pattern PASS, seed guard 8/12
- LSTM: +0.00303938 — large median interaction but seed guard only 6/12; FAIL frozen pattern
- RNN: +0.00032788 — seed guard 6/12; FAIL
- WINDOW_MLP: +0.00000714 — seed guard 1/12; FAIL

No strict-heldout architecture passes the full frozen contingency pattern. The requirement of at least two pattern architectures including one held-out is therefore not met.

H3's universal/replicated task-contingency formulation is NOT SUPPORTED. This does not mean task demand never modulates A; GRU provides a positive example, while LSTM has a large median interaction that lacks the frozen per-seed robustness.

## Combined scientific interpretation

R33's GRU/LSTM A failures should not be interpreted as architectural inability to perform contextual access: H2 directly falsifies that explanation at the tested scope.

The evidence also does not support dropping A from the core on the basis of H1: a fresh GRU panel passes A and the heldout LSTM is mechanism-positive but capability-borderline under the old success gate.

The clean current conclusion is:

- D+C+R remains the most consistently emergent conjunction across the R33 generic learners.
- A is demonstrably realizable by GRU/LSTM when context explicitly controls the relevance of history.
- The specific R33 A_history_specificity metric is not a complete architecture-capacity test.
- A's cross-task contingency is not yet robust enough to be a universal architecture-independent law.
- POLAR Core v1.0 should remain frozen as D+C+R+A pending a revised operational definition of A, rather than being reduced to D+C+R now.

## Development provenance

The first H2/H3 development panel used 120 episodes / 20 epochs and was rejected as a capability-design failure because recurrent learners underperformed CURRENT_MLP even in the uniform-history task. Before confirmatory seeds were opened, the training budget alone was restored to the R33 budget (160 episodes / 30 epochs). This amendment is preserved in R34_DEV1_AMENDMENT.md.

No confirmatory threshold was changed after opening the three seed panels.

## Boundaries

Same-program evidence only. E6b, E7, independent task generation, global minimality and phenomenal consciousness remain open/not established.
