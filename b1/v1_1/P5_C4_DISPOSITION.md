# P5 C4 prospective disposition, v1.1

This disposition resolves the *new* P5 comparison contract; B0 and B1-D v1 remain immutable. The frozen JSON prescribes minimal next-demand replenishment without unnecessary terminal recharge. The frozen prose says always request service and replenishment. Their primary physical ceiling agrees, but their actions, costs, deliveries and stock traces differ.

The detailed, machine-readable operational prescription governs C4-v1.1: serve `min(2, reserve, demand)` and, when another epoch remains, replenish `min(2, max(0, 2 - (reserve - served)))`; otherwise replenish zero. Both lanes retain the same physical gate, information and one-future-epoch horizon. The separate controller `C4_FULL_REFILL_CEILING_WITNESS` always requests service and refill, including the terminal epoch. It is not the matched C4 comparator.

This is a `prospective_protocol_disposition`, not a retrospective correction. Authority follows operational specificity and avoiding unnecessary resource use; it does not follow comparative outcomes. Source paths, exact source hashes and the documentary compatibility audit are in the accompanying JSON. No P5-v1.1 comparator benchmark has run at authoring. Publication of this disposition, the affected protocol, policy source and common resource specification must precede that benchmark.

The external loss, deterministic physical laws and initial-state distribution are unchanged. The analytic per-cell ceilings are 62 completed jobs from reserve zero and 64 from reserve eight. Any controller apparently exceeding these ceilings is an instrumentation or matching defect. Replenishment and waste remain secondary outcomes; no new weighted score is introduced.

P5 remains an engineering and ceiling negative control. `positive_utility_eligible=false` and `positive_pairing_specificity_eligible=false`. Descriptive equality does not establish positive pairing evidence. The new 32-bundle development realization uses namespace `PD-B1-D-v1.1`; it is reported separately from v1. No final seeds or B1-E observations are generated.
