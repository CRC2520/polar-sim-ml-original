# POLAR R11 — four-gap corrective design (development stage)

Base evidence:
- R10 E1 Causal-State Discovery passed 10/12 with the pilot-selected macro-state ERR+GOAL+MEM.
- R10 E2, E3, E4 and E5 failed and remain immutable historical results.
- R10 E5b removed general trajectory collapse but still failed counterfactual and lesion criteria.

R11 does not tune old outcomes. It builds a new agent around the positive E1 result.

## Single architectural correction

R11 uses a compact causal-control state:
1. prediction-error history (ERR),
2. persistent goal state (GOAL),
3. associative transition memory (MEM).

No explicit 16-step R9 history or previous-prediction block is required by the R11 native planner.

The compact state supports:
- dual local/cross recursive least-squares transition models,
- uncertainty estimated from factual residuals,
- a utility gate learned from randomized own gate use rather than hidden labels,
- action-conditioned counterfactual predictions for all four actions,
- native self/world/source attribution from predicted own effect versus residual change,
- native time-since-world-event tracking,
- associative memory used directly in action scoring.

## Gap corrections

### G2 / specificity
The fixed eight-pair catalogue is no longer assumed to be privileged. A separate
relational benchmark discovers a disjoint pairing from training data by residual
interaction evidence. A same-parameter generic comparator uses a random pairing.
The result can support learned relational organization only, not H_CAT.

### G3 / gate
The gate learns expected downstream utility gain from randomized ON/OFF experience.
It is never supervised by a hidden regime label. Hidden labels exist only for final
diagnostic accuracy.

### G4 / transfer
R11 trains a prior only on ecology_train. Target-domain parameters are not fitted
before transfer. A fixed 32-step own-action adaptation prefix is allowed and is
scored separately from the remaining 288-step target trajectory. RLS updates use
only observed transitions/rewards.

### G5 / integrated precursors
One R11 agent in one 640-step trajectory simultaneously exposes native source,
world-age, uncertainty, memory, counterfactual action values and relational/cross
features. Selective lesions are evaluated on matched exogenous tapes.

Development seeds may tune numerical constants and thresholds. No confirmatory
seed is generated or executed until a later frozen PREREG_R11.md commit.
