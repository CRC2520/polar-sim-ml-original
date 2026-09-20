# POLAR R11b — causal-invariance corrections after transfer-benchmark audit

This protocol supersedes R11a for future execution only. R11a remains a failed development pilot.
No confirmatory R11 seed has been observed.

## Seeds
Development/pilot: 972001–972004.
Confirmatory: 973001–973012, untouched until FREEZE_R11B.json exists.
Global PASS: >=9/12 seed-level conjunctions.

## Transfer benchmark validity prerequisite
Two new target families, oscillatory_reservoir_v2 and maintenance_queue_v2, are coded
without using confirmatory outcomes. Before any confirmatory agent is run:
- an evaluator-only best-constant feasibility audit on 100 fixed development tapes must
  show mean alive >= .95 and mean reward >= .50 in each family;
- zero-action mean reward must be <= .10 in each family.
This establishes attainable but nontrivial reward. The oracle/constant results never enter a policy.

## Architecture
R11 uses R9 unchanged plus the 39-dimensional causal-invariance adapter:
24 relational coordinates (8 orientations, 8 intensities, 8 mismatch tensions) +
15 causal-core coordinates (current factual error, uncertainty, goals, delayed memory forecasts).
Action-specific RLS/L1 models learn only from the agent's own executed transitions.
A reliability comparison against R9 controls whether learned predictions enter decision making.
Memory forecasts are additionally utility-gated against a persistence baseline.
Native source attribution is defined from predicted action effect versus unexplained residual;
world-event age resets from the agent's own source estimate, not evaluator labels.

## E2 — isomorphic relational control
Unchanged substantive R11 criterion: relational vs fixed orthogonal isomorphic basis, equal
information/parameters/update budget. PASS per seed requires +.02 return and >=-.01 alive
in both internally designed relational held-out families plus ecology anti-overfit guard >=-.01.

## E3 — utility-aligned gate
Unchanged: balanced accuracy >=.85 for factual cross-utility label, >=.02 MSE gain over the
better constant gate, and >=.02 context-permutation damage.

## E4 — feasible cross-domain fast adaptation
R9 parameters see ecology only. R11 may update only the small adapter from own target-domain
transitions. First 64 steps are declared adaptation; scoring uses steps 64–319.
On BOTH feasibility-audited families:
- alive >=.80
- reward >=.35
- reward improvement over frozen R9 >=.05
- alive improvement over frozen R9 >=.10.

## E5 — integrated functional precursors
A new viable repeated-context trajectory balances self/world/mixed events by construction
without exposing their labels to the policy. Score on second half after online adaptation.
Validity: alive >=.80 and all three source classes represented.
Base: native self/world BA >=.70; source BA >=.55; time BA >=.45;
metacognitive AUC >=.65; native counterfactual best-action accuracy >=.45.
Lesions: noMemory, permuted_content and noCross each reduce return >=.01;
absolute alive change for each <=.10.
Global PASS: complete conjunction >=9/12.

No E5 endpoint measures phenomenal experience.
