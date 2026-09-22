# R31 confirmatory result — matched model-free policy-bandit

Confirmatory GitHub Actions run: **35680918266**  
Confirmatory artifact: **10674197612**  
Artifact digest: `sha256:f29ce5312f6e3a5b760211adba5cbd9b99d2cab910035c73dafbf6f0d252e3b8`  
Frozen source SHA-256: `18fb92e78688d4a3a62713370275916005196d1ad6f318926a1d172feac9c989`  
Confirmatory seeds: **1547001–1547012**.

## Decision

**PRACTICAL_EQUIVALENCE_CORE_MODEL_FREE_RL**

The matched model-free baseline is scientifically valid under the frozen R31
criteria, CORE is non-inferior to it, and the tighter practical-equivalence
criterion passes. Neither controller demonstrates an exclusive >=10% advantage.

## Confirmatory medians

- MODEL_FREE_RL stable fraction: **1.0**
- CORE stable fraction: **1.0**
- MODEL_FREE_RL / FROZEN cost ratio: **0.55274255**
- MODEL_FREE_RL / Adaptive-LQR cost ratio: **0.99683636**
- CORE / MODEL_FREE_RL cost ratio: **1.01232371**
- CORE / MODEL_FREE_RL post-shift ratio: **1.00195437**
- CORE / Adaptive-LQR cost ratio: **1.00160660**
- model-free selected-policy nonzero fraction: **1.0**
- mean selected gain scale: **0.7**

## Frozen guardrails

- model-free baseline validity: **12/12 PASS**
- CORE non-inferiority: **11/12 PASS**
- practical equivalence CORE↔MODEL_FREE_RL: **11/12 PASS**
- exclusive CORE advantage: **4/12 — FAIL**
- exclusive MODEL_FREE_RL advantage: **4/12 — FAIL**

## Interpretation

R31 closes the remaining in-house model-free comparator gap for this specific
matched policy-bandit: the current CORE is **not privileged** relative to a
valid model-free learner operating under the same partial-observation/state
estimator and interaction budget. The result is equivalence/non-inferiority,
not superiority.

This does not establish equivalence to RL as a field. It does not change the
R26 strong relational `H_TRANSFER` FAIL, the R30 Adaptive-LQR equivalence
result, or the OPEN status of E6b and E7.
