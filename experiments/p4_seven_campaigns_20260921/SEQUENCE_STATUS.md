# POLAR — Seven-campaign sequence status (2026-09-21)

## Confirmatory internal campaigns

### E1 Emergent Reconsolidation — PASS
Confirmatory seeds 1312001–1312012. Guardrail 12/12.
Medians:
- overall 0.973750
- reactivation + contradiction 0.995636
- no-reactivation preservation 0.955706
- reactivation + same evidence 1.000000
- distractor preservation 0.924046
- updated-memory hidden AUC 0.988603

### E2 Polarity Discovery — PASS
Confirmatory seeds 1322001–1322012. Guardrail 10/12.
Retained guardrail failures: 1322003, 1322005.
Medians:
- mean axis alignment 0.996678
- minimum alignment 0.991145
- held-out MSE 0.001035
- opposite-extreme cosine approximately -1
- context control ratio approximately 1
- alignment gain over equal-size random axes +0.441887

### E3 Cross-Agent Causal Quotient — PASS
Confirmatory seeds 1332001–1332012. Guardrail 12/12.
Medians:
- d4 MSE 0.001835
- d3 MSE 0.067358
- minimality gap +0.065622
- cross-agent quotient alignment 0.991099
- cross-agent decoder transfer MSE 0.015950
- quotient permutation damage 0.625405
- matched noise damage 0.000785

Adverse boundary retained: raw-hidden transfer MSE is 0.001097, so the quotient is not uniquely required for cross-agent mapping in this benchmark.

### E4 Naturalistic Organizational Necessity — PASS
Confirmatory seeds 1342001–1342012. Guardrail 10/12.
Retained guardrail failures: 1342004, 1342006.
Standard-equation families: SIR, Lotka–Volterra, bimolecular mass-action reaction, two-species competition.
Medians:
- integrated/factorized OOD MSE ratio 0.00004933
- integrated/monolithic OOD MSE ratio 0.557145
- integrated wins factorized 4/4
- integrated wins monolithic 3/4

This supports cross-variable organization/sample efficiency, not a POLAR-specific superiority claim.

### E5 Theory-Discriminating Consciousness Reference Implementations — PASS
Confirmatory seeds 1352001–1352012.
Complete lesion signature counts:
- POLAR-access 12/12
- GNW-like 12/12
- HOT-like 12/12

These are reference implementations for discriminating functional predictions. They are not evidence that one theory is true or that any implementation is conscious.

## E6 Independent Replication

Status: **EXTERNAL INDEPENDENT REPLICATION OPEN**.

Internal clean-room/cross-implementation proxy:
- alternative LSTM reproduces reactivation-dependent updating in 4/4 tested seeds;
- independent MLP + Jacobian/SVD polarity analysis recovers hidden axes with mean/min alignment >0.9998 in 4/4;
- PLS quotient reimplementation yields d4 MSE about 0.0056 vs d3 about 0.0719 and cross-agent alignment about 0.998 in 4/4.

These results establish implementation robustness only. They cannot satisfy the scientific requirement that another team independently implement the frozen specification.

## E7 Prospective Biological Correspondence

Status: **PROSPECTIVE BIOLOGICAL VALIDATION OPEN**.

A retrospective literature/data-availability check finds some published Cogitate findings compatible with POLAR-access predictions, but the aggregate results are already public and therefore cannot be relabeled as prospective validation. A new or genuinely held-out preregistered biological dataset is required.

## Sequence decision

Internal architecture-building sequence E1–E5 is closed under the frozen criteria.
E6–E7 are external evidence requirements and cannot be forced to PASS by internal reformulation without invalidating their meaning.
