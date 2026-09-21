# POLAR P0-Critical R1 — confirmatory result

**Execution date:** 2026-09-20 (America/Lima)  
**R1 confirmatory seeds:** 1004001–1004012  
**Prior P0-Critical confirmation:** retained as FAIL (1002001–1002012)

## Frozen result

**R1_ALL_THREE_PASS = TRUE**

### R1-A — Minimal operational causal feedback quotient
- d=9: FAIL (median metacontrol accuracy 0.876880 < 0.88)
- d=10: PASS
- d=11: PASS
- smallest passing R1 candidate: **d=10**
- d=10 median action agreement: 0.912594
- confidence MSE: 0.003097
- source MAE: 0.055037
- temporal-age MAE: 0.039105
- self MAE: 0.044423
- metacontrol accuracy: 0.882049
- quotient-permutation damage: 0.533553
- nuisance damage: 0.005911

### R1-B — Adaptive relational necessity
- adaptive accuracy: 0.821429
- adaptive − frozen: +0.109286
- adaptive − no-cross: +0.088036
- adaptive − always-on: +0.003214 (descriptive only)
- oracle gap: 0.087143
- gate AUC: 0.847230
- true-relation recovery: 0.804574
- recovery latency: 174.5 steps
- joint per-seed guardrail: 10/12

Interpretation is limited to adaptive relational reconfiguration versus frozen/no-cross organization. The binary gate itself is not claimed necessary.

### R1-C — Theory-discriminating organization
- INTEGRATED OOD accuracy: 0.976687
- GENERIC-ISOMORPHIC OOD accuracy: 0.976687
- FACTORIZED OOD accuracy: 0.837302
- INTEGRATED − FACTORIZED accuracy: +0.141865
- expected-utility gain: +0.018294
- |INTEGRATED − GENERIC|: 0.000000
- primary accuracy: 0.805556
- feature count: integrated 90; factorized 110
- paired accuracy+utility wins: 12/12

Median lesion endpoints:
- SELF -> REPLAN recall drop: 0.427273
- MEMORY -> DEFER recall drop: 0.813657
- RELATION -> ACT recall drop: 0.241465
- RELATION -> utility drop: 0.058608
- META -> REPLAN recall drop: 0.758454

## Reproducibility

The confirmatory campaign was rerun without code changes and reproduced the result JSON exactly.

- base executable SHA-256: `c917e4f8635ad30994d9f21d04ae7a41894b8f653c139e8c0a4371e0ec186f6d`
- R1 executable SHA-256: `ec22406597e15590ed61ad78584fd23a796eaa47c06d75850c684e332da7715f`
- R1 prereg SHA-256: `65f7f12f6db31d6e11fa91339c9370a972ffc048aa34f8ad172482b9bf9a1ce0`
- R1 results SHA-256: `c4afc9f201563e1a501d90703491f645af69b246cd11be875ccc217d82e3f9a2`
- exact replay SHA-256: `c4afc9f201563e1a501d90703491f645af69b246cd11be875ccc217d82e3f9a2`
- arithmetic/hash audit SHA-256: `e04e63fd5a97e7b13955867d51aee9d10f571b4c88e6a52fd82581d88ea87f3d`
- independent audit: 16/16 PASS

## Scientific boundary

This closes the three declared P0 gaps only in the bounded synthetic benchmark and tested model/comparator classes. It does not establish phenomenal consciousness, mathematical irreducibility of the entire agent, universal polarity, neurobiological identity, intrinsic ethics, AGI/ASI, or independent external replication.
