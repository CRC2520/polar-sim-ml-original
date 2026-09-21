# POLAR P0-Critical R1 — corrective prospective campaign

**Freeze date:** 2026-09-20 (America/Lima)  
**Prior confirmatory campaign retained:** P0-Critical seeds 1002001–1002012 = FAIL.  
**R1 development seeds excluded from confirmation:** 1003001–1003004.  
**Untouched R1 confirmatory seeds:** 1004001–1004012.  
**Inferential unit:** seed.

No R1 threshold, task generator, endpoint, comparator, or confirmatory seed may be changed after the R1 confirmatory run begins. The original P0-Critical failure remains part of the scientific record and is not overwritten by R1.

## Why R1 is justified

The original frozen confirmation missed one prespecified endpoint in each experiment while satisfying the remaining conjunctions:

- P0-A: d=8 source MAE = 0.0661156 versus <= 0.065 (miss 0.0011156).
- P0-B: median adaptive-minus-frozen accuracy = +0.0758929 versus >= +0.08 (miss 0.0041071).
- P0-C: median integrated-minus-factorized expected-utility gain = +0.00761738 versus >= +0.008 (miss 0.00038262).

R1 does **not** relax those thresholds. It changes only the mechanism/candidate class indicated by the failed endpoints and uses new untouched confirmatory seeds.

## R1-A — Expanded minimal operational causal feedback quotient

Current sensory observation remains outside the compressed recurrent state. The same train-only PLS quotient and fixed second-order current-input/quotient readout from P0-Critical are retained. Candidate quotient dimensions are now d ∈ {9,10,11}.

The original P0-A PASS thresholds are unchanged for a candidate dimension:
- action agreement >= 0.88;
- confidence MSE <= 0.008;
- source MAE <= 0.065;
- temporal-age MAE <= 0.065;
- gate accuracy >= 0.98;
- relation-routing accuracy >= 0.98;
- self-state MAE <= 0.060;
- metacontrol accuracy >= 0.88;
- quotient-permutation intervention damage >= 0.45;
- matched nuisance perturbation damage <= 0.020;
- causal damage margin >= 0.40.

R1-A PASSes if at least one tested dimension satisfies the complete median conjunction. The reported operational minimum is the smallest passing dimension among {9,10,11}; dimensions 6–8 retain their original first-confirmation results and are not rerun or relabelled. This is minimal only within the declared PLS/readout/candidate class, not a mathematical causal-state theorem.

## R1-B — Faster factual relational reconfiguration

The original P0-B environment, observations, comparator policies, hidden evaluator labels, endpoints and thresholds are unchanged. Only the adaptive learner's factual identification schedule changes:

- relation refresh interval: 15 steps (previously 25);
- factual sliding buffer: 75 steps (previously 80).

The frozen/no-cross/always-on/oracle comparators are unchanged.

The original P0-B PASS thresholds remain:
- adaptive action accuracy >= 0.79;
- adaptive − frozen accuracy >= +0.08;
- adaptive − no-cross accuracy >= +0.07;
- evaluator-only gate AUC >= 0.80;
- evaluator-only true-relation recovery >= 0.65;
- post-map-change recovery latency <= 220 steps;
- oracle − adaptive accuracy gap <= 0.13.

In addition, at least 9/12 seeds must jointly show adaptive−frozen >0.05, adaptive−no-cross >0.05, gate AUC >0.70, relation recovery >0.55 and recovery latency <300. Adaptive-versus-always-on remains descriptive and is not used to claim that binary gating itself is necessary. The intended claim is adaptive relational reconfiguration versus frozen/no-cross organization.

## R1-C — Utility-aligned theory-discrimination with preserved lesion assay

The original module variables, natural-support training distribution, module-recombination OOD test distribution, isomorphic generic recoding, factorized degree-4 comparator, feature budgets, primary path and lesion thresholds are unchanged.

The decision head used for the OOD performance/utility comparison is changed from label classification to direct expected-utility regression. This aligns the optimized target with the preregistered utility endpoint:

- INTEGRATED: full degree-2 cross-module features -> ridge utility head;
- GENERIC-ISOMORPHIC: invertible permutation/sign recoding with the same full quadratic class -> ridge utility head;
- FACTORIZED: degree-4 within-module features only, with more features than INTEGRATED -> ridge utility head.

The original classification head is retained unchanged and evaluated on the same OOD recombination tape solely for the SELF/MEMORY/RELATION/META lesion-signature assay. This prevents the corrective performance head from redefining the lesion endpoints after seeing the original failure.

The original P0-C thresholds remain:
- integrated OOD metacontrol accuracy >= 0.92;
- |integrated − isomorphic generic| accuracy <= 0.002;
- integrated − factorized accuracy >= +0.030;
- integrated − factorized expected utility >= +0.008;
- primary accuracy >= 0.79;
- factorized feature count > integrated feature count;
- SELF lesion: REPLAN recall drop >= 0.25;
- MEMORY lesion: DEFER recall drop >= 0.50;
- RELATION lesion: utility drop >= 0.04 and ACT recall drop >= 0.15;
- META lesion: REPLAN recall drop >= 0.70.

At least 9/12 seeds must show both higher accuracy and higher expected utility for INTEGRATED versus FACTORIZED. GENERIC-ISOMORPHIC is expected to remain equivalent; a PASS supports the organization class, not POLAR-specific labels or notation.

## Global R1 interpretation

`R1_ALL_THREE_PASS` requires R1-A, R1-B and R1-C all to PASS their unchanged criteria. A failure remains a failure; no post-confirmatory threshold changes are permitted.

Even if R1 passes, the result is bounded to this internally designed synthetic campaign. It does not establish phenomenal consciousness, mathematical irreducibility, a neurobiological identity, intrinsic ethics, AGI/ASI, universal polarity, or independent replication.
