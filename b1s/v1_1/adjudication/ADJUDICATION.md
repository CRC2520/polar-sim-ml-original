# B1-S v1.1 adjudication and B1-E disposition

**ADJUDICATION_COMPLETE**

Scientific source: `50c4e08cf3d69c998ce95b8da1e1647a09b31339`
Result publication: `de35a0995d96ff548772fa56c3620466495add72`

This document adjudicates completed development. It is not a preregistration, independent replication or B1-E execution; it changes neither archived results nor thresholds.

## Decision

`ON_HOLD_FOR_CURRENT_STRUCTURAL_CONFIRMATION`

Outstanding reasons: ["STRUCTURAL_STABILITY_DIAGNOSTIC_NOT_PASSED", "GENERIC_COMPARATOR_COMPETENCE_NOT_PASSED"]

Stability and competence diagnostics are preserved exactly as fixed. Favorable loss direction is neither an automatic progression requirement nor a certificate of advantage. This disposition concerns the current contrast of a stable active-route structure against competent alternatives; it does not prohibit a future confirmation of an adverse result under a separately specified hypothesis.

## Adjudicated results

Selected support: `000001`
Winners by replicate: ["010101", "100100", "001000", "110010", "011000", "001101"]
Modal frequency / mean Jaccard: 0.166666667 / 0.202222222
Stability diagnostic: False

| Comparator | Mean loss | Displacement | Competence |
|---|---:|---:|---|
| S-M0 | -1.00394293 | 0.565962508 | False |
| S-M2-fixed | 5.70871184 | 0.619157508 | False |
| S-M3 | 1.53069901 | 2.19718704 | False |
| S-M4 | -5.07249815 | 2.60690617 | False |
| S-M5 | 25.345387 | 1.56178281 | False |
| S-M6 | 4.25314383 | 0.0887630383 | False |

| Contrast: S-M3 minus alternative | Mean | Nominal interval | Preserved classification |
|---|---:|---|---|
| S-M0 | 2.53464195 | [-7.896817934856722, 12.966101826653999] | INCONCLUSIVE |
| S-M2-fixed | -4.17801282 | [-13.464064224438397, 5.108038581003251] | INCONCLUSIVE |
| S-M4 | 6.60319716 | [-3.3836980771091243, 16.590092396428833] | INCONCLUSIVE |
| S-M5 | -23.814688 | [-62.141211321327376, 14.511835253597066] | INCONCLUSIVE |
| S-M6 | -2.72244482 | [-17.444926389317786, 12.000036752876547] | INCONCLUSIVE |

Intervals are exploratory over six independent training replicates; they are not adjusted for multiple confirmatory claims. An identity control supplies no additional evidence. Failure to exceed a margin proves neither equivalence nor a null effect.

## Causal contribution and scope

```json
{
  "available_prefixes": 48,
  "unavailable_prefixes": 0,
  "tested_edge_interventions": 48,
  "route_use_identified": true,
  "external_difference_by_rep": [
    0.010807979408127777,
    0.0,
    0.0,
    -20.710789252480016,
    0.0,
    0.0
  ]
}
```

Local route use and its external effect are distinct questions. Unavailable prefixes are retained and are not counted as zero-effect lesions. Common environmental exposure does not equalize algorithms, parameters, memory or update counts.

## B1-E boundary

`ready_for_b1e_freeze=false`; `ready_for_b1e_confirmatory_run=false`; `B1E_executed=false`; `final_seeds_generated=false`.

Development and adjudication can finish while confirmation remains on hold. B1-E still requires a new protocol identifying the estimand, contrast, precision/sample size, missing-data rules, resources and seed custody before execution. The old P7 confirmation sample size is not inherited. This adjudication authorizes neither new training nor automatic retuning.

## Non-claims

`H_CAT=NOT_EVALUABLE`; `H_TRANSFER=NOT_EVALUATED`.

The eight-polarity catalogue, full architecture, consciousness, ASI and a physical law are not validated. The previous version and numbers are preserved. Integrity checks establish documentary correspondence for examined files, not full training replication or certification of every scientific assumption.
