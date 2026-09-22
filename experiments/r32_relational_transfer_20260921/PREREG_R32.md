# R32 preregistration — strong relational transfer and relation-specific intervention

Status: development protocol, before confirmatory freeze
Date: 21 September 2026
Parent theory: POLAR Core v1.0 (F0/F1)

## Question

Does adaptive relation reidentification have a causal effect that is specific to genuine changes in relational topology, rather than to generic gain changes, diagonal dynamics, or continued fitting?

R32 is not a test of phenomenal consciousness and not a claim that a POLAR-labelled implementation should beat every generic controller.

## Environment

Six-dimensional stable continuous linear dynamical networks:

x_(t+1) = A_t x_t + B u_t + epsilon_t.

The relation object is the off-diagonal causal map in A_t. The action map B is fixed and diagonal so that relation identification is isolated from actuator identification.

No semantic polar labels or fixed historical pairs are supplied.

## Conditions

1. diagonal: no cross-variable relations; diagonal gains vary.
2. gain_shift: relational topology is fixed while local diagonal gains vary.
3. static_relational: cross-variable topology is present but does not change.
4. relational_shift: cross-variable topology is rewired at block boundaries while diagonal gains are preserved.

## Formal development / confirmatory separation

Development seeds: 1606001–1606004.

Formal development families:
- sparse directed;
- modular directed.

Confirmatory families:
- dense low-rank;
- directed ring;
- random DAG;
- skew-coupled.

The random-DAG and skew-coupled families are designated strict held-out topology classes for confirmatory adjudication.

Confirmatory seeds will be 1607001–1607012 and must not be executed until source and criteria are frozen.

## Controllers

- CORE: online full relation estimator with small-coefficient thresholding.
- GENERIC: equal-information dense estimator without POLAR naming.
- NO_R: learns the initial relation map, then freezes off-diagonal relations while allowing diagonal/local terms to continue updating.
- FACTORIZED: diagonal-only adaptive model.
- FROZEN: full model learned in the initial block and then frozen.
- ORACLE: true current dynamics; descriptive upper bound only.

## Primary causal intervention

At each block boundary, the current CORE off-diagonal relation map is snapshotted before the environmental change.

The counterfactual lesion do(R = R_old) holds that pre-shift relation map fixed while retaining the current diagonal/local estimate. All counterfactual actions are evaluated from the same state, target, action map and true current dynamics.

Define Delta_R = J(do(R = R_old)) - J(CORE).

The preregistered discriminating pattern is:

Delta_R(relational shift) >
max[Delta_R(gain shift), Delta_R(diagonal), Delta_R(static relational)].

Additional interventions:
- do(R = R_wrong): permutation of the learned off-diagonal map.
- do(R = R_correct): true current map as an intervention/rescue reference, not a fair performance baseline.

## Secondary mechanism tests

- CORE/GENERIC practical equivalence;
- CORE versus FACTORIZED under relational shift;
- CORE versus FROZEN under relational shift;
- relation-recovery error against NO_R;
- stability guard.

## Interpretation

A PASS can establish only an internal held-out strong relational-transfer result across unlabelled topology classes within this benchmark. It does not constitute E6b independent replication, external-team task generation, general RL superiority, biological validation or consciousness evidence.
