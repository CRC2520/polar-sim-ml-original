# R33 result summary — mechanism emergence in generic learners

**Frozen resolution:** `R33_INCONCLUSIVE_CAPABILITY`

R33 tested whether generic learners trained without POLAR labels, D/C/R/A supervision, topology labels or relation supervision would converge toward functional equivalents of the POLAR Core contracts when successful on hidden-dynamics adaptive control.

## Frozen confirmatory panel

- seeds: **1707001–1707012**
- training topology families: sparse, modular, dense-low-rank
- confirmatory topology families: **random-DAG + skew-coupled**
- architectures: CURRENT_MLP, RNN, GRU, WINDOW_MLP, LSTM
- strict architecture hold-outs: **WINDOW_MLP + LSTM**
- frozen source Git blob: `1f5f36a8da5eebc2b4a61876b727c4f616529170`
- source SHA-256: `841d830fa714b4c8a80b4a5d3cec73ac41197287b13947e28eb6abf3cbae0e5a`

## Capability result

Under the frozen architecture-level success rule:

- CURRENT_MLP: **not successful**
- RNN: **successful**
- GRU: **successful**
- WINDOW_MLP: **successful**
- LSTM: **successful**

Both strict held-out architectures were therefore capable on the confirmatory topologies.

Median adaptive-control gains relative to CURRENT_MLP:

- RNN: **+0.3882**
- GRU: **+0.1312**
- WINDOW_MLP: **+0.7674**
- LSTM: **+0.0760**

WINDOW_MLP also improves hidden-drift prediction by **+0.4419** and RNN by **+0.3337**.

## Mechanism-emergence result

Frozen thresholds:
- D rank-1 collapse damage >= 0.010
- C history-reset damage >= 0.0010
- R wrong-history transplant damage >= 0.0005
- A contextual-history specificity >= 0.0002

Architecture medians:

| Architecture | Success | D | C | R | A | Full conjunction |
|---|---|---|---|---|---|---|
| CURRENT_MLP | no | PASS | FAIL | FAIL | FAIL | no |
| RNN | yes | PASS | PASS | PASS | PASS | **yes** |
| GRU | yes | PASS | PASS | PASS | FAIL | no |
| WINDOW_MLP | yes | PASS | PASS | PASS | PASS | **yes** |
| LSTM | yes | PASS | PASS | PASS | FAIL | no |

The full D+C+R+A conjunction therefore appears in **2/4 = 50%** of successful generic architectures.

The same 50% rate holds in the strict architecture hold-outs: WINDOW_MLP passes the full conjunction; LSTM is successful but misses the frozen A-like threshold.

The CURRENT_MLP negative control behaves as expected: it is not successful and has zero C/R/A-like intervention effects.

## Important scientific consequence

R33 does **not** support the strong statement that every successful generic solution converges to the full D+C+R+A conjunction.

At the same time it does not produce a frozen "organizational counterexample" under the preregistered rule, because GRU and LSTM still exhibit **three of four** functional mechanisms (D+C+R) and no successful architecture passes two or fewer.

The strongest common pattern across all four successful architectures is:

[
\boxed{D+C+R}
]

while the A-like contextual-history specificity is architecture-dependent under the present operationalization.

In particular:

- GRU A-like specificity: **0.00015997**, below the frozen 0.0002 threshold.
- LSTM A-like specificity: **0.00000461**, well below threshold.
- RNN A-like specificity: **0.00034051**, PASS.
- WINDOW_MLP A-like specificity: **0.00046383**, PASS.

This means the current evidence does not justify upgrading D+C+R+A from a candidate reduced core to a universal necessity claim.

## Relation decoding note

Linear decoding of the relation matrix remains negative for every architecture. This diagnostic was explicitly excluded from the frozen gate after development. The causal R-like evidence is instead the preregistered wrong-topology history-transplant damage, which passes in all successful architectures.

The result therefore suggests relational information can be functionally used without being linearly exposed in the learned representation.

## Development provenance

DEV1 direct-action imitation failed as an experimental design because recurrent learners were less capable than the current-only control despite history sensitivity. Before confirmatory seeds were opened, the task was amended to hidden-dynamics prediction and an 8-step post-shift identification burn-in. These development changes are preserved in `R33_DEV1_AMENDMENT.md` and `R33_DEV_REPORT.md`.

No confirmatory seed, random-DAG/skew formal result, WINDOW_MLP formal result or LSTM formal result was opened before the final source and criteria freeze.

## Boundaries

R33 remains same-program evidence. It does not establish:
- global mathematical necessity;
- E6b independent replication;
- external task generation;
- biological correspondence;
- phenomenal consciousness.

R30/R31 implementation non-privilege and R32 internal held-out relation-specific transfer remain unchanged.
