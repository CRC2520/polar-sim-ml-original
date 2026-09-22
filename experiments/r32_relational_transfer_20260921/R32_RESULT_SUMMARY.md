# R32 result summary — strong relational transfer

**Frozen resolution:** `R32_STRONG_RELATIONAL_TRANSFER_PASS_INTERNAL_HELDOUT`

R32 was designed after the F0/F1 POLAR Core v1.0 freeze to test whether adaptive relation reidentification has a causal effect specific to genuine relational-topology change.

## Confirmatory protocol

- confirmatory seeds: **1607001–1607012**
- confirmatory topology families: dense low-rank, directed ring, random DAG, skew-coupled
- strict held-out topology classes: **random DAG + skew-coupled**
- source Git blob: `be82b21e835cea75f493cc47c10c83c5f40d96d0`
- source SHA-256: `81a2a7e5646f41ab7ab69bed76dafe147ed95dab9ed7a1e0658fb081a922e58b`

All frozen median checks pass. General seed guard: **12/12**. Strict held-out seed guard: **12/12**.

## Primary relation-specific intervention

The frozen intervention was:

[
\Delta_R = J(do(R=R_{old})) - J(CORE).
]

Confirmatory medians:

- relational shift: **0.00224315**
- gain shift: **0.00100039**
- diagonal null: **0.00085475**
- static relational: **0.00081558**
- specificity margin: **+0.00124880**

Thus the cost of freezing the old relation map is specifically larger when the causal topology is actually rewired.

## Held-out transfer

On the strict held-out random-DAG and skew-coupled classes:

- relation-lesion damage: **0.00230252**
- specificity margin: **+0.00098702**
- relation-recovery gain: **+0.361833**
- CORE/GENERIC ratio: **0.999545**
- CORE/FACTORIZED ratio: **0.352608**

The held-out transfer criteria and guardrail pass.

## Mechanism interventions and comparators

Across all confirmatory families:

- CORE/GENERIC ratio: **0.999868** — preserves implementation non-privilege.
- CORE/FACTORIZED ratio: **0.340107**.
- CORE/FROZEN ratio: **0.111720**.
- learned-relation recovery gain vs NO_R: **+0.443328**.
- wrong-relation intervention damage: **+0.00334676**.
- correct-relation rescue: **+0.00360928**.
- CORE stable fraction: **1.0**.

## Scientific interpretation

R32 closes the **internal** strong relational-transfer gap left by R26 for this new benchmark: relation reidentification is causally useful specifically under topology changes and the result transfers to held-out graph families without semantic polar labels.

It does **not** establish:
- E6b independent external replication;
- independently developed task generation by another team;
- global minimality of D+C+R+A;
- superiority over RL as a field;
- biological correspondence;
- phenomenal consciousness.

R30/R31 implementation-equivalence results remain unchanged.

## Technical provenance note

The first confirmatory simulation (run 35685648824) completed and produced the same reported medians but its adjudication stopped because the adjudicator reconstructed the Git blob identity incorrectly. No experimental source, seed, topology family, threshold, metric or decision rule was changed. Only that identity-check implementation was repaired. The exact replay (run 35685849021) passed freeze verification, simulation, frozen adjudication and artifact export.
