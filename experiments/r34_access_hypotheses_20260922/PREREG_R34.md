# R34 — three experiments discriminating the status of contextual access A

Status: development protocol before confirmatory freezes  
Date: 22 September 2026  
Parent evidence: POLAR Core v1.0 + R32 + R33_INCONCLUSIVE_CAPABILITY

R33 found four successful generic memory-capable architectures. All four exhibited D+C+R, but only RNN and WINDOW_MLP crossed the frozen A-like contextual-history-specificity threshold. R34 therefore separates three live hypotheses rather than collapsing them into one post-hoc interpretation.

## Experiment H1 — A is not necessary under the R33 task

Question: can a successful generic learner repeatedly solve the original hidden-dynamics adaptive-control task with D+C+R while failing the frozen R33 A criterion?

The original R33 task, success rule and D/C/R/A thresholds are reused without modification. Development uses CURRENT_MLP and GRU on ring. Confirmatory execution adds LSTM and uses random-DAG + skew with fresh seeds.

A positive H1 result requires a reproducible successful D+C+R / A-fail architecture under fresh confirmatory seeds. This supports non-necessity only under the R33 operationalization and task class; it does not prove that no broader access mechanism exists.

## Experiment H2 — the R33 A probe is too narrow / architecture-capacity test

Question: do GRU and LSTM express direct contextual history access when the task contains an explicit query demand that makes history relevant for one query and irrelevant for another?

A generic one-bit query cue is appended to the observation. In the selective-query task:
- q=1 asks for hidden drift A_t x_t, which requires transition history;
- q=0 asks for a current-only local target computable from the current observation/action token.

No D/C/R/A label or access-gate supervision is provided.

The direct A metric is:

A_switch = history_damage(q=1) - history_damage(q=0).

The same history and current physical token are held fixed while only the query cue changes at the probe point.

H2 is supported only if both GRU and strict-heldout LSTM are capable on q=1 without material q=0 degradation and show frozen positive A_switch. This would establish architecture capacity for a broader A-like operation, not prove that the original R33 probe was a false negative on the original task.

## Experiment H3 — A is contingent on task demand

Question: with architecture and generic cue held fixed, does contextual access emerge specifically when the cue changes whether history is useful?

Each architecture is trained twice:

1. SELECTIVE — q=1 requires hidden drift/history; q=0 requires a current-only target.
2. UNIFORM — q is present with the same distribution, but both q values require hidden drift/history.

Primary interaction:

A_interaction = A_switch(SELECTIVE) - A_switch(UNIFORM).

A contingency result requires:
- capability in both tasks;
- positive A_switch in SELECTIVE;
- low/near-zero A_switch in UNIFORM;
- positive architecture-by-demand interaction;
- replication across multiple architectures including a strict-heldout architecture.

## Splits

Development seeds:
- H1: 1806001–1806004
- H2: 1816001–1816004
- H3: 1826001–1826004

Confirmatory seeds remain unopened until the three freezes:
- H1: 1807001–1807012
- H2: 1817001–1817012
- H3: 1827001–1827012

Formal development topology: directed ring.

Confirmatory topologies: random-DAG + skew.

Held-out architecture logic:
- H1/H2: LSTM absent from formal development.
- H3: WINDOW_MLP and LSTM absent from formal development.

No confirmatory threshold may be relaxed after the corresponding seed panel is opened.

## Boundaries

R34 does not by itself establish global minimality, independent replication, biological correspondence, phenomenal consciousness, AGI or ASI. POLAR Core v1.0 remains frozen until all three results are adjudicated.
