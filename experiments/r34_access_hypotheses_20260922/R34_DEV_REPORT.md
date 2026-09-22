# R34 final development report before confirmatory freezes

R34 separates three explanations for the mixed A-like result in R33.

## DEV1 retained failure

The first H2/H3 query-context implementation used 120 episodes and 20 epochs. It produced clear context interactions but recurrent learners were not capable relative to the current-only baseline, including in the UNIFORM task. That design is retained as a development failure in R34_DEV1_AMENDMENT.md.

Before any confirmatory seed was opened, the H2/H3 training budget was restored to the R33 budget (160 episodes, 30 epochs). No task/probe definition or confirmatory split changed.

## Final development H1

Fresh H1 development on ring:
- GRU control gain: +0.13853
- GRU prediction gain: +0.07887
- D: 0.01867
- C: 0.00522
- R: 0.00161
- A: 0.000958

Thus GRU is successful and D+C+R+A in development. H1 non-necessity is not assumed; the strict-heldout LSTM confirmatory result remains decisive.

## Final development H2

Selective-query GRU:
- A_switch: +0.00073448
- q=1 history damage: +0.00242853
- q=1 gain vs CURRENT_MLP: -0.11817 (ratio ~1.118; accepted only as non-inferior capability)
- q=0 ratio vs CURRENT_MLP: 0.62835

The confirmatory H2 gate therefore uses q=1 non-inferiority (gain >= -0.25), not superiority, plus direct A_switch and q=1 history damage. LSTM remains strict-heldout.

## Final development H3

RNN:
- SELECTIVE A_switch: -0.00020154
- UNIFORM A_switch: +0.00002338
- interaction: -0.00020869

GRU:
- SELECTIVE A_switch: +0.00134784
- UNIFORM A_switch: -0.00032473
- interaction: +0.00188067

The development architectures therefore already show architecture dependence. H3 will be considered supported only if the contingency pattern replicates in at least two confirmatory memory architectures, including at least one strict-heldout architecture. Universal contingency is not required.

## Confirmatory panels remain unopened

H1: 1807001–1807012
H2: 1817001–1817012
H3: 1827001–1827012

Confirmatory topology families are random-DAG + skew. WINDOW_MLP/LSTM remain absent from formal H3 development, and LSTM remains absent from H1/H2 development.

Source blob: d20877489b14ac482beecca75bbbb6c26ad5fb78
Adjudicator blob: fad1644bb95d41c76f42db8a088c9ca32af65e05
