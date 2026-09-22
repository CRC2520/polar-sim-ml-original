# R33 preregistration — mechanism emergence in generic learners

Status: development protocol, before confirmatory freeze
Date: 21 September 2026
Parent theory: POLAR Core v1.0 + post-R32 evidence

## Question

When generic learners are trained only to solve a hidden-dynamics adaptive-control task, without D/C/R/A labels, POLAR modules or relation supervision, do successful solutions develop functional equivalents of the organizational contracts?

R33 is a falsification test of organizational necessity/emergence. It is not a benchmark claiming that a POLAR-labelled implementation should outperform generic learners.

## Training task

All learners receive only:
- current noisy state observation;
- previous action;
- current control target.

The true transition matrix and its relation topology are never supplied.

Training targets are oracle control actions generated from hidden dynamics, but there is no auxiliary loss for D, C, R, A, topology class, relation matrices, memory, or access.

Training topology families:
- sparse directed;
- modular directed;
- dense low-rank.

Training mixes:
- diagonal;
- gain shift;
- static relational;
- relational shift.

## Generic architecture families

Development architectures:
- CURRENT_MLP — current token only, negative memory control;
- RNN — vanilla recurrent network;
- GRU — gated recurrent network.

Confirmatory architectures:
- CURRENT_MLP;
- RNN;
- GRU;
- WINDOW_MLP;
- LSTM.

Strict architecture hold-outs:
- WINDOW_MLP;
- LSTM.

Neither strict hold-out architecture may be executed during formal development.

## Topology split

Formal development evaluation:
- directed ring.

Confirmatory evaluation:
- random DAG;
- skew-coupled.

Neither confirmatory topology family is used in formal development evaluation.

## Development / confirmatory seed split

Development seeds:
- 1706001–1706004.

Confirmatory seeds:
- 1707001–1707012.

Confirmatory seeds must remain unopened until source, thresholds and adjudicator are frozen.

## Operational mechanism probes

The probes are post-training analyses. They are not training targets.

### D-like differentiation

For the learned representation z, fit a rank-1 PCA reconstruction on the calibration portion of a held-out sequence. Compare action error after replacing z by that rank-1 reconstruction with the intact action error.

D_damage = MSE(rank1(z)) - MSE(z).

Positive damage indicates that causally useful action information is distributed over differentiated latent degrees of freedom rather than a single scalar direction.

### C-like causal history

Evaluate the intact learner and a reset-history intervention on identical held-out tokens.

C_damage is the mean increase in oracle-action MSE after erasing recurrent/window history in relationally informative contexts.

CURRENT_MLP provides the zero-history negative control.

### R-like relation representation

Two diagnostics are retained:
1. linear held-out decoding of the hidden off-diagonal transition map from z;
2. causal history transplant: history generated under a different hidden topology is inserted before the same query token.

Primary R intervention:
R_transplant_damage = MSE(wrong-history query) - MSE(own-history query).

### A-like contextual history access

History should matter more when relational information exists than in a diagonal null.

A_specificity =
mean(history damage in gain-shift, static-relational, relational-shift)
- history damage in diagonal.

This is an A-like functional-access probe; it is not a claim about a psychological workspace.

## Success criterion

The formal confirmatory success rule will be frozen after the development panel. It will combine:
- closed-loop control gain relative to CURRENT_MLP, normalized by the oracle gap;
- held-out action-imitation gain relative to CURRENT_MLP;
- stability.

The confirmatory primary hypothesis is not "one architecture wins". It is:

successful generic solutions -> D-like + C-like + R-like + A-like conjunction.

The strongest disconfirming result is a successful generic learner that repeatedly solves the held-out control problem while failing one or more required functional probes.

## Interpretation boundaries

Even a PASS would establish only same-program evidence that generic learner families converge toward functional equivalents of the proposed organizational contracts in this benchmark.

It would not establish:
- global mathematical necessity;
- E6b independent replication;
- independent task generation by another research group;
- biological correspondence;
- phenomenal consciousness;
- AGI/ASI.
