# R26 development protocol

Date: 2026-09-21

This file governs **development seeds only**. Confirmatory thresholds and seeds will be
frozen only after development output is inspected.

Development seeds:
- 1406001--1406004.

Confirmatory seeds are reserved and MUST NOT be executed in development:
- 1407001--1407012.

The three assay definitions and controller families are fixed before development output:

## R26-A
Single integrated trajectory with:
- dual continuous control channels whose independent coactivation can be useful;
- online relation estimation from own actions/outcomes;
- delayed one-shot memory cue affecting later auxiliary actions;
- capacity-limited workspace that promotes that memory only at query;
- coordinate-isomorphic orientation+intensity control;
- targeted no-D, no-C, no-R and no-A interventions.

## R26-B
A null generator has diagonal action effects and one-channel-at-a-time targets. The
desired result is not superiority: off-diagonal discovery, relation-lesion damage and
coactivation advantage should remain small.

## R26-C
Canonical continuous CartPole and Pendulum equations are used without POLAR semantic
pairs. Parameters shift twice during a long trajectory. Controllers:
- CORE: recurrent state estimate + adaptive sparse relational model + LQR;
- GENERIC: recurrent state estimate + adaptive dense model + LQR;
- FACTORIZED: same data and update budget but cross-state terms removed;
- FROZEN: nominal model is never reidentified;
- NO_C: velocity/history estimate removed;
- ORACLE: exact-current-parameter local LQR with true state (privileged reference).

All non-oracle adaptive controllers receive the same noisy partial observations and
action opportunity. ORACLE is not a matched baseline; it is an upper/reference
controller with privileged state/physics.

No confirmatory claim may be based on development seeds.
