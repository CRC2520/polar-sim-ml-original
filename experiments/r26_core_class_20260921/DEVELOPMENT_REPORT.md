# R26 development report

Official development run: GitHub Actions run **35647228720**  
Development commit: `2a5c3d064af53f44a4324cc79974a011fd1cb11b`  
Experiment source SHA-256:
`5217ff2124db15eeb883c9b66816727490f74085aa1e8a4c92d4944155cf30f4`

Development seeds only: 1406001--1406004. Confirmatory seeds 1407001--1407012
were not executed.

## Development medians

### R26-A Integrated Core Conjunction
- full control MSE: 0.02905084
- generic-isomorphic gap: 0
- no-D damage: +0.18465020
- no-R damage: +0.10457221
- memory advantage over no-C: +0.41666667
- query-access advantage over no-A: +0.41666667
- contextual access promotion: +0.50000000
- relation score: 0.94257950
- full memory accuracy: 1.0
- full query report accuracy: 1.0

### R26-B Contingent-Polarity Null
- generic-isomorphic gap: 0
- final off-diagonal norm: 0.00057075
- no-R damage: +0.00000063
- no-D/coactivation damage: 0

### R26-C External Standard Dynamics
Canonical CartPole and Pendulum equations, with physical-parameter shifts:
- CORE/GENERIC cost ratio: 0.99984346
- CORE/FACTORIZED ratio: 1.00005612
- GENERIC/FACTORIZED ratio: 1.00013193
- CORE/FROZEN ratio: 0.79961044
- CORE/NO-C ratio: 0.01281263
- CORE/ORACLE ratio: 2.90150400 (descriptive only; oracle has privileged state/physics)
- CORE stable fraction: 1.0
- GENERIC stable fraction: 1.0

Development therefore suggests a clean distinction to freeze prospectively:
external recurrent-state utility and generic equivalence are testable; **external
relational advantage is not assumed** and receives a separate strong-transfer criterion.
