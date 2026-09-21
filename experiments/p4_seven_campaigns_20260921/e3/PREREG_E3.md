# E3 PREREG — Cross-Agent Causal Quotient

Freeze: 2026-09-21 America/Lima.
Development seeds excluded: 1331001–1331104.
Untouched confirmatory seeds: 1332001–1332012.

Two independently initialized agents with heterogeneous hidden architectures learn the same five functional endpoints. A post-hoc bottleneck is learned separately for each agent from its hidden state. The core question is whether a 4D quotient is functionally sufficient, lower dimension is inadequate, and independently learned quotients can be aligned/transplanted across agents.

Frozen median criteria:
- 4D quotient held-out MSE <=0.005
- 3D-minus-4D MSE gap >=0.05
- cross-agent quotient alignment >=0.97
- cross-agent quotient-decoder transfer MSE <=0.03
- quotient-permutation causal damage >=0.50
- matched small-noise damage <=0.005

At least 9/12 seeds must satisfy: d4 MSE <=0.008; minimality gap >=0.04; alignment >=0.94; transfer MSE <=0.05; permutation damage >=0.40.

Raw-hidden transfer is reported descriptively and is NOT a success criterion. Development showed that raw hidden states can also be linearly mapped in this simple shared-observation benchmark; a quotient PASS therefore establishes minimality/invariance, not exclusive necessity.
