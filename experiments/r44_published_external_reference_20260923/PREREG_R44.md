# R44 — published external-reference compatibility on the original POPGym StatelessCartPole benchmark

Status: prospective before R44 development/evaluation seeds are opened.
Date: 23 September 2026 America/Lima.

## External reference fixed before R44 execution

Source: Morad et al., POPGym: Benchmarking Partially Observable Reinforcement Learning, ICLR 2023, Appendix Table 3.
Paper experiment commit: proroklab/popgym e397e5e.
Paper recipe: 15M environment timesteps per experiment, three trials, externally executed by the POPGym authors.

Published StatelessCartPole reference:
- Easy LSTM: 1.000 ± 0.000; MLP: 0.722 ± 0.001.
- Medium LSTM: 1.000 ± 0.000; MLP: 0.398 ± 0.006.
- Hard LSTM: 1.000 ± 0.000; MLP: 0.265 ± 0.002.

These are published MMER/max-training endpoints, not checkpoints and not paired evaluation episodes. R44 therefore tests benchmark-ceiling compatibility, not paired statistical non-inferiority.

## Scientific question

Does the frozen POLAR-class recurrent observer/controller used in R37 reach near-ceiling performance on the exact historical StatelessCartPole implementation that underlies the external published LSTM result, while retaining a clear recurrence effect and coordinate non-privilege?

R44 does not train or tune the controller.

## Exact external environment

Install and execute proroklab/popgym at commit e397e5e with its historical Gym API.

The environment class is popgym.StatelessCartPole{Easy,Medium,Hard}. In this exact source, the observation contains cart position and pole angle; velocities are hidden.

## Frozen controllers

R44 reproduces the pre-existing R37 controller equations without parameter search:
- CORE_C: finite-difference recurrent velocity estimate with EMA 0.65/0.35 and nominal LQR.
- GENERIC_ISO: exact signed-coordinate recoding of CORE_C.
- NO_C_CURRENT: same LQR and visible coordinates but hidden velocities set to zero.
- ORACLE_STATE: evaluator-only reference using get_state().

No learned parameters, environment-specific search, checkpoint selection or reward shaping.

## Development / confirmation split

Development Easy:
- seeds 2156001–2156008;
- 4 episodes per seed/controller.

Development authorization requires:
- median CORE_C >= 0.95;
- median CORE_C - NO_C_CURRENT >= +0.20;
- median |CORE_C-GENERIC_ISO| <= 1e-12;
- median ORACLE_STATE >= 0.99;
- at least 7/8 seeds with CORE_C >=0.90, recurrence gain >=0.15 and iso gap <=1e-12.

If development fails, Medium/Hard seeds stay unopened.

Confirmatory Medium and Hard:
- seeds 2157001–2157020;
- 4 episodes per seed/controller/environment.

Each difficulty independently requires:
- median CORE_C >=0.95;
- median CORE_C - NO_C_CURRENT >= +0.20;
- median |CORE_C-GENERIC_ISO| <=1e-12;
- median ORACLE_STATE >=0.99;
- at least 18/20 seed guards with CORE_C >=0.90, recurrence gain >=0.15 and iso gap <=1e-12.

Published-reference compatibility additionally requires:
- absolute gap from the published LSTM ceiling 1.000 <=0.05 for BOTH Medium and Hard.

R44 PASS requires development authorization plus both confirmatory difficulties.

## Interpretation

A PASS would show that a frozen POLAR-class recurrent observer/controller is compatible with the ceiling reached by an externally published LSTM baseline on the exact historical benchmark implementation, while preserving a causal recurrence contrast and coordinate non-privilege.

It would NOT be a paired model-vs-model experiment because the external paper does not provide a reusable checkpoint in this campaign and its published endpoint is MMER rather than our held-out episode mean.

It would NOT establish:
- POLAR superiority;
- E6b independent replication;
- D or R on this benchmark;
- universal recurrence benefit;
- global minimality;
- E7;
- consciousness.

Core v1.1 remains unchanged.
