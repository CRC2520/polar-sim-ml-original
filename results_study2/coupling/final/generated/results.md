# Study 2: contextual coupling

Split: **final**. 1920 complete controller trials.

Primary unit: seed, equally averaging the six task-by-regime cells; lower loss is better.

Paired minus planning lesion: **-0.002700**, 95% paired-seed interval **[-0.002829, -0.002577]**, n=40 seeds.

Predeclared decision: **pivot_to_conditional_structural_prior**.

| Controller | Mean loss | Tracking passes | Mean action cost | Recovery fraction | Seconds/trial |
|---|---:|---:|---:|---:|---:|
| paired | 0.007767 | 240/240 | 2.9079 | 1.000 | 0.1030 |
| paired_lesion | 0.010467 | 240/240 | 3.0099 | 0.999 | 0.1082 |
| diagonal | 0.008768 | 240/240 | 2.9482 | 1.000 | 0.1047 |
| shuffled | 0.007571 | 240/240 | 2.9192 | 1.000 | 0.1031 |
| dense | 0.005776 | 240/240 | 2.8478 | 1.000 | 0.1020 |
| signed_intensity | 0.007767 | 240/240 | 2.9079 | 1.000 | 0.1046 |
| zero | 0.138770 | 0/240 | 0.0000 | 0.013 | 0.0020 |
| hold | 0.066705 | 0/240 | 2.3320 | 0.128 | 0.0020 |

## Regime dependence

| Regime | Paired minus lesion | 95% interval |
|---|---:|---|
| paired | -0.004004 | [-0.004206, -0.003814] |
| diagonal | -0.001542 | [-0.001622, -0.001468] |
| misaligned | -0.002553 | [-0.002736, -0.002376] |

## Guardrails

- failures: met
- action_cost: met
- return_loss: met
- stockout: met
- overflow: met
- hard_constraints: met
- runtime: met

Observed paired/lesion controller walltime ratio: 0.952; this is machine- and implementation-dependent, not a FLOP count.
Maximum paired/signed-intensity action difference: 9.99e-16.

Both environments are internally designed. Inventory adds finite-capacity clipping and stockouts; the shared planner still uses an affine horizon approximation, a documented model mismatch.
Changing the within-pair planner affects subsequent actions and therefore subsequent learning data; it does not preserve identical fitted weights throughout each paired run.
Return-to-initial-context performance measures reacquisition in this task, not general lifelong capacity retention. No consciousness indicator is inferred from these outcomes.
