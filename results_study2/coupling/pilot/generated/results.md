# Study 2: contextual coupling

Split: **pilot**. 288 complete controller trials.

Primary unit: seed, equally averaging the six task-by-regime cells; lower loss is better.

Paired minus planning lesion: **-0.002711**, 95% paired-seed interval **[-0.003228, -0.002173]**, n=6 seeds.

Predeclared decision: **pilot_not_confirmatory**.

| Controller | Mean loss | Tracking passes | Mean action cost | Recovery fraction | Seconds/trial |
|---|---:|---:|---:|---:|---:|
| paired | 0.007704 | 36/36 | 2.9582 | 1.000 | 0.1088 |
| paired_lesion | 0.010415 | 36/36 | 3.0523 | 1.000 | 0.1146 |
| diagonal | 0.008700 | 36/36 | 2.9936 | 1.000 | 0.1099 |
| shuffled | 0.007581 | 36/36 | 2.9744 | 1.000 | 0.1063 |
| dense | 0.005863 | 36/36 | 2.9099 | 1.000 | 0.1082 |
| signed_intensity | 0.007704 | 36/36 | 2.9582 | 1.000 | 0.1120 |
| zero | 0.138611 | 0/36 | 0.0000 | 0.009 | 0.0021 |
| hold | 0.066380 | 0/36 | 2.3529 | 0.120 | 0.0021 |

## Regime dependence

| Regime | Paired minus lesion | 95% interval |
|---|---:|---|
| paired | -0.004110 | [-0.004881, -0.003292] |
| diagonal | -0.001330 | [-0.001538, -0.001127] |
| misaligned | -0.002692 | [-0.003338, -0.002059] |

## Guardrails

- failures: met
- action_cost: met
- return_loss: met
- stockout: met
- overflow: met
- hard_constraints: met
- runtime: met

Observed paired/lesion controller walltime ratio: 0.950; this is machine- and implementation-dependent, not a FLOP count.
Maximum paired/signed-intensity action difference: 9.99e-16.

Both environments are internally designed. Inventory adds finite-capacity clipping and stockouts; the shared planner still uses an affine horizon approximation, a documented model mismatch.
Changing the within-pair planner affects subsequent actions and therefore subsequent learning data; it does not preserve identical fitted weights throughout each paired run.
Return-to-initial-context performance measures reacquisition in this task, not general lifelong capacity retention. No consciousness indicator is inferred from these outcomes.
