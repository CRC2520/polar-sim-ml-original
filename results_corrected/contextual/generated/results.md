# Frozen synthetic benchmark

Protocol SHA-256: `21ca01d0de32c1c0294da6d4025470e93a8da0cd943e07d2c3832f2860608426`

All tables regenerated from checksummed per-step traces. Negative differences favor dual_pole. Missing values are NA, not zero.

| Controller | Mean regret | RMSE | Recall regret | External pass fraction |
|---|---:|---:|---:|---:|
| dual_pole | 0.00401 | 0.10720 | 0.01278 | 0.96667 |
| signed_intensity | 0.00401 | 0.10720 | 0.01278 | 0.96667 |
| recurrent | 0.00458 | 0.10880 | 0.01148 | 1.00000 |
| no_memory | 0.01707 | 0.16387 | 0.16241 | 0.50000 |
| no_workspace | 0.02210 | 0.15220 | 0.01278 | 0.46667 |
| no_self_model | 0.00798 | 0.11618 | 0.01069 | 0.98333 |
| memory_erase | 0.01707 | 0.16387 | 0.16241 | 0.50000 |
| memory_shuffle | 0.01206 | 0.14413 | 0.10591 | 0.63333 |
| workspace_shuffle | 0.03150 | 0.17031 | 0.01281 | 0.46667 |
| self_model_reset | 0.00430 | 0.10811 | 0.01278 | 0.96667 |
| zero | 0.17410 | 0.42821 | 0.16672 | 0.00000 |
| uniform | 0.08317 | 0.30437 | 0.08121 | 0.00000 |
| disconnected | 0.10684 | 0.34124 | 0.10750 | 0.00000 |

| Contrast (dual_pole minus comparator) | Regret difference | 95% CI | Paired dz |
|---|---:|---|---:|
| signed_intensity | -0.00000 | [-0.00000, 0.00000] | NA |
| recurrent | -0.00057 | [-0.00070, -0.00043] | -1.46261 |
| no_memory | -0.01306 | [-0.01349, -0.01262] | -10.45104 |
| no_workspace | -0.01808 | [-0.01821, -0.01795] | -48.31897 |
| no_self_model | -0.00396 | [-0.00404, -0.00388] | -16.51080 |
| memory_erase | -0.01306 | [-0.01349, -0.01262] | -10.45104 |
| memory_shuffle | -0.00804 | [-0.00976, -0.00616] | -1.56039 |
| workspace_shuffle | -0.02748 | [-0.02861, -0.02639] | -8.61105 |
| self_model_reset | -0.00029 | [-0.00030, -0.00027] | -6.44910 |
| zero | -0.17008 | [-0.17347, -0.16678] | -17.86644 |
| uniform | -0.07916 | [-0.07936, -0.07896] | -135.96973 |
| disconnected | -0.10282 | [-0.10536, -0.10035] | -14.59891 |

Primary paired mean-regret difference, dual_pole minus recurrent: -0.00057; 95% seed-bootstrap interval [-0.00070, -0.00043]; paired dz -1.46261; n=30 paired seeds.

Signed-intensity is an exact coordinate control, not an independent competing theory. Recurrent shares state size, observations, memory, capability estimation, workspace, constraints, and one update per step; its projected-gradient rule differs from inverse-planning. Any difference is a rule comparison and cannot be attributed uniquely to polarity.

All other contrasts are exploratory unadjusted intervals, without confirmatory significance claims. Recovery means exclude censored events and must be read with recovery_fraction and censor counts. Heldout tasks share the same generator family; this is not broad out-of-distribution generalization.

These results test functional mechanisms in an engineered synthetic controller. They do not establish consciousness, sentience, ASI, or universality of polarity.
