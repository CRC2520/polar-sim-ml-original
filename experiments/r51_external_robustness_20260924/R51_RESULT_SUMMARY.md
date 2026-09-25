# R51 result — fresh external robustness battery

Frozen final resolution:

`R51_EXTERNAL_ROBUSTNESS_BATTERY_PASS`

## Canonical execution

- run: `36095588879`
- scientific head SHA: `59f355b5a6152fa3435b4db8896a18089f80a591`
- external source: `proroklab/popgym@e397e5eac9965f9963d18c9f455cd1983bca14fb`

Development:
- AutoencodeMedium
- RepeatPreviousMedium
- seeds 2216001–2216016
- seed guards: **16/16** (required 13/16)

Confirmation:
- AutoencodeHard
- RepeatPreviousHard
- seeds 2217001–2217032
- seed guards: **32/32** (required 28/32)

Integrity: PASS.

## Confirmatory medians

### AutoencodeHard

| Perturbation | FULL accuracy | D effect | C effect | R effect |
|---|---:|---:|---:|---:|
| CLEAN | 1.0000 | +0.7500 | +0.7436 | +0.7500 |
| NOISE_05 | 0.9487 | +0.6987 | +0.6939 | +0.6987 |
| OCCLUSION_10 | 0.9263 | +0.6763 | +0.6715 | +0.6811 |
| STALE_05 | 0.9631 | +0.7131 | +0.7067 | +0.7083 |
| SEMANTIC_SHIFT_HALF | 0.5000 | +0.2500 | +0.2500 | +0.2500 |

CLEAN GENERIC_ISO:
- accuracy gap = 0;
- action agreement = 1.0.

### RepeatPreviousHard

| Perturbation | FULL accuracy | D effect | C effect | R effect |
|---|---:|---:|---:|---:|
| CLEAN | 1.0000 | +0.7527 | +0.7527 | +0.7554 |
| NOISE_05 | 0.9511 | +0.7065 | +0.7065 | +0.6984 |
| OCCLUSION_10 | 0.9348 | +0.6848 | +0.6848 | +0.6821 |
| STALE_05 | 0.9620 | +0.7174 | +0.7174 | +0.6712 |
| SEMANTIC_SHIFT_HALF | 0.8478 | +0.6005 | +0.6005 | +0.6087 |

CLEAN GENERIC_ISO:
- accuracy gap = 0;
- action agreement = 1.0.

## Interpretation

R51 closes a bounded external-robustness gap that remained after R37/R38.

Across two fresh external sequence POMDP families, the intact functional
organization remains capable under mild symbol noise, occlusion and stale
observations, while D/C/R lesions retain large causal effects.

The unannounced semantic recoding shift is the strongest perturbation:
Autoencode falls to 0.50, whereas RepeatPrevious remains at 0.848. Both still
retain positive D/C/R effects above the frozen threshold.

This is bounded robustness evidence for functional D+C+R-like sequence
organization. It does not overturn the historical R37/R38 failures, which used
a different recurrent observer and different external control tasks.

## Boundaries

R51 does not establish:
- neural learnability;
- global minimality;
- E6b;
- E7;
- higher-order metacognitive/planning integration;
- AGI/ASI;
- phenomenal consciousness.
