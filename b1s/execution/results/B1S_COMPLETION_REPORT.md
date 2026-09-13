# B1-S — Development execution report

**B1S_DEVELOPMENT_COMPLETE_WITH_LIMITATIONS**

A budget-capped development campaign was executed under a plan fixed before learning. This is neither architectural confirmation nor a B1-E evaluation.

Source commit: `e7fab2dc8076a3605565524939a09dc0e677a2e5`
Selected support: `100000`
Training fits: 381; native training steps: 12484608

## Descriptive competitive outcomes

| Comparator | Candidate | Mean loss | Mean displacement | Fall fraction |
|---|---|---:|---:|---:|
| S-M0 | graph-000000 | 98.4740876 | 1.58666579 | 1 |
| S-M2-fixed | graph-011001 | 78.675254 | -0.184939682 | 0.722222222 |
| S-M3 | graph-100000 | 47.9475975 | 0.681573497 | 0.458333333 |
| S-M4 | graph-111111 | 107.821275 | -0.234802339 | 1 |
| S-M5 | ppo-21 | 52.7367192 | 0.100041449 | 0.486111111 |
| S-M6 | sac-37 | 74.5471336 | 0.511624177 | 0.666666667 |

| Contrast | Delta by independent training replicate | Mean delta | Nominal interval | Development class |
|---|---|---:|---|---|
| S-M3 minus S-M0 | [-93.98590235799524, 3.685329383944108, -61.278897225274704] | -50.5264901 | [-174.02656682289273, 72.97358669000886] | INCONCLUSIVE |
| S-M3 minus S-M2-fixed | [-17.65045925606844, -9.700525187463631, -64.83198478851247] | -30.7276564 | [-104.758681061399, 43.303368240035965] | INCONCLUSIVE |
| S-M3 minus S-M4 | [-104.3351860834971, -7.972820229181607, -67.31302621976162] | -59.8736775 | [-180.62766840814032, 60.88031338651343] | INCONCLUSIVE |
| S-M3 minus S-M5 | [-98.8447869901817, 95.0790032217693, -10.601581076780954] | -4.78912162 | [-245.98018640540633, 236.4019431752774] | INCONCLUSIVE |
| S-M3 minus S-M6 | [-97.87817924655974, -3.7971768249343665, 21.87674782742074] | -26.5995361 | [-183.22380477224016, 130.02473260952456] | INCONCLUSIVE |

Intervals are nominal and exploratory, with only three initializations, common model selection and multiple comparisons. They do not guarantee confirmatory coverage or establish population equivalence.

## Structure and causality

Selected by replicate: `['010101', '011010', '000110']`
Exact-degree alternatives: `[]`
Route-use dependence observed: `True`

The causal assay bypasses one message use after an audited action-prefix replay. Immediate action changes show local computational dependence; subsequent loss is a total policy effect through native physics. This is not a physical causal edge or psychological evidence.

## Comparator competence and limits

```json
{
  "S-M5": {
    "additional_displacement": -1.7089986867374845,
    "basic_task_diagnostic_pass": false,
    "improvement_over_zero_loss": 44.715259197525924,
    "optimality_or_convergence_certified": false
  },
  "S-M6": {
    "additional_displacement": -1.2974159585105047,
    "basic_task_diagnostic_pass": false,
    "improvement_over_zero_loss": 22.904844731232586,
    "optimality_or_convergence_certified": false
  }
}
```

The budget does not establish convergence. Beating a no-action witness does not establish optimality. If comparators fail the diagnostic, no general graph superiority is inferred. Training resources, SAC replay and architecture differences are reported separately.

## Subsequent status

`H_CAT=NOT_EVALUABLE`; `H_TRANSFER=NOT_EVALUATED`; `B1E_executed=false`; `final_seeds_generated=false`.

B1-E remains on hold. These results, precision and alternative competence must be reviewed before designing a new final evaluation. A favorable outcome is not required to finish the development work.

The eight-pair taxonomy, B0, B1-D and historical studies are not changed. Reported information is rendered from this campaign JSON; messages are not consciousness and reward is not pleasure.
