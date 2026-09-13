# B1-S — Informe de ejecución de desarrollo

**B1S_DEVELOPMENT_COMPLETE_WITH_LIMITATIONS**

Se ejecutó una campaña de desarrollo limitada por un presupuesto fijado antes del aprendizaje. No es una confirmación de la arquitectura ni una evaluación B1-E.

Source commit: `e7fab2dc8076a3605565524939a09dc0e677a2e5`
Selected support: `100000`
Training fits: 381; native training steps: 12484608

## Resultados competitivos descriptivos

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

Los intervalos son nominales y exploratorios, con solo tres inicializaciones, selección común de modelo y múltiples comparaciones. No garantizan cobertura confirmatoria ni demuestran equivalencia poblacional.

## Estructura y causalidad

Selected by replicate: `['010101', '011010', '000110']`
Exact-degree alternatives: `[]`
Route-use dependence observed: `True`

La prueba causal retira una sola utilización del mensaje después de un prefijo de acciones reproducido. Cambios inmediatos de acción muestran dependencia computacional local; la pérdida posterior es un efecto total de política a través de la física nativa. No es una arista causal física ni evidencia psicológica.

## Competencia de comparadores y límites

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

El presupuesto no demuestra convergencia. Superar al testigo de inacción tampoco acredita optimalidad. Si los comparadores no cumplen el diagnóstico, no se atribuye superioridad general al grafo. Los recursos de entrenamiento, el replay de SAC y las arquitecturas distintas se informan por separado.

## Estado posterior

`H_CAT=NOT_EVALUABLE`; `H_TRANSFER=NOT_EVALUATED`; `B1E_executed=false`; `final_seeds_generated=false`.

B1-E permanece en pausa. Se requiere revisar estos resultados, la precisión y la competencia de las alternativas antes de diseñar una evaluación final nueva. No se impone un resultado favorable para considerar terminado el trabajo de desarrollo.

La taxonomía de ocho pares, B0, B1-D y los estudios históricos no se modifican. La información procede de los JSON de esta campaña; los mensajes no se denominan consciencia, ni la recompensa placer.
