# B1-S v1.1 — Estabilidad y competencia de comparadores

**B1S_V1_1_DEVELOPMENT_COMPLETE_WITH_LIMITATIONS**

Campaña de desarrollo; no es confirmación ni ejecución B1-E.

Commit científico: `50c4e08cf3d69c998ce95b8da1e1647a09b31339`
Soporte seleccionado: `000001`
Entrenamientos y pasos nativos: 312 / 62128128

## Estabilidad

```json
{
  "conditional_edgecount_null_mean_jaccard": 0.2581851851851853,
  "counts": {
    "001000": 1,
    "001101": 1,
    "010101": 1,
    "011000": 1,
    "100100": 1,
    "110010": 1
  },
  "edges": [
    {
      "count": 2,
      "edge": [
        0,
        1
      ],
      "frequency": 0.3333333333333333,
      "wilson_nominal_95": [
        0.09677141110578041,
        0.700006684861608
      ]
    },
    {
      "count": 3,
      "edge": [
        0,
        2
      ],
      "frequency": 0.5,
      "wilson_nominal_95": [
        0.1876163064826506,
        0.8123836935173494
      ]
    },
    {
      "count": 3,
      "edge": [
        1,
        0
      ],
      "frequency": 0.5,
      "wilson_nominal_95": [
        0.1876163064826506,
        0.8123836935173494
      ]
    },
    {
      "count": 3,
      "edge": [
        1,
        2
      ],
      "frequency": 0.5,
      "wilson_nominal_95": [
        0.1876163064826506,
        0.8123836935173494
      ]
    },
    {
      "count": 1,
      "edge": [
        2,
        0
      ],
      "frequency": 0.16666666666666666,
      "wilson_nominal_95": [
        0.030053369748306635,
        0.5635028221864702
      ]
    },
    {
      "count": 2,
      "edge": [
        2,
        1
      ],
      "frequency": 0.3333333333333333,
      "wilson_nominal_95": [
        0.09677141110578041,
        0.700006684861608
      ]
    }
  ],
  "engineering_stability_diagnostic": false,
  "mean_jaccard": 0.20222222222222225,
  "modal_fraction": 0.16666666666666666,
  "n_independent_training_initializations": 6,
  "pairwise_jaccard": [
    [
      1.0,
      0.25,
      0.0,
      0.2,
      0.25,
      0.5
    ],
    [
      0.25,
      1.0,
      0.0,
      0.25,
      0.0,
      0.25
    ],
    [
      0.0,
      0.0,
      1.0,
      0.0,
      0.5,
      0.3333333333333333
    ],
    [
      0.2,
      0.25,
      0.0,
      1.0,
      0.25,
      0.0
    ],
    [
      0.25,
      0.0,
      0.5,
      0.25,
      1.0,
      0.25
    ],
    [
      0.5,
      0.25,
      0.3333333333333333,
      0.0,
      0.25,
      1.0
    ]
  ],
  "scope": "Descriptive engineering thresholds fixed before v1.1; Wilson intervals assume independent training selections and are marginal, not simultaneous. Count-conditioned null is a reference, not a catalogue or symmetry p-value.",
  "winners": [
    "010101",
    "100100",
    "001000",
    "110010",
    "011000",
    "001101"
  ]
}
```

Los umbrales son diagnósticos prospectivos de ingeniería; seis selecciones no establecen estabilidad poblacional ni un catálogo universal.

## Comparación externa reservada de desarrollo

| Comparador | Pérdida media | Desplazamiento | Fracción con caída |
|---|---:|---:|---:|
| S-M0 | -1.00394293 | 0.565962508 | 0.00520833333 |
| S-M2-fixed | 5.70871184 | 0.619157508 | 0.046875 |
| S-M3 | 1.53069901 | 2.19718704 | 0.078125 |
| S-M4 | -5.07249815 | 2.60690617 | 0.0260416667 |
| S-M5 | 25.345387 | 1.56178281 | 0.286458333 |
| S-M6 | 4.25314383 | 0.0887630383 | 0.0208333333 |
| no-action | 97.4940751 | 1.79310973 | 1 |

| Contraste S-M3 menos alternativa | Media | Intervalo nominal | Clase exploratoria |
|---|---:|---|---|
| S-M0 | 2.53464195 | [-7.896817934856722, 12.966101826653999] | INCONCLUSIVE |
| S-M2-fixed | -4.17801282 | [-13.464064224438397, 5.108038581003251] | INCONCLUSIVE |
| S-M4 | 6.60319716 | [-3.3836980771091243, 16.590092396428833] | INCONCLUSIVE |
| S-M5 | -23.814688 | [-62.141211321327376, 14.511835253597066] | INCONCLUSIVE |
| S-M6 | -2.72244482 | [-17.444926389317786, 12.000036752876547] | INCONCLUSIVE |

Los intervalos corresponden a seis medias por entrenamiento independiente, no a pasos ni episodios independientes; no están corregidos para una conclusión confirmatoria familiar.

## Competencia y causalidad

```json
{
  "S-M0": {
    "additional_displacement": -1.2271472265323002,
    "convergence_or_optimality_certified": false,
    "descriptive_competence_pass": false,
    "loss_improvement_over_no_action": 98.498018057746
  },
  "S-M2-fixed": {
    "additional_displacement": -1.173952226837476,
    "convergence_or_optimality_certified": false,
    "descriptive_competence_pass": false,
    "loss_improvement_over_no_action": 91.7853632901298
  },
  "S-M3": {
    "additional_displacement": 0.404077301422755,
    "convergence_or_optimality_certified": false,
    "descriptive_competence_pass": false,
    "loss_improvement_over_no_action": 95.96337611184737
  },
  "S-M4": {
    "additional_displacement": 0.8137964357932408,
    "convergence_or_optimality_certified": false,
    "descriptive_competence_pass": false,
    "loss_improvement_over_no_action": 102.56657327150722
  },
  "S-M5": {
    "additional_displacement": -0.23132692277431488,
    "convergence_or_optimality_certified": false,
    "descriptive_competence_pass": false,
    "loss_improvement_over_no_action": 72.1486880779822
  },
  "S-M6": {
    "additional_displacement": -1.704346696535746,
    "convergence_or_optimality_certified": false,
    "descriptive_competence_pass": false,
    "loss_improvement_over_no_action": 93.24093129362674
  }
}
```

Uso de ruta observado: True
Diferencia externa de lesión por réplica: [0.010807979408127777, 0.0, 0.0, -20.710789252480016, 0.0, 0.0]

Un cambio inmediato de acción prueba dependencia computacional en el estado inspeccionado, no utilidad universal, polaridad psicológica o consciencia. La discrepancia posterior de pérdida es un efecto total de política. Se preservan prefijos no disponibles y resultados adversos.

## Procedencia, traducción y límites

`H_CAT=NOT_EVALUABLE`; `H_TRANSFER=NOT_EVALUATED`; `B1E_executed=false`; `final_seeds_generated=false`.

Se conserva B1-S v1. La nueva selección y las inicializaciones competitivas están separadas. No se reajusta el modelo tras abrir los datos reservados. Los JSON y esta traducción comparten cifras, identificadores y decisiones; las claves técnicas permanecen sin traducir para conservar trazabilidad.

Un diagnóstico de competencia fallido o una estructura inestable no se corrigen alterando la tarea. B1-E permanece en pausa hasta una decisión posterior explícita. Los presupuestos finitos, algoritmos y recursos diferentes limitan los claims; no se declara superioridad general.
