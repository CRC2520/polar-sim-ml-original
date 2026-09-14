# Adjudicación de B1-S v1.1 y disposición hacia B1-E

**ADJUDICATION_COMPLETE**

Fuente científica: `50c4e08cf3d69c998ce95b8da1e1647a09b31339`
Publicación de resultados: `de35a0995d96ff548772fa56c3620466495add72`

Este documento adjudica el desarrollo ya ejecutado. No es un prerregistro, una réplica independiente ni una ejecución B1-E; no cambia los resultados ni los umbrales archivados.

## Decisión

`ON_HOLD_FOR_CURRENT_STRUCTURAL_CONFIRMATION`

Motivos pendientes: ["STRUCTURAL_STABILITY_DIAGNOSTIC_NOT_PASSED", "GENERIC_COMPARATOR_COMPETENCE_NOT_PASSED"]

Los diagnósticos de estabilidad y competencia se mantienen exactamente como fueron fijados. El signo favorable de una pérdida no es requisito automático de paso ni certifica por sí solo una ventaja. La disposición se refiere al contraste actual de una estructura estable con rutas activas frente a alternativas competentes; no prohíbe una futura confirmación de un resultado adverso, si su hipótesis se especifica por separado.

## Resultados adjudicados

Soporte seleccionado: `000001`
Ganadores por réplica: ["010101", "100100", "001000", "110010", "011000", "001101"]
Frecuencia modal / Jaccard medio: 0.166666667 / 0.202222222
Diagnóstico de estabilidad: False

| Comparador | Pérdida media | Desplazamiento | Competencia |
|---|---:|---:|---|
| S-M0 | -1.00394293 | 0.565962508 | False |
| S-M2-fixed | 5.70871184 | 0.619157508 | False |
| S-M3 | 1.53069901 | 2.19718704 | False |
| S-M4 | -5.07249815 | 2.60690617 | False |
| S-M5 | 25.345387 | 1.56178281 | False |
| S-M6 | 4.25314383 | 0.0887630383 | False |

| Contraste: S-M3 menos alternativa | Media | Intervalo nominal | Clasificación conservada |
|---|---:|---|---|
| S-M0 | 2.53464195 | [-7.896817934856722, 12.966101826653999] | INCONCLUSIVE |
| S-M2-fixed | -4.17801282 | [-13.464064224438397, 5.108038581003251] | INCONCLUSIVE |
| S-M4 | 6.60319716 | [-3.3836980771091243, 16.590092396428833] | INCONCLUSIVE |
| S-M5 | -23.814688 | [-62.141211321327376, 14.511835253597066] | INCONCLUSIVE |
| S-M6 | -2.72244482 | [-17.444926389317786, 12.000036752876547] | INCONCLUSIVE |

Los intervalos son exploratorios sobre seis réplicas independientes de entrenamiento; no están corregidos para afirmaciones confirmatorias múltiples. Un control idéntico no cuenta como evidencia adicional. No superar un margen no prueba equivalencia ni efecto nulo.

## Contribución causal y alcance

```json
{
  "available_prefixes": 48,
  "unavailable_prefixes": 0,
  "tested_edge_interventions": 48,
  "route_use_identified": true,
  "external_difference_by_rep": [
    0.010807979408127777,
    0.0,
    0.0,
    -20.710789252480016,
    0.0,
    0.0
  ]
}
```

El uso local de una ruta y su efecto externo son preguntas distintas. Se conservan los prefijos no disponibles, sin contarlos como lesiones nulas. El presupuesto ambiental común no iguala algoritmos, parámetros, memoria o número de actualizaciones.

## Frontera de B1-E

`ready_for_b1e_freeze=false`; `ready_for_b1e_confirmatory_run=false`; `B1E_executed=false`; `final_seeds_generated=false`.

La campaña de desarrollo y su adjudicación pueden terminar aunque el paso a confirmación quede en pausa. Para B1-E aún se requiere un protocolo nuevo que identifique el estimando, contraste, precisión/tamaño muestral, reglas de datos faltantes, recursos y custodia de semillas antes de su ejecución. No se hereda el tamaño del antiguo confirmatorio de P7. Esta adjudicación no autoriza nuevos entrenamientos o reajustes automáticos.

## No afirmaciones

`H_CAT=NOT_EVALUABLE`; `H_TRANSFER=NOT_EVALUATED`.

No se valida el catálogo de ocho polaridades, la arquitectura completa, consciencia, ASI ni una ley física. La versión anterior y sus cifras se conservan. La huella de integridad acredita correspondencia documental de los archivos examinados; no reproduce todo el entrenamiento ni certifica por sí sola cada supuesto científico.
