# Informe de revisión y verificación de prioridades

Revisión del 6 de septiembre de 2026. Alcance: referencia corregida 2.0.1,
regulador contextual 2.1 y evaluación sintética reproducible.

**Se implementaron y verificaron los diez entregables dentro de este alcance.**
La comprobación produjo 38 pruebas automatizadas aprobadas, 40 ejecuciones
P0 y 910 ejecuciones contextuales con trazas. Esto completa
la revisión de ingeniería y la comparación propuesta; no equivale a confirmar
las diez hipótesis científicas ni a demostrar conciencia propia.

La evidencia favorece algunos mecanismos funcionales de memoria, estimación de
capacidades y distribución de recursos en las tareas definidas. La ventaja media
sobre el comparador recurrente es pequeña y depende del indicador: el modelo
polar obtiene menor regret medio, pero peor recuerdo y menos aprobaciones del
criterio externo. El control de coordenadas es equivalente. **No se identificó
una ventaja exclusiva de la polaridad ni evidencia de conciencia o ASI.**

## Evidencia reproducible y alcance ejecutado

| Evidencia | Resultado verificado | Artefacto |
|---|---|---|
| Pruebas de implementación | 38 aprobadas; incluyen mecanismos, errores, replay, aislamiento del objetivo oculto y regeneración de informes | [Registro completo](validation_tests.txt) |
| Referencia P0 | 40 ejecuciones: 8 escenarios × 5 semillas, 2025–2029, más controles emparejados cuando aplican | [Índice y configuración](../results_corrected/p0/index.json), [resúmenes](../results_corrected/p0_reports/p0_summaries.csv) |
| Regulador contextual | 910 ejecuciones: 2 familias de tareas × 35 semillas × 13 controladores, 96 pasos por ejecución | [Manifiesto](../results_corrected/contextual/manifest.json), [resultados por ejecución](../results_corrected/contextual/generated/per_run.csv) |
| Partición de evaluación | 5 semillas de desarrollo y 30 semillas heldout; cada controlador tiene 60 ejecuciones heldout | [Protocolo](EVALUATION_PROTOCOL.md), [protocolo de la ejecución](../results_corrected/contextual/protocol.json) |
| Integridad y trazabilidad | Hashes del protocolo, implementación y trazas; semilla, configuración y versiones registradas | [Manifiesto contextual](../results_corrected/contextual/manifest.json) |
| Informes desde datos | CSV, JSON, Markdown, LaTeX y figuras regenerados a partir de las trazas completas | [Resultados generados](../results_corrected/contextual/generated/results.md), [figura](../results_corrected/contextual/generated/evaluation_summary.png) |

Las 910 ejecuciones no son 910 réplicas independientes para el contraste
principal. Se promedian las dos tareas con igual peso dentro de cada semilla,
y se comparan **30 semillas emparejadas**. Las series temporales tampoco se
tratan como réplicas independientes. Los escenarios P0 son descriptivos y no se
agregan a ese contraste estadístico.

## Diez prioridades: implementación, evidencia y límite

| Orden / prioridad | Cambio completado en esta revisión | Evidencia concreta | Límite de la conclusión |
|---|---|---|---|
| **1 / P0** | Especificación versionada con ecuaciones, significado/rangos, retención de memoria, tiempo, activación, métricas y N frente a ocho tipos; original conservado | [Especificación general](../SPECIFICATION.md), [referencia corregida](SPEC_LEGACY_CORRECTED.md), [modelo contextual](SPEC_CONTEXTUAL.md), [original archivado](../legacy/v2_0/) | Los dos motores son modelos distintos y están identificados como tales. Una especificación coherente no valida por sí sola la teoría de conciencia. |
| **2 / P0** | Mini-IACL usa estados reales antes del paso; errores explícitos para estados ausentes; polos del estímulo tienen signos opuestos; P7 recibe el pulso registrado; SAT=.80 es alcanzable con clip de tensión 1.2 | [Experimentos corregidos](EXPERIMENTS_P0_CORRECTED.md), [pruebas de experimentos](../tests/test_experiments_corrected.py), [pruebas del motor](../tests/test_engine_corrected.py) | El mapeo textual sigue siendo una regla léxica; Mini-IACL conserva consenso como referencia histórica. Ninguno constituye comprensión del lenguaje o integración consciente. |
| **3 / P0** | Trazas completas, semillas/configuración, controles emparejados y reportes regenerables; modulación, override, efecto del override, guards y clipping separados; recuperación sostenida y censura explícita | [Resumen P0 de 40 ejecuciones](../results_corrected/p0_reports/p0_summaries.csv), [diferencias originales/corregidas](../results_corrected/comparison/legacy_revision_deltas.md), [pruebas de reporte](../tests/test_experiments_corrected.py) | Los datos históricos ausentes no se reconstruyen como cero. Los cambios acumulados de versión no son efectos causales aislados. |
| **4 / P1** | Evaluación externa fijada antes de la ejecución completa: regret frente a oracle factible, seguimiento, discriminación, costos, recuperación, recuerdo y restricciones; descriptores de semejanza separados | [Protocolo](EVALUATION_PROTOCOL.md), [reglas ejecutables](evaluation_protocol.json), [resultados](../results_corrected/contextual/generated/results.md) | Los controles cero, uniforme y desconectado fallan en 0/60 aprobaciones cada uno; la regla de éxito es una convención de estas tareas, no un test validado de conciencia. |
| **5 / P1** | Dos polos independientes, catálogo operacional de proxies, inactividad/predominio/coactivación distintos; reversión de signos con estado aprendido y control firmado/intensidad | [Catálogo y ecuaciones](SPEC_CONTEXTUAL.md), [pruebas contextuales](../tests/test_contextual_model.py), [contraste de coordenadas](../results_corrected/contextual/generated/paired_effects.csv) | Las interpretaciones psicológicas son etiquetas sintéticas aún sin validación empírica. La equivalencia de coordenadas impide atribuir una ventaja a la mera notación polar. |
| **6 / P1** | Objetivos, prioridades, presupuesto, acciones admisibles y escala de respuesta cambian con el entorno; coordinación por recursos compartidos; reacquisición y cambio de orientación | [Especificación contextual](SPEC_CONTEXTUAL.md), [familias de tareas](EVALUATION_PROTOCOL.md), [prueba de acceso a ambos polos](../tests/test_contextual_model.py) | El horizonte es un multiplicador de velocidad de respuesta, no planificación predictiva de varios pasos. Las tareas suministran los objetivos; no se demostró generación autónoma de propósitos. |
| **7 / P1** | Clipping numérico, seguimiento de metas, estimación de efectos y prohibiciones/presupuesto tienen funciones y trazas propias; HGI/INC no definen ética | [Especificaciones](../SPECIFICATION.md), [pruebas de estabilidad/admisibilidad](../tests/test_contextual_model.py), [trazas evaluadas](../results_corrected/contextual/generated/aggregate.csv) | Se verificó cumplimiento de restricciones formales del entorno. No se implementó razonamiento ético, ni una garantía general de estabilidad para todos los parámetros. |
| **8 / P2** | Memoria persistente indexada por claves, aprendizaje con observación visible, decaimiento de influencia, recuperación, edición/borrado/permutación; intervención después de retirar el objetivo visible | [Memoria y reglas de actualización](SPEC_CONTEXTUAL.md), [pruebas causales de memoria](../tests/test_contextual_model.py), [ablaciones](../results_corrected/contextual/generated/paired_effects.csv) | Se recuerda un patrón asociado a una clave conocida; no memoria autobiográfica, aprendizaje semántico abierto ni reconsolidación clínica. Un shuffle puede ser identidad y se registra su efecto real. |
| **9 / P2** | Workspace distribuye presupuesto según demanda; modelo de capacidades aprende de acciones y efectos, estima incertidumbre y modifica decisiones; lesiones selectivas y controles de abundancia de recursos | [Workspace/modelo propio](SPEC_CONTEXTUAL.md), [pruebas causales](../tests/test_contextual_model.py), [comparaciones](../results_corrected/contextual/generated/results.md) | Workspace es un asignador funcional, y el modelo propio es un estimador heurístico de ganancias por canal. No equivalen a un espacio global neuronal validado ni a un yo fenomenológico. |
| **10 / P2** | Comparadores emparejados, ablaciones, intervenciones, múltiples semillas, variantes heldout, tamaños de efecto e intervalos; conclusiones incluyen resultados favorables, adversos y nulos | [Resultados completos](../results_corrected/contextual/generated/results.md), [efectos emparejados](../results_corrected/contextual/generated/paired_effects.csv), [registro de aceptación](PRIORITY_ACCEPTANCE.md) | Las variantes heldout pertenecen a las mismas dos familias generadoras; no prueban generalización a dominios nuevos. No se demostró causalidad exclusiva de la organización polar. |

“Completado” significa que existe implementación identificable y una comprobación
del criterio operacional indicado. Donde la aspiración original era más amplia
que la capacidad implementada —ética, episodios abiertos, autoconciencia o tareas
externas independientes— el límite permanece abierto y se declara en la tabla.

## Resultado principal y resultados que lo limitan

La comparación primaria usa regret de error cuadrático ponderado respecto de
un oracle restringido a los mismos límites de acción y recursos. El oracle es
externo al controlador y utiliza la ganancia nominal; no tiene acceso al ruido
futuro. Un pequeño regret realizado negativo es posible por ruido y no se recorta.

**Diferencia dual_pole − recurrent:**
`-0.000565122`, intervalo bootstrap del 95 %
`[-0.000699750, -0.000427399]`,
`dz = -1.462606`, con `30` semillas emparejadas y
`2000` remuestreos. Menor regret es mejor. La diferencia
absoluta es pequeña en estas unidades normalizadas; un dz grande frente a la
variabilidad entre semillas no la convierte en una mejora universal.

| Controlador | Regret medio ↓ | Regret de recuerdo ↓ | Aprobaciones externas |
|---|---:|---:|---:|
| `dual_pole` | 0.004014 | 0.012783 | 58/60 |
| `signed_intensity` | 0.004014 | 0.012783 | 58/60 |
| `recurrent` | 0.004579 | 0.011477 | 60/60 |
| `no_memory` | 0.017071 | 0.162410 | 30/60 |
| `no_workspace` | 0.022096 | 0.012783 | 28/60 |
| `no_self_model` | 0.007976 | 0.010688 | 59/60 |
| `memory_erase` | 0.017071 | 0.162410 | 30/60 |
| `memory_shuffle` | 0.012058 | 0.105907 | 38/60 |
| `workspace_shuffle` | 0.031499 | 0.012811 | 28/60 |
| `self_model_reset` | 0.004301 | 0.012783 | 58/60 |
| `zero` | 0.174099 | 0.166722 | 0/60 |
| `uniform` | 0.083175 | 0.081209 | 0/60 |
| `disconnected` | 0.106836 | 0.107499 | 0/60 |

Esta tabla utiliza únicamente heldout. El regret medio combina 60 ejecuciones
por controlador; el recuerdo se calcula en las 30 ejecuciones de
`switching_memory`, porque no aplica a la otra familia. Las aprobaciones son
conteos descriptivos del criterio conjuntivo, no una segunda afirmación de
significancia estadística.

El modelo dual aprueba 58/60 ejecuciones y el recurrente 60/60. Los dos fallos
del dual ocurren en `switching_memory`; su regret de recuerdo es mayor que el
del recurrente. Por eso **no corresponde afirmar que el dual domina al
comparador en todos los objetivos**. Eliminar el modelo propio empeora el regret
medio, pero mejora el recuerdo y alcanza 59/60 aprobaciones: su utilidad también
depende de la tarea y del indicador, no de una necesidad universal de ese módulo.

El control `signed_intensity` produce el mismo comportamiento dentro del error
numérico. Su diferencia media de regret es
`-3.036e-19`; el tamaño estandarizado es
indefinido y se informa como NA. La transformación conserva toda la información
entre dos actividades y orientación/intensidad. No es una teoría competidora
independiente ni un descubrimiento de superioridad de unas coordenadas.

Las ablaciones y lesiones de memoria y asignación de recursos empeoran varios
resultados en estas condiciones, coherentemente con sus funciones programadas.
Los intervalos de contrastes adicionales son **exploratorios y sin ajuste por
multiplicidad**. El comparador recurrente comparte dimensiones, observaciones,
memoria, estimador de capacidades, asignador de recursos, restricciones y un paso
de actualización; difiere la regla de actualización (gradiente proyectado frente
a planificación inversa). Las operaciones aritméticas y los FLOPs no son idénticos.
La diferencia primaria identifica esa comparación concreta; no aísla una causa
exclusivamente polar.

## Cambios respecto del artículo y resultados originales

La referencia original conserva archivos y resultados históricos. La nueva
versión no intenta mantener los valores publicados mediante ajustes de métricas.
La [comparación regenerada de extremos](../results_corrected/comparison/legacy_revision_deltas.md)
para semilla 2025 muestra, por ejemplo, que el INC final de P7 cambia de 0.9870
a 0.9514, y HGI de 0.9992 a 0.9949. Cambiaron el calendario efectivo del estímulo,
el momento de medición de INC y otros mecanismos documentados; esas diferencias
no permiten asignar el cambio a un único factor.

El antiguo escenario llamado “trauma” es una perturbación externa persistente,
con una variante que reduce programáticamente esa entrada. Se presenta con esa
interpretación. La nueva memoria interna tiene otras variables y experimentos;
no se usa para reinterpretar retroactivamente aquella perturbación como evidencia
de trauma, experiencia subjetiva o reconsolidación.

Los resultados P0 muestran que modulación continua, override, correcciones de
métricas y clipping pueden comportarse de manera diferente. Por ejemplo, P7
base con semilla 2025 registra 100 % de pasos con α<1, 0 % de override, 25 %
con guard HGI, 62.5 % con guard INC y 29.1667 % con clipping. Su recuperación
queda censurada al terminar el registro: **no es cero pasos ni recuperación
demostrada**. Los porcentajes provienen del
[resumen trazable P0](../results_corrected/p0_reports/p0_summaries.csv).

## Límites científicos y continuación justificada

1. El protocolo está congelado internamente, con hash antes de la ejecución
   completa; no es un registro público prospectivo independiente. La prueba
   mecánica inicial usó las primeras dos semillas de cada partición, incluidas
   heldout. Esa exposición está declarada; no se hizo selección de parámetros
   basada en resultados, pero heldout no fue completamente ciego.
2. Ambas familias de tareas y sus mecanismos fueron diseñados en el mismo
   proyecto. Variantes nuevas de semillas, ganancias y prioridades prueban una
   transferencia limitada dentro de esas familias, no una tarea independiente
   diseñada por terceros ni un dominio real.
3. La memoria recibe una clave simbólica conocida y almacena objetivos vistos.
   El estimador propio aprende razones efecto/acción por canal; ruido y clipping
   pueden sesgarlo. No aprende un modelo general del mundo ni identidad personal.
4. La asignación global no redistribuye todo presupuesto sobrante; el horizonte
   no simula planes futuros; las restricciones son máscaras booleanas, no una
   representación completa de conflictos complejos. Los ocho nombres no poseen
   validación neurocientífica o psicológica en estas pruebas.
5. Estabilidad de valores, cumplimiento de restricciones, coordinación,
   persistencia de memoria y capacidad predictiva son funciones verificadas en
   contextos concretos. Ninguna establece por sí misma experiencia subjetiva,
   vida, ASI o un principio universal de polaridad.

Hay una base verificable para continuar con **investigación de regulación
contextual y memoria**, siempre que el siguiente estudio incorpore tareas
independientes, comparadores recurrentes y adaptativos más amplios, presupuesto
computacional controlado y criterios definidos antes de observar resultados.
El valor de esta revisión es transformar afirmaciones ambiguas en modelos,
pruebas y resultados refutables. La meta de conciencia artificial sigue siendo
una hipótesis abierta, y esta evidencia no permite prometer que se alcanzará.
