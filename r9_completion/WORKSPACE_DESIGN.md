# R9: espacio de trabajo tipado, memoria factual y metas revisables

Versión de la fuente: `R9.workspace.1`. Esta es una realización operacional nueva. No modifica P1 ni R8 y no convierte sus resultados negativos en positivos. Los mecanismos descritos no prueban conciencia fenomenológica, motivación moral intrínseca ni descubrimiento autónomo de valores.

## Contrato público e integración

`CognitiveWorkspace` recibe exclusivamente la observación pública actual de tres componentes `[reserva, salud, señal de demanda]`, el comando de acción propio enviado en la transición anterior, una predicción emitida anteriormente y una magnitud de incertidumbre suministrada por el agente. No recibe entorno, estado latente, dominio, cola física, cinta de azar, resultado contrafactual ni `info_eval`. El comando enviado puede diferir de la acción física efectiva: después de la muerte absorbente, el entorno ejecuta acción0 aunque reciba otro comando. Las asociaciones describen consecuencias factuales del comando bajo esa dinámica, sin consultar esa información privada para corregir etiquetas.

```python
w = CognitiveWorkspace(history=16)
message = w.observe(obs_t, prev_action=a_previous,
                    prev_prediction=prediction_previous, uncertainty=uncertainty)
v = viability_consumer(route_message(message, 'intact', 'viability'))
r = resource_consumer(route_message(message, 'intact', 'resources'))
```

El mensaje es un `WorkspaceMessage` inmutable compuesto por tres tipos:

| Tipo | Contenido | Interpretación |
|---|---|---|
| `StateEstimate` | observación3, pendiente3, variabilidad3, clave de contexto3, incertidumbre3, error de predicción3, longitud de historia | Resumen de observaciones propias disponibles antes de elegir la acción actual. |
| `ActionConsequences` | `delayed_estimates[4,3,3]`, `counts[4,3]`, horizontes `(1,4,12)` | Predicciones absolutas de observaciones futuras por acción e intervalo. Sin experiencia, el contenido es persistencia de la observación actual. |
| `GoalState` | nombre operacional, pesos3, presiones3, edad y número de revisión | Prioridad persistente de supervivencia, reserva y tarea bajo normas impuestas. |

La cabecera registra episodio, tiempo local, tiempo absoluto, número de pendientes y revisión acumulada de memoria. `to_dict()` permite auditoría JSON. `payload_vector()` determina la anchura y bytes exactos del contenido numérico. Los eventos de aprendizaje se registran aparte y no entran al consumidor como canal adicional.

Los consumidores son funciones independientes. No acceden al objeto de memoria, al entorno, a la salida del otro consumidor ni al futuro. El de viabilidad utiliza el horizonte1 de salud y la tendencia observada; devuelve `energy_forecast[4]`, `energy_floor=.2` e `uncertainty`. El de recursos utiliza por igual los horizontes4 y12 de reserva; devuelve `resource_forecast[4]` y los pesos persistentes `[supervivencia,reserva,tarea]`.

Para cada horizonte, una estimación asociativa con `n` observaciones se mezcla mediante `n/(n+3)` con una extrapolación de la historia propia. La variabilidad histórica y la incertidumbre suministrada son heurísticas: no constituyen intervalos de confianza ni garantías de viabilidad. La incertidumbre adicional de salud es su desviación histórica dividida por la raíz del número de observaciones.

`NativeAgent` utiliza realmente ambas salidas: los pronósticos de salud participan en el filtro de acciones y la puntuación; los de reserva y los pesos participan en la puntuación. Los coeficientes, umbrales y objetivo son decisiones del diseñador. La prueba funcional local usa el combinador real del agente, no un segundo actor artificial.

## Memoria y cronología

La historia de trabajo contiene hasta16 filas `[reserva,salud,demanda,acción anterior]`; el primer registro sin acción lleva `-1`. Pendiente y variabilidad se calculan sólo sobre observaciones presentes y pasadas. La clave contextual utiliza discretizaciones fijadas: reserva con cortes `.3,.65`, salud `.25,.6` y demanda `.33,.67`. Estas categorías no se presentan como descubiertas por el sistema.

Al observar el estado en tiempo `t`, la acción anterior crea un evento cuyo origen es la observación `t−1`. Cada horizonte madura una sola vez, cuando está disponible su observación real de destino. La etiqueta es el cambio observado respecto del origen, no un resultado obtenido consultando el simulador. La memoria guarda medias de cambios por `(clave, acción, horizonte)`, con hasta324 entradas posibles. Hay como máximo11 eventos pendientes después de procesar una observación y16 filas de historia. No se mantienen etiquetas pendientes entre episodios.

El estimando asociativo es el cambio factual bajo la continuación de la política realmente ejecutada. A horizontes4 y12 incorpora acciones posteriores y cambios ambientales. No identifica por sí mismo el efecto marginal causal aislado de la primera acción. La identificación funcional proviene de intervenir el contenido durante la evaluación nativa; su utilidad debe medirse empíricamente y puede ser nula o adversa.

La actualización ordinaria usa una media incremental. Una entrada recuperada desde el origen del evento que recibe evidencia nueva con error absoluto medio de al menos `.08` se actualiza con tasa `.6`. La contradicción se calcula contra la media vigente justo antes de actualizar; se conserva la marca temporal de recuperación, no una copia del valor que tenía la entrada cuando fue recuperada. Para llamar a este evento **reconsolidación operacional** se requieren conjuntamente: entrada existente recuperada, nueva consecuencia factual, contradicción suficiente y cambio retenido. Se conserva el valor anterior y posterior en el registro. Es un mecanismo computacional diseñado; no una equivalencia validada con reconsolidación biológica.

`memory_updates` contiene sólo los eventos del último `observe`, con episodio, tiempos de origen/destino locales y absolutos, acción, horizonte, clave, observación base/destino, delta, antes/después, conteos, recuperación previa, error, tasa, bandera de actualización y bandera de reconsolidación. `last_events` contiene sólo eventos de metas del mismo paso. El ejecutor debe escribirlos en su traza antes de la próxima llamada. Los contadores agregados sí persisten.

## Metas persistentes bajo normas explícitas

Las presiones diseñadas dependen de salud menor que `.55`, reserva menor que `.65`, disminución observada y señal de demanda. Se elige entre supervivencia, reserva, tarea o balance. Los pesos respectivos son `(.75,.15,.10)`, `(.25,.65,.10)`, `(.25,.15,.60)` y los iniciales `(.5,.3,.2)`.

Una prioridad se mantiene al menos tres observaciones. Una sustitución requiere dos observaciones consecutivas apoyando una candidata y una diferencia de presión mayor que `.15`, o que la prioridad previa esté resuelta con presión menor que `.05`. La primera formación ya puede escoger pesos diferentes de los iniciales, antes de registrar cualquier revisión. Entre formación y revisiones, los pesos se mantienen exactamente en su último valor elegido; sólo `noGoalRevision` los fija a los iniciales. La revisión registra abandono de la anterior, pesos antes/después y formación de la nueva. La persistencia y el abandono son propiedades del controlador implementado, no evidencia de deseos experimentados.

## Intervenciones y persistencia

| Variante | Cambio | Lo que permanece |
|---|---|---|
| `noMemory` | Vacía tabla asociativa, pendientes e historia; usa sólo observación actual. | Controlador de metas y estados internos de otros módulos del agente. No equivale a un agente sin ningún estado. |
| `frozenMemory` | Impide nuevas entradas y cambios en valores/conteos asociativos. | Recuperación de lo aprendido e historia reciente. Es lesión de plasticidad asociativa, no congelación de todo el estado. |
| `noGoalRevision` | Fija pesos iniciales. | Observación, memoria, pronósticos y registro de persistencia. |
| `block_viability` | Sustituye únicamente contenido recibido por viabilidad por contenido neutro. | Esquema, ancho numérico, ruta de recursos y parámetros. |
| `block_resources` | Sustituye únicamente contenido recibido por recursos por contenido neutro. | Esquema, ancho numérico, ruta de viabilidad y parámetros. |
| `permuted` | Permuta coordenadas semánticas y filas de acción mediante asignaciones fijas. | Multiconjunto numérico por campo, tamaño, cabecera temporal y estado original. |

La lesión de contenido conserva el esquema y cantidad de bytes, pero altera el contenido y su distribución; no se afirma igualdad de información útil. La permutación conserva todos los valores numéricos y cambia su asociación semántica; no es una reducción de ancho de banda. Las dos rutas usan permutaciones declaradas distintas y no acceden a contadores futuros.

`set_modes(...)` permite fijar las lesiones; `consume(...)` permite aplicar puertas independientes. El ejecutor principal puede llamar directamente a los consumidores para que las condiciones sean explícitas. `reset_episode()` conserva la tabla aprendida, revisión y contadores acumulados, pero limpia pendientes, historia, activaciones de recuperación y metas. Incrementa el identificador de episodio; reinicia el tiempo local y mantiene el absoluto. Nunca se atribuyen consecuencias entre episodios.

`state_dict()` y `from_state_dict()` guardan/restauran tabla, historia acotada, pendientes acotados, metas, modos, contadores y eventos del último paso; no duplican una historia acumulada de actualizaciones. En evaluación se congelan los parámetros predictivos/Q/gates del agente, pero la memoria factual sigue adaptándose en línea salvo la lesión correspondiente. Cada variante debe partir de una copia del mismo checkpoint y esa plasticidad debe quedar en la traza.

## Verificación previa a campaña

`python -m unittest r9_completion.test_workspace -v` ejecuta12 contratos: llegada factual de horizontes, ausencia de objetivos entre episodios, retención del cambio tras recuperación y contradicción, la misma contradicción sin reactivación produce sólo actualización ordinaria, diferencias entre retirar memoria y congelar actualización, persistencia/revisión/abandono de metas, ancho y multiconjuntos de las rutas, replay JSON exacto, almacenamiento acotado, consumidores independientes y transición pública del entorno real.

Una prueba construye un checkpoint transparente del `NativeAgent` para exponer ambas rutas: con contenido íntegro elige acción2, al lesionar viabilidad elige0 y al lesionar recursos elige1. Las predicciones paramétricas y el consumidor no intervenido son idénticos. Es un test de mecanismo con una fixture construida, no evidencia de que un checkpoint aprendido muestre ese efecto ni de mejora de retorno. La prueba con entorno real recorre15 transiciones de desarrollo para comprobar integración y maduración de H12; tampoco es una campaña de utilidad. La evaluación confirmatoria y sus criterios pertenecen al protocolo global R9 y no se han ejecutado aquí.
