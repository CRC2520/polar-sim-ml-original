# Resolución de gaps P0–P1–P2: implementación, evidencia y límites

## Estado verificable

Se implementó un prototipo integrado en `integrated_polar/` y se ejecutaron el
diagnóstico P0, la evaluación prospectiva P1 y las capacidades específicas P2.
Las 128 pruebas pasaron. Los resultados originales están fijados en
`1bda1e22364ba068eb8b14289512b0776b4242bd`; el registro anterior a los finales es
`34ce5268e614b49b8910a355d7de603689b7431e`.

La decisión registrada es `engineering_verified_with_partial_empirical_support`.
Dos de cuatro comparaciones prácticas previas se cumplieron: uso de prioridades
y memoria recuperada al retornar un contexto. No se declara cerrado el requisito
general de utilidad de todas las capas ni se infiere conciencia. El cierre de
interfaces y mecanismos no equivale al cierre de las hipótesis filosóficas originales.

## Qué cambió en el regulador

La nueva realización no es un ajuste de los coeficientes antiguos. Conserva el
marco por capas, pero sustituye la suma de plantillas fijas por identificación de
efectos físicos dirigidos y planificación predictiva con restricciones.
RLS y MPC son métodos convencionales; no se atribuye su invención ni una ventaja
a los nombres de los polos. Las sensibilidades K_eff=J^T Q y W_eff=-J^T Q J se
derivan del predictor aprendido y de las prioridades. Las funciones y sus
intervenciones quedan expuestas dentro del ciclo integrado.

Los motores y datos de los Estudios 1–3 permanecieron sin cambios. La arquitectura
teórica y la figura original se mantienen en el mismo manuscrito LaTeX; se agrega
la realización actual con sus límites, no se sustituye la propuesta por el benchmark.

## P0: 2.448 ensayos de diagnóstico factorial

Se cruzaron 12 semillas, 12 celdas conocidas del Estudio 3, 16 combinaciones de
intervenciones y un control sin K. Cada ejecución tiene 64 pasos. Son ensayos
exploratorios en una familia conocida. El efecto principal se promedia sobre las
otras intervenciones; todas las celdas e interacciones se conservan.

| Intervención | Diferencia promedio en MSE | IC descriptivo 95 % |
|---|---:|---|
| Sustituir señal agregada por residuos firmados por canal | -0,000681667 | [-0,000980397; -0,000389697] |
| Cambiar aisladamente el signo del segundo polo receptor | +0,000508437 | [+0,000126742; +0,000920811] |
| Usar estimador acoplado e inversión directa aislada | +0,001493617 | [+0,000691230; +0,002400147] |
| Proyectar acciones según prioridades | -0,000161871 | [-0,000357381; -0,000007657] |

El primer cambio también elimina el sumando agregado de conflicto por coactivación:
no aísla exclusivamente dirección frente a todas las demás diferencias informativas.
El cambio de modelo modifica estimación y regla inversa conjuntamente. Los efectos
corresponden a esas intervenciones completas. No justifican que toda inversión de
signo ayude ni que toda estimación acoplada perjudique. Aplicar los cuatro cambios
juntos tampoco reparó uniformemente el controlador anterior.

Esto motivó una realización coherente: efectos dirigidos aprendidos, mensajes de
error firmados, prioridades en el objetivo y restricciones dentro del planificador,
en lugar de sumar correcciones aisladas sin evaluar sus consecuencias conjuntas.

## P1: 1.056 ensayos nuevos de control dinámico

24 semillas finales, cuatro celdas (dinámica fija/cambiante y dos presupuestos),
11 condiciones y 48 transiciones de evaluación por ejecución: 50.688 transiciones.
Cada condición recibe antes 48 acciones propias de identificación, idénticas entre
comparadores. El estimador no recibe la matriz verdadera del entorno.

El registro fija cuatro comparaciones y una mejora mínima de 0,0002 MSE. Se usan
20.000 remuestreos por semilla y límites superiores unilaterales de 98,75 %
(ajuste .05/4). Los intervalos percentiles son aproximados.

| Función intervenida | Diferencia full menos ablación | Límite superior ajustado | Criterio previo |
|---|---:|---:|---|
| Efectos cruzados en planificación | -0,000452838 | -0,000059130 | No cumple la mejora mínima garantizada por el criterio |
| Prioridades en la decisión | -0,001493797 | -0,001088915 | Cumple |
| Memoria en el retorno de contexto | -0,046257406 | -0,042981422 | Cumple |
| Horizonte multietapa, error tras cambios | +0,000087530 | +0,000145225 | No cumple; pequeño deterioro observado |

El MSE total del prototipo es 0,022983036; sin prioridades, 0,024476833:
reducción descriptiva del 6,10 %. El error de retorno es 0,031773430 con memoria
y 0,078030836 sin memoria: reducción del 59,28 %. No hubo violaciones duras.

El predictor acoplado mejora la media 1,93 % respecto a su lesión, pero su límite
ajustado no acredita el mínimo registrado. Eso no es una demostración de ausencia
de efecto; tampoco permite declararlo aprobado conforme al protocolo.

El horizonte multietapa existe y una prueba específica demuestra anticipación
cuando se suministran metas futuras distintas. En el benchmark principal los
pronósticos repiten la meta vigente; no anticipan el próximo cambio abrupto.
La mejora práctica del horizonte no se acredita en estas tareas. La continuidad
del estado interno es causalmente verificable, pero no mostró una necesidad general
de desempeño: el MSE de su lesión es prácticamente igual al del modelo completo.

La fórmula matricial genérica equivalente reproduce las acciones hasta
1,7486e-15. Esto delimita la contribución: una organización intervenible de
funciones de control, no una superioridad de las coordenadas polares.

## P2: capacidades medidas por separado

Se usaron 24 semillas distintas de las de P1.

| Capacidad | Resultado | Interpretación válida |
|---|---|---|
| Inferencia de fuente entre acciones candidatas | 278/288 identificaciones correctas (96,53 %) | Atribución condicional a las historias candidatas disponibles; no descubrimiento universal de agentes |
| Ambigüedad de fuentes con efectos idénticos | 100 % de abstención | No inventa una identificación cuando las predicciones son indistinguibles |
| Reglas institucionales con y sin autorización | 24/24 pares de casos correctos | Cumplimiento del conjunto explícito de reglas; no moralidad general |
| Gramática y signos físicos de llenado/vaciado | Error de interpretación 0; signos aprendidos correctos | Semántica operacional de canales, no validación de categorías filosóficas |
| Coordinación de dos controladores independientes | Diferencia MSE -0,028032373; IC95% [-0,030766464; -0,025125471]; cero violaciones de presupuesto | Beneficio observado de ofertas escalares frente a reparto igualitario en esta tarea |
| Pronóstico probabilístico de éxito | Brier 0,044062 frente a 0,227354 del predictor de tasa base; ECE 0,013207 | Calibración aceptada en este dominio de evaluación |

El monitor usa 4.608 pronósticos antes de recibir resultados. Estos pasos no son
4.608 réplicas independientes: pertenecen a 24 semillas y cuatro celdas.

### Política de revisión: resultado adverso que se conserva

La política experimental de revisión con umbral 0,55 activó revisión en 92,32 % de
sus decisiones y aumentó el MSE en +0,073295857 frente a la política base.
Por tanto, un pronóstico bien calibrado NO basta para elegir una respuesta útil.
La revisión reduce la acción a mínimos institucionales factibles; no garantiza
seguridad física ni mejor desempeño. No se recomienda habilitar esa política como
una mejora: la configuración por defecto mantiene meta_threshold=0. La política
experimental y sus datos se conservan para diagnóstico, no se retocan para aprobarla.

## Matriz de cierre por gap

| Gap | Prioridad | Resolución implementada | Estado y alcance pendiente |
|---|---|---|---|
| G01 | P1 | Estados separados para activación interna, intención y acción; intervención sobre continuidad | Ingeniería verificada. Beneficio general de la continuidad no establecido |
| G02 | P0 | Residuos firmados, déficit positivo/negativo y coactivación separados | Ingeniería verificada y diagnóstico favorable de la sustitución completa; falta aislar cada subcomponente informativo |
| G03 | P0 | Sensibilidades por canal receptor derivadas del modelo aprendido | Ingeniería verificada. Se rechaza la idea de que basta invertir signos de forma fija |
| G04 | P1 | Aprendizaje de cada coeficiente acción–efecto dirigido | Ingeniería verificada. No aprende todavía la taxonomía psicológica ni una topología neural arbitraria |
| G05 | P0 | Predictor acoplado y lesión exclusiva de efectos cruzados en planificación | Implementación verificada; mejora media favorable, pero criterio práctico final no aprobado |
| G06 | P0 | Prioridades dentro del objetivo y de la asignación restringida | Ingeniería y mejora práctica aprobadas en las tareas ensayadas |
| G07 | P1 | Contenido identificable distribuido a planificación, memoria e informe, con cortes selectivos | Transferencia funcional verificada. Competencia rica entre contenidos y especialistas heterogéneos sigue pendiente |
| G08 | P1 | Memoria por claves, compuertas, borrado y reconsolidación versionada autorizada | Ingeniería y mejora práctica de recuerdo aprobadas. No conciencia autobiográfica, sueños o trauma clínico |
| G09 | P2 | Prohibiciones, precedencia, obligaciones, razones y estado de revisión | Cumplimiento institucional delimitado verificado; no ética general ni autenticación de metadatos |
| G10 | P1 | Metas suministradas persistentes y planificación multietapa | Ingeniería verificada; utilidad práctica del horizonte no aprobada. Motivación intrínseca no implementada |
| G11 | P2 | Contrato acción–feedback, ensayo de fuentes y monitor probabilístico | Pronóstico calibrado en el dominio; la política de revisión ensayada es adversa y no está validada como mejora |
| G12 | P2 | Controladores locales con estados/modelos propios y comunicación de ofertas | Coordinación autónoma de recursos delimitada; no negociación social general |
| G13 | P2 | Adaptador de sensores/actuadores y gramática estricta de tanques/válvulas | Significado operacional verificado; categorías filosóficas originales y lenguaje general no validados |
| G14 | P1 | Ciclo integrado, contratos, trazas completas, pruebas y reproducción | Integración finita verificada; estabilidad adaptativa global y escalabilidad no demostradas |

## Reproducibilidad y preservación

128 pruebas aprobadas; 1.056 trayectorias P1 y 192 trayectorias P2 comprobadas.
20 reproducciones completas de controladores y dos de coordinación, con diferencia
máxima de acciones P1 igual a cero. Las observaciones y efectos físicos se reproducen
y las métricas se recalculan desde los archivos. Son verificaciones internas,
no una replicación independiente por otro equipo.

La nueva evaluación no registró perfiles de tiempo por controlador; no se infiere
ventaja de latencia ni de hardware a partir del costo de acciones. Las reglas,
semillas, calibrador y fuentes congeladas no se modificaron tras observar finales.

Para regenerar evidencia existente dentro del checkout del repositorio:

```bash
git fetch origin
git switch research/integrated-polar-p0-p2
python scripts/run_gap_tests.py
python -m gap_resolution.run regenerate
```

Usar el entorno registrado en docs/P0P2_FREEZE.json. La ejecución final rechaza
sobrescribir resultados: sus semillas ya están abiertas y reproducirlas no es una
nueva prueba confirmatoria. El código standalone del agente necesita NumPy; el
conjunto de pruebas y regeneradores también usa dependencias del repositorio.

Las fuentes del manuscrito y su PDF permanecen en CRC2520/POLAR_MODEL_CRC, rama
research/integrated-polar-p0-p2. Ambas entregas se mantienen en revisión, sin merge
a main. El prototipo implementa las funciones delimitadas, pero no convierte
los 14 gaps amplios en una lista de 14 capacidades generales científicamente demostradas.
