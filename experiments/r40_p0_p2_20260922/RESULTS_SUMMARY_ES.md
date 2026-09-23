# R40 — resultados de los ajustes de diseño y pruebas P0–P2

Fecha local de ejecución: 22 de septiembre de 2026 (Lima).
Ejecución UTC: 23 de septiembre de 2026.
GitHub Actions: 35814392593.
Commit científico ejecutado: e1a361b6febaf3b3660f275d13ab597060017168.
Protocolo previo: aa6d1371cabd6dbce8940912a0687dbe696a420c.
Base del simulador: d28958279eae052cffcc319146a200614575c030.

## 1. Alcance y decisión

Esta campaña responde a las prioridades P0–P2 posteriores a R39; no reescribe los antiguos experimentos llamados P0/P1/P2. Las ejecuciones computacionales terminaron, pero los frentes científicos no quedaron todos resueltos.

| Frente | Resultado | Interpretación válida |
|---|---|---|
| P0: comparador recurrente de terceros | P0_BASELINE_DEV_FAIL_NO_CONFIRM | Tres entrenamientos terminados; ninguno alcanza 80% de acierto en el nuevo problema de recuerdo diferido. Confirmación no abierta. |
| P0: replicación independiente E6b | OPEN | Se entrega paquete reproducible; no se atribuye una réplica independiente al mismo equipo/programa. |
| P1: control y conjunción en dominio nuevo | P1_NEW_DOMAIN_JOINT_FAIL | Conjunción estricta 0/32 por condición y controlador; no se demuestra beneficio del filtro propuesto. |
| P1: reducción de estado | Parcial, limitada a actuación | El archivo episódico puede eliminarse sin cambiar acciones/predicciones en 32/32 casos, pero se pierde la consulta de episodios antiguos. |
| P2: especificidad de historia del perfil actuador | P2_PROFILE_SPECIFICITY_PASS_LOCAL | 25/32 superan el efecto diferencial frente a un donante con el mismo perfil; no demuestra identidad fenomenal. |
| P2: E7 biológico | BLOCKED_REAL_DATA_PENDING | Cero registros biológicos. Solo se ejecuta calibración estadística sintética. |

No se adopta BAYES_CUE como una mejora de rendimiento validada. POLAR Core v1.1 permanece como hipótesis candidata D+C+R con A condicional. Ninguno de estos resultados demuestra conciencia, minimalidad universal o superioridad POLAR.

## 2. Cambios de diseño ejecutados

Se conserva el código histórico y se abre una campaña nueva, con semillas y escenarios distintos. El comparador recurrente se prueba en POPGym RepeatPreviousEasy, no en el panel CartPole ya consumido. Se entrenan tres modelos independientes, con igual presupuesto fijo y sin seleccionar checkpoints según evaluación.

En P1, las matrices, secuencias contextuales, metas de control y perturbaciones se generan con flujos aleatorios separados. Las metas no identifican el régimen oculto. El agente no recibe matrices verdaderas, etiquetas de fuente externa ni identificadores del régimen. Se conserva la observación terminal real, evitando el pseudo-objetivo cero del runner histórico.

La única modificación de arquitectura evaluada es un filtro HMM convencional de cuatro contextos, con emisiones Student-t. Se compara con el agente R36-R1 congelado bajo condiciones emparejadas. No hubo búsqueda de hiperparámetros ni selección de variante posterior a los resultados.

La planificación primaria se evalúa sobre todos los estados elegibles, no solo sobre los estados donde el selector decide planificar. La ganancia condicionada a invocación se conserva como métrica secundaria. La coexistencia se evalúa dentro de la misma semilla, trayectoria y agente.

En P2 se incorpora un donante de igual perfil actuador y otro de perfil diferente, ambos con experiencia independiente y las mismas matrices del mundo. Esto separa especialización al propio perfil de daño genérico por trasplantar cualquier estimador.

## 3. P0 — comparador recurrente externo

Se ejecutó RecurrentPPO/MlpLstmPolicy de sb3-contrib en el entorno externo no modificado RepeatPreviousEasy. Cada uno de los tres modelos recibió 262144 transiciones: 786432 en total. Se usaron los estados LSTM y las marcas de inicio de episodio en evaluación. El control sin historia reinicia el estado en cada decisión.

| Semilla de entrenamiento | Acierto con historia | Beneficio frente al reinicio | Elegible |
|---|---:|---:|---|
| 2066001 | 0.750000 | +0.526042 | No |
| 2066002 | 0.744792 | +0.493056 | No |
| 2066003 | 0.744792 | +0.491319 | No |

La regla previa exigía al menos dos modelos con acierto >=0.80 y beneficio de historia >=0.10. Los tres superan el requisito de utilidad de historia, pero ninguno el de capacidad absoluta. El control de cola de observaciones supera la verificación del instrumento.

Por tanto, las semillas confirmatorias 2067001–2067032 permanecen sin utilizar. Tampoco se ejecuta el panel confirmatorio con corrupción de observaciones. El resultado no implica que RecurrentPPO sea incapaz en general ni valida el comparador en el antiguo panel de control ruidoso.

## 4. P1 — robustez y coexistencia en un generador nuevo

Se completaron 144 vidas primarias de 1760 pasos: 16 de desarrollo y 128 confirmatorias. Las 32 semillas confirmatorias se evaluaron en dos controladores y dos condiciones de observación. Se retuvieron trazas completas.

| Controlador | Condición | Conjunción estricta | Conjunción con planificación invocada | Costo medio |
|---|---|---:|---:|---:|
| R36-R1 congelado | Estándar | 0/32 | 1/32 | 0.05632829 |
| BAYES_CUE | Estándar | 0/32 | 1/32 | 0.05694446 |
| R36-R1 congelado | Estrés | 0/32 | 0/32 | 0.05579895 |
| BAYES_CUE | Estrés | 0/32 | 0/32 | 0.05603547 |

La mejora emparejada se define como costo congelado menos costo BAYES_CUE; valores positivos favorecerían el ajuste:

- Estándar: -0.00061618; intervalo bootstrap del 95% [-0.00181023, +0.00041316].
- Estrés: -0.00023652; intervalo del 95% [-0.00123649, +0.00090178].

No se cumple el criterio de beneficio positivo y no se alcanza la conjunción exigida de 24/32 en ninguna condición. El ajuste no queda validado.

Estos resultados prueban un límite de transporte de la arquitectura y de sus guardas. Los umbrales de daño MSE se conservaron numéricamente desde R36-R1; no son invariantes a cualquier nueva distribución. No superar un umbral no equivale a demostrar ausencia de D, C o R. Tampoco es lícito comparar directamente el 7/12 histórico con este 0/32 como si población y endpoint fueran idénticos.

## 5. P1/P2 — archivo episódico, estado causal e identidad operacional

La batería confirmatoria contiene 32 receptores, con donantes emparejados e intervenciones de clonación, eliminación de archivo, trasplante de archivo, reinicio recurrente, sustitución de matriz actuadora, sustitución gradual y mezcla aritmética.

- Clonación exacta: acciones y predicciones conservadas en 32/32.
- Eliminación del archivo episódico: acciones y predicciones conservadas en 32/32.
- Trasplante del archivo episódico: acciones conservadas en 32/32.
- Reinicio del estado recurrente: acciones modificadas en 27/32.
- Mayor daño por donante de perfil diferente que por donante del mismo perfil: 25/32 superan el margen previo de 0.0001; requisito 24/32.
- Margen mediano de especificidad de perfil: +0.00073869.

El estado serializado mediano baja de 332866 a 137633 bytes al retirar el archivo y campos auxiliares. Esta reducción de almacenamiento NO es una prueba de dimensión causal mínima. Las consultas de historia antigua dejan de estar disponibles: la reducción de capacidad completa tiene 0/32 éxitos.

La conclusión más importante es que el archivo episódico del controlador ensayado funciona como registro, no como dependencia necesaria de su política. En cambio, la estimación recurrente y los parámetros aprendidos sí pueden afectar a la actuación. El PASS de perfil es especialización de identificación/control al propio actuador; no prueba un sujeto, identidad universal o memoria autobiográfica causal.

## 6. P2 — E7 y condición de evidencia real

No se generaron datos biológicos ni se sustituyó E7 por datos sintéticos. La calibración de la tubería estadística simuló 1000 estudios con 64 unidades emparejadas y cuatro contrastes, corrección Bonferroni y efecto alternativo d=0.5.

- Error familiar bajo nulo: 0.046.
- Potencia para detectar al menos un efecto: 0.999.
- Potencia para detectar simultáneamente los cuatro efectos: 0.753.
- Registros biológicos reales: 0.

El instrumento pasa su control sintético, pero E7 permanece OPEN. E6b también permanece OPEN: usar código de terceros en una ejecución propia no es una réplica independiente.

## 7. Integridad y entregables

Una auditoría posterior reconstruyó 477 comprobaciones desde los registros crudos, incluyendo elegibilidad P0, presupuestos y hashes de checkpoints, 144 trazas finitas completas, criterios por semilla P1 y controles P2. Los cuatro ZIP de evidencia coinciden con sus SHA-256 de GitHub Actions.

Artefactos del run 35814392593:
- P0: 10730949430.
- P1: 10731210026.
- P2: 10730749515.
- Instrumento E7 y handoff E6b: 10730594451.

Se entregan protocolo, fuentes exactas, checkpoints, registros completos, trazas, verificador y reporte. El éxito técnico de GitHub Actions significa ejecución correcta, no aceptación científica de las hipótesis.

## 8. Decisión de investigación

Se conservan los resultados negativos sin modificar umbrales, consumir las semillas confirmatorias bloqueadas ni sustituirlos por una narrativa de éxito. No cambia la versión del núcleo. Sigue pendiente demostrar robustez externa, conjunción persistente robusta, minimalidad de capacidades completas, identidad fuerte, replicación independiente y correspondencia biológica.

La corrección inmediata al manuscrito debe distinguir memoria archivada, estado predictivo aprendido e identidad operacional. Para cualquier nuevo intento de mejora, los criterios de control global, utilidad de memoria episódica y comparación con donantes emparejados deben fijarse antes de evaluar; los paneles de esta campaña no deben reciclarse para una nueva confirmación.
