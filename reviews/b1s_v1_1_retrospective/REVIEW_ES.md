# Revisión retrospectiva de B1-S v1.1
## Competencia, estabilidad de selección y contribución de ruta

**Estado: RETROSPECTIVE_REVIEW_COMPLETE. Análisis exploratorio posterior al desarrollo; no es una nueva campaña ni una nueva adjudicación.**

## 1. Encargo, procedencia y límites

Se ejecutó la revisión solicitada sobre registros ya archivados, en este orden: competencia, estabilidad y contribución causal. No se cargaron modelos, no se entrenó, no se evaluaron políticas de nuevo, no se generaron semillas y no se ejecutó B1-E. Los cálculos adicionales no cambian el protocolo, los umbrales, la selección ni el veredicto congelados.

| Referencia inmutable | Valor |
|---|---|
| Repositorio científico | `CRC2520/polar-sim-ml-original` |
| Fuente científica | `50c4e08cf3d69c998ce95b8da1e1647a09b31339` |
| Publicación científica recuperada | `de35a0995d96ff548772fa56c3620466495add72` |
| Cierre de copia documental en código | `6266513ef74f8f662b94c8dd026cefd4be77f7f6` |
| Repositorio y commit documental | `CRC2520/POLAR_MODEL_CRC@6625fb6dfbd9505115017d51eb6d534add9fc35b` |
| Run y artefacto documentales originales | `34798275181` / `10330029112` |
| SHA-256 del ZIP documental usado | `42504ec6a08c9bdb3cf75fd8d89fb2af2e8502396ae7baf917f7acfa1ba6eeb7` |
| SHA-256 de los bytes de FREEZE.json | `9a61e714e78b9c2235dfc6dfe4093be0f5b66a9fa3ae4b15ef967a12f354d7c8` |

Se verificaron el ZIP documental original, sus 44 entradas de importación, los 39 archivos textuales presentes del freeze, el receipt y el manifiesto de adjudicación. El paquete conserva fuente, publicación y disposición esperadas. Se revisaron también, como texto y sin importarlos, `analysis.py`, `train.py`, `core.py` e `instrument.py` de la fuente científica.

**Cobertura real:** 1.344 registros competitivos (192 por cada uno de seis modelos y por el testigo); 276 registros de selección/calibración con sus episodios; 72 puntos de curva de calibración (18 ajustes × 4 puntos); 48 casos causales con snapshots y cuatro variantes; y 90 comparaciones funcionales en seis paneles. Son registros dependientes dentro de sus políticas/paneles, no 1.344 o 90 réplicas independientes de entrenamiento.

Esta iteración no abrió las trayectorias completas comprimidas, los checkpoints ni los 561 binarios que no están en la importación documental. Esas trayectorias sí se retuvieron en el repositorio científico: **no inspeccionadas aquí** no significa **no archivadas**. En cambio, el protocolo declara que no se conservaron todas las transiciones de entrenamiento ni los buffers completos de SAC. No se reconstruyeron datos ausentes. [S1–S6]

El informe distingue hechos archivados, cálculos retrospectivos e inferencias. Un `PASS` de integridad o aritmética no es una certificación científica ni una reproducción independiente.

## 2. Competencia: causa aritmética del diagnóstico fallido

La conjunción archivada exige una mejora de pérdida mayor que `130/30` y un desplazamiento medio adicional de al menos una unidad nativa, ambos frente al testigo `no-action`. La pérdida es el negativo del retorno acumulado; un valor menor es mejor. El desplazamiento es la posición final del paquete menos su posición al reiniciar el episodio, no una medida de locomoción exitosa condicionada a supervivencia. [S1, S2, S8]

El testigo tiene desplazamiento medio **1,7931097349**. Por tanto, en este panel competitivo, el componente de desplazamiento exige alcanzar al menos **2,7931097349** de media absoluta. No es un umbral nuevo: es la misma desigualdad archivada expresada con el testigo observado.

| Modelo | Mejora de pérdida | Desplazamiento | Adicional al testigo | Episodios con caída /192 | Alcanza 500 ciclos /192 | Ciclos medios |
| --- | --- | --- | --- | --- | --- | --- |
| S-M0 | 98.4980 | 0.5660 | -1.2271 | 1 | 190 | 495.86 |
| S-M2-fixed | 91.7854 | 0.6192 | -1.1740 | 9 | 177 | 470.56 |
| S-M3 | 95.9634 | 2.1972 | 0.4041 | 15 | 172 | 484.08 |
| S-M4 | 102.5666 | 2.6069 | 0.8138 | 5 | 182 | 494.40 |
| S-M5 | 72.1487 | 1.5618 | -0.2313 | 55 | 134 | 397.49 |
| S-M6 | 93.2409 | 0.0888 | -1.7043 | 4 | 185 | 486.31 |
| no-action | 0.0000 | 1.7931 | 0.0000 | 192 | 0 | 111.64 |

**Hecho verificado:** los seis modelos superan el componente de mejora de pérdida y los seis fallan el de desplazamiento adicional. No hay un modelo que falle por ambas condiciones. El fallo agregado de competencia queda explicado aritméticamente por el componente de desplazamiento; la causa del comportamiento aprendido requiere una interpretación distinta. [S2; cálculo R1]

El testigo solicita acción cero y sus **192/192 episodios registran caída**, con duración media **111,640625 ciclos**. Su desplazamiento positivo no demuestra transporte sostenido ni competencia. Los modelos registran muchas menos caídas y episodios más largos. En consecuencia, mejorar la pérdida frente a este testigo y no alcanzar su desplazamiento más una unidad no son resultados contradictorios.

**Inferencia compatible, no causa demostrada:** parte de la mejora de pérdida puede relacionarse con evitar estados de fracaso sin conseguir suficiente avance sostenido. El instrumento fija `terminate_reward=-100` y `fall_reward=-10`. No se ha reconstruido una descomposición exacta de recompensa; no se asigna un porcentaje de mejora a evitar caídas ni se atribuye una estrategia intencional al agente. Tampoco se diagnostica un error físico a partir del desplazamiento del testigo. [S8]

Como descripción posterior al desarrollo, la pérdida media de episodios con caída o paquete perdido es 75,3042 en S-M3 frente a −7,0476 sin esos indicadores; en PPO/S-M5 es 101,2436 frente a −7,5061. Estas particiones se definen por el resultado y son **asociaciones descriptivas**, no efectos causales. Caída y paquete perdido pueden coincidir; sus conteos no se suman como si fueran disjuntos. Alcanzar 500 ciclos tampoco se identifica automáticamente con ausencia de fracaso. [R1]

### 2.1. Heterogeneidad por inicialización

Aplicar descriptivamente la misma conjunción a cada bloque de 32 episodios, con su propio testigo emparejado, da 2/6 bloques que la satisfacen para S-M3 (réplicas 3 y 5) y 2/6 para S-M4 (4 y 5). Los otros cuatro modelos dan 0/6. Este cálculo **no reemplaza el diagnóstico agregado** ni autoriza seleccionar las réplicas favorables.

S-M3 varía entre desplazamientos medios por réplica de aproximadamente 0,325 y 5,556. En su réplica 3 se combina mayor desplazamiento con 12/32 episodios con caída y pérdida media 21,546; la réplica 5 obtiene desplazamiento 5,294 y pérdida −13,071. Esto muestra heterogeneidad de comportamiento, no una única política estable reproducida seis veces. [R1]

### 2.2. Calibración: no confundir callback y modelo final

Se recalcularon las medias de los tres ajustes por perfil en los cuatro puntos archivados, manteniendo la selección exclusivamente terminal. [S3, S7]

| Perfil | 32.768 pasos | 131.072 | 262.144 | 524.288: checkpoint | Selección: modelo terminal |
| --- | --- | --- | --- | --- | --- |
| ppo-conservative | 105.7666 | 44.0528 | 10.3589 | -2.3100 | -2.3100 |
| ppo-standard | 101.3507 | 22.4974 | 28.5642 | -5.4548 | -5.4548 |
| ppo-v1selected | 104.0495 | 69.3265 | 0.8568 | 17.8466 | 17.8466 |
| sac-conservative | 103.2575 | 28.9843 | 36.4168 | 16.6429 | 21.6748 |
| sac-standard | 105.1919 | 57.3424 | 23.0266 | 9.0115 | 0.2867 |
| sac-v1selected | 82.2527 | 106.9810 | 85.0480 | 91.9952 | 90.3592 |

PPO seleccionó `ppo-standard`; SAC seleccionó `sac-standard`. Las pérdidas de selección terminal son −5,4548 y 0,2867. Los resultados competitivos de nuevos ajustes son 25,3454 y 4,2531: existe deterioro descriptivo entre fases, pero cambian inicializaciones y paneles, de modo que esa diferencia no demuestra por sí sola sobreajuste ni identifica su causa.

Para SAC, el checkpoint del callback a 524.288 pasos antecede las actualizaciones del último bloque, mientras que la selección usa el modelo terminal posterior. En `sac-standard`, **9,0115 y 0,2867 son observaciones distintas**. El código y el protocolo explican el momento de medición; no se corrige uno con el otro ni se afirma una inconsistencia de selección. La variación entre puntos y fases tampoco certifica convergencia. [S1, S3, S7; R2]

## 3. Selección: regla cumplida, soporte no robusto

Se reconstruyeron las medias de los 42 soportes elegibles (el denso `111111` no participa en esa selección) desde los registros de selección. Se comprobó el desempate por número de aristas y orden lexicográfico. El resultado coincide con `SELECTION_FREEZE.json`: **`000001` es el mínimo de la media agregada**. No hay discrepancia aritmética en esa elección. [S4, S7]

| Soporte | Pérdida media de selección | Desviación entre seis medias |
| --- | --- | --- |
| 000001 | 3.235475 | 6.277820 |
| 001001 | 3.491142 | 3.411077 |
| 111000 | 3.903506 | 7.270780 |
| 000110 | 4.815083 | 10.486326 |
| 010010 | 5.180905 | 4.598577 |
| 010101 | 5.534341 | 12.511367 |

La separación entre los dos primeros es **0,2556674172**, y sus diferencias por réplica cambian de signo. La desviación mostrada es entre medias de ajustes, no error estándar de todos los episodios. Una diferencia pequeña respecto de la dispersión es descriptiva; no prueba equivalencia ni una probabilidad de ser el mejor. [R3]

| Réplica | Ganador | Segundo | Diferencia segundo−ganador | Rango de 000001 |
| --- | --- | --- | --- | --- |
| 0 | 010101 | 100100 | 0.397871 | 13 |
| 1 | 100100 | 001000 | 0.977937 | 10 |
| 2 | 001000 | 000010 | 0.990978 | 16 |
| 3 | 110010 | 010101 | 0.330621 | 11 |
| 4 | 011000 | 001100 | 0.029558 | 30 |
| 5 | 001101 | 000110 | 0.703619 | 11 |

El soporte agregado no gana ninguna réplica y sus rangos individuales son **13, 10, 16, 11, 30 y 11**. No es una contradicción: minimizar una media y contar victorias son objetivos diferentes. No corresponde sustituir la media por una votación ni elevar al ganador de una réplica a selección oficial.

Se conservan seis ganadores distintos, frecuencia modal **1/6** y Jaccard medio **0,2022222222**, por debajo de los requisitos archivados **2/3** y **0,6**. El referente condicionado al número de aristas es 0,2581851852; no es un valor p ni permite declarar “peor que azar”. [S4]

### 3.1. Sensibilidad por eliminación de una inicialización

Se recalculó, exclusivamente como diagnóstico retrospectivo, qué máscara minimiza la media de las **mismas** pérdidas al excluir sucesivamente una de las seis inicializaciones. No se entrenó ni evaluó ninguna máscara:

| Inicialización omitida | Mínimo aritmético de las cinco restantes | Rango de 000001 |
| --- | --- | --- |
| 0 | 010010 | 2 |
| 1 | 110100 | 5 |
| 2 | 001001 | 2 |
| 3 | 001001 | 3 |
| 4 | 000010 | 3 |
| 5 | 010101 | 4 |

La identidad del mínimo cambia en las seis eliminaciones. Esto respalda **sensibilidad de esta elección a la composición de la muestra de desarrollo**, no una estimación de estabilidad poblacional. No se elige ninguna de esas alternativas, no se cambia el freeze y no se usan resultados competitivos para escoger un reemplazo. Los datos no separan con certeza ruido de evaluación, optimización finita y múltiples estructuras de desempeño parecido. [R3]

## 4. Contribución causal: dependencia local no es utilidad externa

Con `000001`, la única arista activa es `edge-5=(2,1)`, del nodo 2 al receptor 1 según el orden de `EDGES` y el agregador del actor. La intervención se aplica una sola vez después del prefijo archivado de ocho ciclos, y después se continúa con la política intacta. No es una ablación permanente ni una medición de un coeficiente causal físico. [S5, S7, S8]

Se verificó directamente en los snapshots que las **48/48 intervenciones activas cambian la acción** por encima del umbral de uso archivado `2e-7`. Las características `h` y los mensajes calculados `m` son idénticos antes/después de intervenir. Las 96 comparaciones de control (sham y fuera de soporte, dos por caso) conservan acción, pérdida, desplazamiento y duración. Los registros declaran pesos sin cambios. [R4]

| Réplica | Rango de cambio máximo de acción | Δpérdida =0 | Δpérdida >0 | Δpérdida <0 | Media Δpérdida |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.008262–0.011662 | 0 | 5 | 3 | 0.010807979 |
| 1 | 0.053430–0.085534 | 8 | 0 | 0 | 0.000000000 |
| 2 | 0.032824–0.039066 | 8 | 0 | 0 | 0.000000000 |
| 3 | 0.029714–0.039276 | 2 | 3 | 3 | -20.710789252 |
| 4 | 0.075527–0.088951 | 8 | 0 | 0 | 0.000000000 |
| 5 | 0.067766–0.111001 | 8 | 0 | 0 | 0.000000000 |

El signo es **bypass menos intacto**: positivo favorece a la ruta intacta en pérdida; negativo favorece al bypass para ese caso. Hay **34 ceros exactos registrados, ocho diferencias positivas y seis negativas**. No son ceros por prefijos faltantes: los 48 estaban disponibles. El rango global de cambio máximo local es 0,0082623–0,1110011.

Los otros dos bloques motores permanecen idénticos en los 48 casos: cambia solo el bloque del receptor 1. Además, los signos de las doce coordenadas de acción no cambian en ninguno. Bajo la parametrización del instrumento, esto mantiene las consignas de velocidad que dependen del signo y cambia los límites de par que dependen de la magnitud. **No demuestra que el par realizado ni la trayectoria física permanezcan iguales.** Una limitación no activa del actuador es una explicación compatible con parte de los ceros externos, no una causa verificada: faltan en esta revisión las trayectorias completas y la medición de esfuerzos realizados. [S8; R4]

### 4.1. Casos externos adversos conservados

En la réplica 3, dos casos concentran cambios externos grandes:

| Caso | Pérdida intacta → bypass | Desplazamiento intacto → bypass | Ciclos intactos → bypass |
|---|---|---|---|
| r3, episodio 2 | 69,1447 → −16,5514 | 8,0953 → 3,8906 | 459 → 500 |
| r3, episodio 7 | 62,5350 → −31,9436 | 9,6322 → 7,4672 | 472 → 500 |

En ambos, el bypass mejora pérdida, reduce desplazamiento final y prolonga el episodio. Es evidencia concreta de que los dos objetivos pueden moverse en sentidos distintos; no prueba que retirar la ruta sea universalmente mejor ni que la diferencia provenga exclusivamente de una penalización terminal. La media externa de las seis réplicas es −3,4499968788, dominada por el patrón de la réplica 3; no se presenta como conclusión confirmatoria ni se descartan esos casos. [S5; R4]

**Interacción entre rutas no evaluada:** `two_edge_interaction` es nulo en los 48 casos porque solo hay una arista activa. Eso es “no evaluado”, no interacción igual a cero. Este diagnóstico no prueba propagación entre varias polaridades ni sinergia entre dos aristas activas.

## 5. Acuerdo funcional y recursos

Ninguna de las **90 comparaciones** (15 pares × 6 paneles) cumple simultáneamente los márgenes archivados de diferencia media ≤0,01 y diferencia máxima ≤0,05. Los tamaños de panel son 512, 512, 499, 512, 512 y 512. No satisfacer ese criterio descriptivo no es un test de diferencia estadística, y los paneles finitos no establecen equivalencia o no equivalencia global. [S6; R5]

Se conserva igual exposición competitiva de 524.288 pasos por ajuste, pero no igual computación:

| Modelo | Actualizaciones registradas | Parámetros actor | Parámetros crítico | Flujos |
| --- | --- | --- | --- | --- |
| S-M0 | 16384 | 2288 | 10241 | 8 |
| S-M2-fixed | 16384 | 2288 | 10241 | 8 |
| S-M3 | 16384 | 2288 | 10241 | 8 |
| S-M4 | 16384 | 2288 | 10241 | 8 |
| S-M5 | 81920 | 10968 | 10241 | 8 |
| S-M6 | 514288 | 11736 | 22018 | 1 |

SAC conserva además replay y críticos objetivo; las unidades de actualización no representan operaciones idénticas entre algoritmos. Estas diferencias estaban declaradas y limitan una lectura de “capacidad equivalente”. Los cuatro modelos de grafo comparten arquitectura de actor y número de parámetros, pero sus rutas utilizadas difieren. [S6]

## 6. Matriz de afirmaciones y conclusión admisible

| Afirmación | Evidencia examinada | Conclusión admisible |
|---|---|---|
| La selección publicada contradice su regla | Medias y desempates reconstruidos | No: `000001` coincide con la regla agregada. |
| Se descubrió una estructura estable | Ganadores, Jaccard y sensibilidad retrospectiva | No respaldado en este desarrollo. |
| No hubo ninguna mejora conductual | Pérdida, caída y duración frente al testigo | Demasiado fuerte: mejora de pérdida y menos caídas coexistieron con competencia fallida. |
| Los seis modelos fueron competentes | Conjunción archivada aplicada a cada modelo | No: todos fallan el desplazamiento adicional agregado. |
| El fallo es exclusivamente de PPO/SAC | Competencia de los seis modelos | No: tampoco pasan los cuatro modelos estructurados. |
| La ruta seleccionada es computacionalmente inerte | 48 snapshots intervenidos | No en estos estados: hay dependencia local medible. |
| La ruta aporta una ventaja externa estable | Efectos heterogéneos y cinco contrastes inconclusos | No demostrado; se conservan los signos adversos. |
| Se probó interacción entre polaridades | Una arista activa; factorial no aplicado | No evaluado por este diagnóstico. |
| Hay equivalencia global entre políticas | Paneles finitos y contrastes inconclusos | No demostrado. |
| Se validó la arquitectura completa o consciencia | Tarea nativa, tres nodos y funciones parciales | No: catálogo, transferencia y arquitectura completa siguen fuera de alcance. |

**Síntesis:** el desarrollo muestra organización computacional operativa y dependencia local de una ruta, pero no una estructura estable ni una ventaja comparativa estable frente a alternativas competentes. La reducción de pérdida y de caídas no basta para cumplir el criterio multicomponente. La inestabilidad no implica automáticamente un error de selección y los ceros externos no implican una ruta computacionalmente inactiva.

No se determina una causa única de la falta de competencia. La influencia relativa de la optimización, los presupuestos, la sensibilidad de selección y la física de los actuadores no se identifica de manera completa con los resúmenes y snapshots examinados. Tampoco se obtiene evidencia nueva de consciencia o del catálogo polar.

## 7. Disposición conservada y reproducción

La revisión **no modifica** el resultado científico ni la adjudicación:

```text
scientific_status=B1S_V1_1_DEVELOPMENT_COMPLETE_WITH_LIMITATIONS
adjudication_status=ADJUDICATION_COMPLETE
B1E_disposition=ON_HOLD_FOR_CURRENT_STRUCTURAL_CONFIRMATION
ready_for_b1e_protocol_design=false
ready_for_b1e_freeze=false
ready_for_b1e_confirmatory_run=false
B1E_executed=false
final_seeds_generated=false
H_CAT=NOT_EVALUABLE
H_TRANSFER=NOT_EVALUATED
```

El siguiente punto de decisión es la revisión explícita por el autor de esta matriz de evidencia y límites. Este documento no propone ni autoriza automáticamente otra campaña, un freeze, un registro o una ejecución B1-E.

El script adjunto usa solo la biblioteca estándar, lee el ZIP original y produce métricas descriptivas. No importa módulos científicos, no usa generadores aleatorios, no carga checkpoints y no ejecuta el adjudicador:

```bash
python -S -B review_archive.py \
  --archive /ruta/B1S-v1-1-documentary-Spanish-package.zip \
  --out /ruta/nueva/REVIEW_METRICS.json
```

Se incluyen los resultados completos en `REVIEW_METRICS.json` y sus hashes en `REVIEW_MANIFEST.json`. Las comprobaciones de integridad, reglas y aritmética de esta revisión no sustituyen la evaluación original. Los archivos de entrada y el cierre se mantienen sin cambios. Las nuevas notas se publican fuera de `b1s/v1_1/results/` y `adjudication/`.

## 8. Fuentes internas y trazabilidad de cálculos

Todas las cifras empíricas proceden de los registros archivados, sin fuentes externas para rellenar resultados:

- **S1:** `b1s/v1_1/PROTOCOL.md` y `PLAN.json`, fuente científica fijada arriba.
- **S2:** `b1s/v1_1/results/COMPETITIVE_RESULTS.json`, publicación científica fijada.
- **S3:** `results/BASELINE_CALIBRATION.json` y registros de calibración de `DISCOVERY_RESULTS.json`.
- **S4:** `results/DISCOVERY_RESULTS.json`, `SELECTION_FREEZE.json` y `STABILITY.json`.
- **S5:** `results/CAUSAL_RESULTS.json`; resúmenes y snapshots, no relectura de trayectorias completas.
- **S6:** `results/FUNCTIONAL_PROBES.json`, `RESOURCE_AUDIT.json`, `FREEZE.json`, `PUBLICATION_RECEIPT.json` e importación/adjudicación documentales.
- **S7:** `b1s/v1_1/analysis.py` y `train.py`, fuente científica fijada; lectura estática.
- **S8:** `b1s/execution/instrument.py` y `core.py`, fuente científica fijada; lectura estática.
- **R1–R5:** cálculos retrospectivos reproducibles del script adjunto: competencia/estratos, curvas, selección/eliminación, casos causales y paneles. Son añadidos analíticos, no cambios del registro histórico.

Los mensajes de publicación documental incluyen `[skip ci]` para no disparar de nuevo campañas o adjudicación por eventos de push/PR. No se modifica ningún workflow ni se convierte esa omisión en un `PASS` de CI. Las PR permanecen en draft y no se hace merge.
