# Ensayo acotado de competencia: normalización y presupuesto

**CP-NB-1.0.0 — desarrollo prospectivo nuevo, autorizado el 14 de septiembre de 2026. No es B1-S v1.1 reejecutado ni B1-E.**

## 1. Objeto y autorización

El autor solicitó empezar un ensayo acotado con `000000`, `111111`, PPO y SAC, probando normalización y presupuesto de aprendizaje de forma controlada. Se implementa únicamente ese ensayo. No se busca otra máscara, no se aplican lesiones y no se confirma la arquitectura polar. Los valores operativos siguientes son decisiones nuevas de diseño para limitar este piloto, no parámetros recuperados ni garantías de potencia.

Fuentes conservadas: código `98d286f90a23b8a2b5ade19dfe0eb305d03c5468`; documento `d6eada4ea40a28c192f22e7eecaf3d8dd540af79`; fuente científica histórica `50c4e08cf3d69c998ce95b8da1e1647a09b31339`; publicación original `de35a0995d96ff548772fa56c3620466495add72`; freeze histórico `9a61e714e78b9c2235dfc6dfe4093be0f5b66a9fa3ae4b15ef967a12f354d7c8`. Se conserva la disposición `ON_HOLD_FOR_CURRENT_STRUCTURAL_CONFIRMATION`.

Rama nueva: `research/competence-normalization-budget-20260914`. Directorio nuevo: `experiments/competence_pilot/`. Las ramas históricas, sus fuentes, `results/`, adjudicación, plan y workflows no se modifican.

## 2. Matriz fija y unidades

Cuatro roles: `G0` = RoutingActor `000000`; `GD` = RoutingActor `111111`; `PPO` = GenericActor con el perfil PPO estándar seleccionado históricamente; `SAC` = SAC estándar 2.3.2. Dos entradas: `raw` y `standardized`. Tres bloques independientes de desarrollo, `rep=0,1,2`. Se efectúan **24 entrenamientos**, cada uno hasta **1.048.576 transiciones nativas**, conservando además un checkpoint después de **524.288** transiciones y sus actualizaciones correspondientes.

Los dos presupuestos son medidas anidadas del mismo entrenamiento, no entrenamientos independientes; el mayor continúa incondicionalmente hasta su tope sin consultar el resultado inferior. Misma semilla de inicialización dentro del par raw/standardized de cada rol y bloque. Misma secuencia de identidades ambientales por bloque/slot entre roles compatibles; ocho flujos PPO y uno SAC siguen siendo muestreos distintos. Misma distribución, no trayectorias de entrenamiento idénticas.

Se evalúan los dos checkpoints sobre los mismos **32 episodios reservados de desarrollo por bloque**, emparejados con un testigo `no-action`. Total: 16 celdas descriptivas, tres medias de entrenamiento por celda, 1.536 episodios de modelos y 96 de testigo. Ni episodios, checkpoints ni pasos se cuentan como réplicas independientes. Tres bloques solo dan un diagnóstico piloto de precisión limitada; no una certificación de competencia poblacional o estabilidad estructural.

## 3. Normalización como único factor de entrada

Se ajusta un normalizador estático por bloque, común a los cuatro roles. Antes de entrenar, se recogen **32.768 transiciones nativas por bloque**: 16.384 con acción cero y 16.384 con acciones uniformes en [-1,1], usando identidades exclusivas de calibración de entradas. Solo se incorporan las observaciones crudas anteriores a cada acción. Se registran observaciones, acciones, identidades y momentos de reset. No se usan checkpoints históricos, resultados reservados, recompensas ni éxito para estimar o elegir el normalizador.

Para cada una de las 93 coordenadas: `z'=(z-media)/max(desviacion_poblacional,0.01)`, con estadísticas float64 y salida float32. No hay clipping ni eliminación de coordenadas. Las estadísticas quedan congeladas antes del primer entrenamiento y no se actualizan durante entrenamiento o evaluación. No se normalizan recompensas; las métricas externas siempre se calculan en unidades nativas. La entrada del actor y del crítico usa la misma transformación, también el replay SAC, por lo que no cambia su interpretación con el tiempo. Los identificadores one-hot internos de los nodos no se normalizan.

La distribución de esta calibración puede no cubrir estados competentes. Se informan magnitudes fuera de escala y saturación, sin recalibrar después de observar resultados. Esta receta prueba normalización estática específica, no todas las normalizaciones posibles.

## 4. Invariantes y perfiles

Se importan sin editar `b1s/execution/core.py` e `instrument.py` de la fuente histórica y se fijan sus hashes. Se preservan Multiwalker de tres procesos, raw93, doce acciones, física, recompensa, reglas de terminación, una ronda de mensajes, todos los mensajes calculados y denominador de agregación dos. Ni información local restringida ni memoria contextual nuevas.

G0/GD conservan PPO común: lr 0.0003, gamma 0.99, rollout 512 total en 8 flujos, minibatch 128, 4 épocas. PPO genérico conserva lr 0.0003, gamma 0.99, rollout 2048, minibatch 64, 10 épocas. GAE 0.95, clip 0.2, coeficiente de valor 0.5, clipping de gradiente 0.5, Adam eps 1e-5. SAC conserva lr 0.0003, gamma 0.99, batch 256, learning_starts 10000, train_freq=1, gradient_steps=1, tau=0.005, entropía auto y redes [64,64] ReLU; replay de 1.048.576 entradas durante ambos horizontes. El tratamiento de timeouts y las demás diferencias algorítmicas nativas se conservan y se declaran; no se afirma capacidad o computación igual.

No hay búsqueda de hiperparámetros adicional. Las comparaciones de normalización y presupuesto son dentro de cada rol; entre roles persisten diferencias de arquitectura, actualización y recursos.

## 5. Evaluación y regla de competencia

Evaluación determinista, física/recompensa nativas, normalizador congelado, sin actualizaciones del modelo. Checkpoints medidos después de las actualizaciones del presupuesto: en SAC no se evalúa desde el callback previo al update. Se conserva y restaura el estado de generadores de entrenamiento alrededor de la evaluación. No se elige el checkpoint por desempeño; ambos se evalúan y el presupuesto mayor se completa siempre salvo fallo técnico retenido.

La conjunción descriptiva es la histórica: mejora media de pérdida respecto al testigo **>130/30** y desplazamiento medio adicional **>=1**. La pérdida es menos retorno. El testigo se mide de nuevo en el panel emparejado: no se importa su desplazamiento de v1.1 como umbral absoluto. Se conserva simultáneamente caída, pérdida del paquete, duración, esfuerzo solicitado y conteos terminales.

Se muestran las tres medias por bloque, agregado y número de bloques que cumplen la conjunción. Se informan contrastes raw/standardized a cada presupuesto, mayor/menor por entrada y diferencia de diferencias, emparejados por bloque, sin prueba confirmatoria, p-value, búsqueda del mejor soporte ni actualización del dictamen histórico.

Resultado computacional completo: `PILOT_COMPLETE_FOR_REVIEW`, independientemente del signo o del cumplimiento. Si falta un ajuste/checkpoint/evaluación o un hash falla: `PILOT_BLOCKED_WITH_RETAINED_EVIDENCE`, sin completar artificialmente celdas. Los indicadores descriptivos no autorizan continuación automática. Cualquier propuesta posterior de búsqueda estructural requiere decisión explícita tras revisar estos resultados.

## 6. Custodia y límites de ejecución

Namespace exclusivo `CP-NB-1.0.0-20260914`; particiones `normalization`, `training`, `evaluation` y `qa`. Nunca se acepta `final` ni B1-E. Antes de entrenar se publica un freeze de desarrollo con protocolo/plan/código, runtime, registro de los 24 ajustes, estadísticas de normalización y huellas. Una petición inicial inmutable habilita solo un run, intento 1; una repetición o un freeze existente bloquea el entrenamiento. No hay reemplazo de semillas ni reintento científico automático.

Tope: **25.165.824 transiciones de aprendizaje +98.304 de calibración de entradas**; como máximo **816.000** de evaluación. Máximo concurrente cuatro ajustes CPU. Se reservan hasta 180 minutos por ajuste y 30 minutos para preparación/cierre; los timeouts son incidentes, no autorización para extender. QA es una partición separada y se registra fuera de la evidencia empírica.

Se retienen ambos checkpoints, optimizer cuando corresponda, estadísticas, runtime, registros de episodios, curvas, trazas de evaluación y manifiestos. No se guardan todos los minibatches ni replay SAC completo; no se garantiza réplica bit a bit entre hardware. Los binarios se retienen en artefactos separados por ajuste, con metadatos y huellas, 90 días; el cierre textual y el inventario de artefactos se publican en Git. Esa retención es temporal y debe conservarse fuera de Actions antes de vencer; no se promete permanencia de binarios en Git. Ningún ZIP único gigante se envía como un solo commit de resultados.

## 7. Estados separados y referencias

`B1E_executed=false`; `final_seeds_generated=false`; `ready_for_b1e_protocol_design=false`; `ready_for_b1e_freeze=false`; `ready_for_b1e_confirmatory_run=false`; `H_CAT=NOT_EVALUABLE`; `H_TRANSFER=NOT_EVALUATED`.

Este ensayo no identifica una estructura correcta, no mide estabilidad de descubrimiento y no valida consciencia. Estudia si estos cuatro controladores alcanzan competencia al variar de forma controlada la escala de entrada y el presupuesto. Un hallazgo adverso o inconcluso se conserva.

Base interna: protocolo, design.py y train.py históricos en `50c4e08cf3d69c998ce95b8da1e1647a09b31339`. Contexto metodológico: documentación oficial SB3 2.3.2, RL Tips and Tricks y OffPolicyAlgorithm (consultada el 14-09-2026). La recomendación general de normalizar y evaluar por separado no valida los números ni garantiza un resultado. Las cifras de este piloto son decisiones prospectivas de ingeniería.
