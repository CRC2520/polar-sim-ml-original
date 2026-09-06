# Avance de la investigación: segunda etapa

Este informe integra los seis pasos acordados después de la primera evaluación. La propuesta sigue siendo una **investigación no publicada**. El repositorio privado `CRC2520/POLAR_MODEL_CRC` contiene un solo manuscrito vigente: `main.tex`, `body.tex` y `references.bib` son sus fuentes, y `main.pdf` es su salida compilada. Las redacciones anteriores se conservan en el historial de Git, no como artículos alternativos. Los protocolos, datos y especificaciones de este repositorio son evidencia complementaria de ese mismo documento.

El equilibrio dinámico mantiene su significado: regularse según las demandas y conservar acceso a cualquiera de los polos, a su coactivación y a la inactividad. La evaluación distingue esta capacidad de una tendencia al consenso o a la uniformidad.

## Los seis pasos y su evidencia

| Paso | Trabajo realizado | Evidencia y límite de interpretación |
|---|---|---|
| **1. Formular una hipótesis distintiva y refutable** | Se especificó una intervención matemática: usar los efectos cruzados aprendidos dentro de cada pareja al planificar acciones. El contraste principal desactiva esos efectos solo en la planificación y conserva la arquitectura del estimador, memoria, información disponible y restricciones. | `STUDY2_CONTROLLER_SPEC.md`, `STUDY2_PROTOCOL.md` y sus pruebas. Se contrasta el efecto funcional de una política; las acciones posteriores cambian las observaciones y pueden producir estimaciones distintas. Los nombres de los polos y el cambio de coordenadas no son el mecanismo causal. |
| **2. Explicar los resultados desfavorables previos** | Se identificaron los dos fallos exactos y se ejecutaron 300 diagnósticos emparejados, incluidas intervenciones que empeoran el desempeño. Se reprodujeron exactamente 90 resultados de referencia comprobados. | `STUDY1_DIAGNOSIS.md` y `results_study2/diagnosis/`. La memoria almacena correctamente el objetivo; su confianza atenúa la respuesta y el estimador puede distorsionarse con acciones pequeñas y ruido. Son datos ya observados y análisis exploratorios. |
| **3. Registrar un nuevo protocolo antes de evaluar** | Se separaron comprobaciones mecánicas, un piloto de 288 ensayos y la evaluación final. La regla de precisión basada solo en la variabilidad del piloto fijó 40 semillas nuevas. Código, protocolo y criterios se registraron públicamente antes de generar los resultados finales. | Registro Git [`fde8d0ac9b56b035a0a53c40c8cd5b79b2bc6b32`](https://github.com/CRC2520/polar-sim-ml-original/commit/fde8d0ac9b56b035a0a53c40c8cd5b79b2bc6b32), `STUDY2_FREEZE.json` e `INTERNAL_REVIEW_STUDY2.md`. Es un registro prospectivo en el repositorio, no revisión por pares externa ni garantía de potencia estadística. |
| **4. Comparar el mecanismo con controles adecuados** | Se diseñaron dos familias dinámicas, tres regímenes estructurales y ocho controladores. Se comparan efectos aprendidos por parejas, ablación de planificación, estimación diagonal, parejas desplazadas, estimación densa, coordenadas signo–intensidad y dos controles negativos. | El diseño final comprende 40 semillas × 2 familias × 3 regímenes × 8 controladores: 1.920 ensayos de 128 transiciones. Los datos, acciones y actualizaciones se conservan y los informes se calculan desde las trazas. El estimador denso tiene más coeficientes; no se afirma igualdad de capacidad total entre todos los comparadores. |
| **5. Decidir según criterios previos** | La regla combina mejora mínima relevante, incertidumbre, costos, fallos, restricciones, consecuencias de inventario y comparaciones estructurales. La auditoría posterior detectó una discrepancia entre la condición textual y el código que asigna la etiqueta final; se conserva y declara esa desviación. | `results_study2/coupling/final/generated/decision.json` conserva resultados y clasificación originales; `results_study2/adjudication.json` registra por separado la adjudicación conforme al texto. No se alteran datos, umbrales ni intervalos para corregirla. |
| **6. Dar un protocolo propio a las funciones relacionadas con conciencia** | Se definieron por separado experiencia subjetiva, acceso global a contenido, modelo funcional de capacidades y metacognición. Se ejecutaron sondas exploratorias de memoria, predicción acción–efecto y asignación de recursos con diez semillas. | `CONSCIOUSNESS_RESEARCH_PROTOCOL.md` y `results_study2/functional/`. Se demuestran funciones limitadas y se identifican funciones ausentes. No existe un clasificador, porcentaje ni prueba validada de conciencia. |

## Qué explican los fallos del primer estudio

Las semillas 1012 y 1020 fallaron únicamente el máximo de pérdida de recuerdo de 0,015: alcanzaron 0,015247 y 0,015101, respectivamente. No hubo violaciones de restricciones ni divergencia numérica.

Durante el recuerdo, el error del valor almacenado fue cero, pero la confianza multiplicativa descendió de 0,797687 a 0,739910. Además, dividir efectos observados por acciones muy pequeñas contaminó algunas ganancias estimadas: llegaron a 2,945 y 3,505 en canales cuya ganancia real era uno. El planificador redujo la acción ante esas estimaciones excesivas.

El recuerdo medio pasó de 0,012783 a 0,010689 al reiniciar la capacidad al comienzo del recuerdo. Reiniciar solo la ganancia, conservando conteos e incertidumbre, produjo un resultado similar. En cambio, congelar las estimaciones sin corregirlas empeoró la pérdida a 0,020983 y ninguna de las 30 ejecuciones aprobó. La retroalimentación posterior estaba corrigiendo parte del error heredado.

Esto no justifica eliminar toda estimación de capacidades: en la tarea con cambios reales de ganancia, esa eliminación elevó la pérdida media de 0,004625 a 0,013432. Tampoco justifica fijar siempre la confianza en uno. Las intervenciones identifican mecanismos bajo condiciones controladas; no son soluciones generales validadas en entornos desconocidos.

## Evidencia prospectiva del segundo estudio

El registro público se creó el **6 de septiembre de 2026 a las 02:20:56 UTC**. La ejecución final comenzó a las **02:21:51 UTC** y utiliza las semillas **62001–62040**. El piloto empleó 52001–52006; una comprobación mecánica anterior utilizó 51999. La evaluación final no se utilizó para modificar el diseño ni sus criterios.

Las familias son seguimiento dinámico e inventario de capacidad limitada, este último con faltantes y desbordes calculados como consecuencias físicas. Los regímenes incluyen efectos alineados con las parejas propuestas, dinámica diagonal y relaciones desalineadas. Son entornos diseñados dentro del proyecto. El cambio de régimen también cambia objetivos y condiciones de la planta; por ello, comparar regímenes no aísla exclusivamente la topología.

La implementación utiliza **dos grupos de canales dentro de un único planificador centralizado de ocho acciones**. No son dos agentes autónomos que negocian. Los dos tipos de acción son aproximaciones operacionales y no validan las ocho categorías filosóficas del borrador inicial.

El estimador utiliza mínimos cuadrados recursivos —RLS— y la planificación sigue métodos conocidos de control predictivo con restricciones. La contribución que se evalúa es una organización y un uso específicos de esos mecanismos, no la invención de esos métodos.

La evaluación final completó los **1.920 ensayos previstos**, separados de los 288 ensayos del piloto. Cada controlador tiene 240 ejecuciones finales. La pérdida media de seguimiento fue:

| Controlador | Pérdida media | Ejecuciones que cumplen los criterios de seguimiento |
|---|---:|---:|
| Efectos aprendidos por parejas | 0,007766826 | 240/240 |
| Mismo estimador, efectos cruzados desactivados en planificación | 0,010466629 | 240/240 |
| Estimación exclusivamente diagonal | 0,008767611 | 240/240 |
| Parejas desplazadas, con igual cantidad de coeficientes | 0,007571311 | 240/240 |
| Estimador denso | 0,005776053 | 240/240 |
| Coordenadas equivalentes signo–intensidad | 0,007766826 | 240/240 |
| Acción cero | 0,138770049 | 0/240 |
| Acción fija | 0,066705295 | 0/240 |

Estos conteos describen los ensayos realizados. La diferencia observada de fallos entre el controlador por parejas y su ablación fue cero y su intervalo de remuestreo degeneró en [0; 0]; ello no demuestra que la tasa poblacional de fallos sea cero.

El contraste principal —parejas menos ablación de planificación— fue **−0,002699802**, intervalo del 95 % **[−0,002828557; −0,002576647]**, con 40 semillas emparejadas. Equivale a una reducción descriptiva de **25,79 %** respecto de la pérdida media de la ablación. El límite superior del intervalo queda por debajo de −0,002, cumpliendo el requisito previo de mejora práctica. Hay, por tanto, evidencia de que **usar los efectos cruzados aprendidos mejora esta política frente a su propia ablación**, bajo los entornos ensayados.

Los siete controles de tolerancia se cumplieron: fallos, costo de acción, retorno al contexto inicial, faltantes, desbordes, restricciones duras y tiempo de ejecución. Cumplirlos no significa que todas las medidas mejoren: el cambio de pérdida en la fase de retorno empeoró en 0,002237 frente a la ablación y los faltantes de inventario aumentaron en 0,000119; ambos permanecieron dentro de sus tolerancias previas. El costo medio de acción fue un 3,39 % menor. No hubo violaciones duras en el controlador por parejas.

**La conclusión conforme al texto del protocolo es suspender la afirmación de ventaja polar exclusiva: `suspend_exclusive_polar_advantage_claim`.** No se aprobó la continuación amplia prevista en el protocolo, porque:

- El controlador de parejas desplazadas, con los mismos 16 coeficientes dinámicos, obtuvo menor pérdida. La diferencia parejas menos desplazadas fue **+0,000195515**, intervalo **[+0,000133561; +0,000261025]**. La agrupación originalmente propuesta no ganó esta comparación estructural.
- El estimador denso también obtuvo menor pérdida. Fuera del régimen alineado, la diferencia fue **+0,001498288**, intervalo **[+0,001419792; +0,001574459]**; incumplió el margen de no inferioridad de 0,001. El denso utiliza 64 coeficientes frente a 16, por lo que esta comparación informa el costo de restringir la estructura y no demuestra superioridad a igual capacidad.
- La transformación a signo–intensidad conservó las acciones hasta una diferencia máxima aproximada de **9,99 × 10⁻¹⁶**, compatible con redondeo numérico. No hay ventaja debida al nombre o a las coordenadas de los polos.

La auditoría detectó, **después de ejecutar el estudio, un fallo del clasificador congelado**. El exportador devolvió `pivot_to_conditional_structural_prior`, pero el protocolo reservaba esa etiqueta para un beneficio práctico presente **solo** en el régimen alineado. El código comprobaba que hubiera beneficio alineado y omitía exigir su ausencia en los demás regímenes. Los datos muestran beneficio práctico también en el régimen desalineado: −0,002553, intervalo [−0,002736; −0,002376]. Por ello, la etiqueta automática no corresponde literalmente a la condición textual. Se preservan tanto el código y JSON congelados como la discrepancia; la adjudicación se registra por separado, sin modificar pérdidas, intervalos, umbrales ni controles. Es un error concreto que las pruebas previas no detectaron, no una ejecución completamente conforme a su clasificación especificada.

La adjudicación se documentó el 6 de septiembre de 2026 a las **02:32:51 UTC** en `results_study2/adjudication.md` y `adjudication.json`, reproducibles con `python scripts/adjudicate_study2.py`. El JSON tiene SHA-256 `4ff0e11f4ea35468debda775e4256577ed32309eb8d399c0275551f2bc1ce70a`. Esta corrección de interpretación es posterior al estudio; no es una nueva prueba confirmatoria ni una modificación retroactiva del registro.

Tampoco sería válido traducir la etiqueta automática como si ya se hubiera demostrado la superioridad de las parejas correctas. Lo respaldado es un efecto de usar interacciones aprendidas en esta política; sigue abierta la pregunta de cómo elegir una estructura que mejore alternativas de igual tamaño. Investigar esa pregunta es una propuesta futura acotada, **no** el cumplimiento del requisito registrado para continuar la hipótesis más amplia.

Las familias tampoco respondieron igual. La diferencia principal fue −0,004814 en seguimiento dinámico y −0,000586 en inventario. La mejora de inventario, aunque favorable, quedó por debajo de la diferencia mínima relevante de 0,002. Estos análisis secundarios son descriptivos y no sustituyen el criterio principal agregado.

Los resultados completos están en `results_study2/coupling/final/generated/results.md` y `decision.json`. Este último registra el SHA-256 del manifiesto final: `147c2a0adc054895b2787684a20379b4a3f3d9676bff4b69c6a39afad8dd7ede`.

## Qué establecen las sondas funcionales

Las diez semillas 73001–73010 corresponden a diagnósticos exploratorios separados de la evaluación prospectiva anterior. Sus intervalos del 95 % son descriptivos y se calculan por semilla.

| Función intervenida | Resultado observado | Conclusión funcional |
|---|---|---|
| Memoria después de retirar el estímulo y reiniciar el estado de acción | El error de recuerdo fue 0,052050 con memoria intacta y 0,288013 después de borrarla; aumento emparejado de 0,235963, intervalo [0,231253; 0,240604]. | El registro interno conserva información que influye en la acción. No demuestra memoria autobiográfica ni experiencia de recordar. |
| Predicción de los efectos propios | El error de predicción fue 0,000208 con el estimador entrenado y 0,007719 después de reiniciarlo. | La estimación aprendida aporta capacidad predictiva bajo el contrato experimental. |
| Contaminación con efectos provenientes de otra fuente | El error subió a 0,026022; la medida heurística de incertidumbre no identificó esa procedencia equivocada. | El modelo recibe el emparejamiento acción–efecto del experimentador; no atribuye por sí mismo quién produjo un efecto. |
| Permutación de contenidos que mantiene la demanda agregada | La diferencia máxima de asignación fue del orden de 10⁻¹⁶. | El asignador responde a demanda agregada. No se ha implementado difusión de contenido a consumidores independientes ni una prueba positiva de acceso global. |

La metacognición calibrada, la atribución propia/ajena y el acceso global a contenido quedan definidos como objetivos futuros con intervenciones y controles propios. Ninguno debe darse por implementado por usar términos como «incertidumbre», «espacio global» o «modelo de sí mismo».

## Verificación, alcance y reproducibilidad

Antes de generar resultados finales, la batería completa aprobó **63 pruebas**: las 38 anteriores y 25 nuevas. La revisión interna adicional comprobó 18 pruebas del segundo estudio y reprodujo 12 ensayos del piloto —1.536 transiciones— desde observaciones y retroalimentación registradas, con diferencia máxima cero en acciones y campos mecanísticos auditados. Es revisión interna entre implementadores y revisores del proyecto, no revisión externa o replicación en otro laboratorio.

Después de la ejecución, un revisor interno regeneró los resultados de los 1.920 ensayos desde sus trazas. Los valores, textos y gráficos PNG derivados coincidieron exactamente con los exportados; los hashes de los 11 archivos congelados permanecieron iguales. Esa comprobación confirma reproducción e integridad de los resultados, aunque no subsana por sí misma el fallo de clasificación detectado y declarado arriba.

La unidad inferencial principal del segundo estudio es la semilla, con igual peso para las seis combinaciones de familia y régimen. Las 1.920 ejecuciones, los canales y los pasos temporales no se contabilizan como 1.920 observaciones independientes. Se emplean 5.000 remuestreos emparejados por semilla. La diferencia mínima relevante se fijó en 0,002 unidades de error cuadrático como criterio de ingeniería; no tiene significado biológico.

Los manifiestos conservan versiones, hashes, índices completos de archivos y criterios de decisión. Los resúmenes y figuras se regeneran desde trazas completas; lo ausente no se convierte en cero. Los tiempos de ejecución son mediciones de esta implementación con instrumentación, no conteos de operaciones ni una comparación universal de eficiencia.

La propuesta no se comparó con LLM de frontera, no realizó experimentos humanos y no analizó datos neuronales. Ningún resultado demuestra conciencia, ASI ni que la polaridad sea un principio físico universal. La evidencia tiene valor al identificar qué funciones ayudan, cuándo fallan y qué afirmaciones deben estrecharse.

## Siguientes hitos derivados del resultado

No se ha iniciado otra optimización sobre las semillas finales. Estas ejecuciones quedan cerradas como evidencia de la segunda etapa; cualquier ajuste posterior deberá evaluarse con datos nuevos.

1. **Acotar la afirmación científica y declarar la desviación.** Mantener en el manuscrito que usar interacciones aprendidas mejora el controlador frente a su ablación en estos entornos. Suspender la afirmación de superioridad exclusiva de las parejas propuestas y mantener sin demostrar conciencia y universalidad. Conservar la clasificación automática discrepante junto con su adjudicación conforme al texto; no ocultar el fallo ni convertir el resultado en una aprobación general.
2. **Explicar la ventaja de las parejas desplazadas.** Realizar un nuevo diagnóstico exploratorio que separe sesgo estructural, identificación, observabilidad y distribución de objetivos. La comparación por régimen deberá mantener objetivos comunes o declarar explícitamente sus diferencias. Ese análisis podría sugerir cómo escoger relaciones; aún no constituye un método de selección validado.
3. **Diseñar una comparación de estructura a recursos equivalentes.** Contrastar una estructura aprendida o regularizada, grupos impuestos y controles genéricos, distinguiendo cantidad de coeficientes, incertidumbre, memoria y costo computacional. Conservar el denso como referencia de capacidad más amplia y añadir un control genérico de igual tamaño. Registrar de nuevo criterios, datos reservados y tolerancias antes de evaluar.
4. **Investigar los conflictos de desempeño.** Separar el error de predicción del cumplimiento de objetivos y de sus consecuencias. Probar el tratamiento de estimaciones con poca excitación y la confianza del recuerdo en entornos donde los recuerdos también puedan volverse incorrectos. Incluir faltantes y reacquisición como resultados propios; no ocultarlos dentro del promedio de seguimiento.
5. **Desarrollar únicamente las funciones de conciencia que puedan probarse.** El siguiente candidato concreto es una interfaz que distinga fuentes de efectos y un monitor que emita probabilidades de acierto calibrables antes de observar el resultado. Compararlo con estimadores convencionales igualmente informados. Para acceso global se necesitaría además transferencia de contenido a consumidores independientes, todavía ausente. Cada función requiere pruebas y controles propios; ninguna será un indicador automático de experiencia subjetiva.
6. **Buscar evaluación externa antes de ampliar el alcance.** Preparar una réplica con un entorno y un evaluador ajenos al diseño de estas tareas, sin revelar identidades de los controladores hasta cerrar el análisis. La revisión por especialistas de distintas teorías, cualquier registro independiente y cualquier validación neuronal permanecen pendientes; no se presenta ninguna de esas actividades como ya realizada.

Una continuación delimitada sobre **aprendizaje de interacciones, regulación contextual y estimación fiable de capacidades** puede formularse como nueva investigación. La segunda etapa aporta una mejora causal concreta, pero no aprobó los requisitos registrados para continuar con la afirmación amplia de ventaja polar. Esa afirmación queda suspendida; la conciencia artificial permanece sin demostrar.
