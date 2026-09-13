# Candidato a protocolo B1-E v2: diseño prospectivo fijo de medias pareadas

El tamaño de muestra propuesto es de **138,688 paquetes P7 independientes**, fijado antes de generar semillas finales. El diseño v1 congelado permanece en **915,501 paquetes por piloto**. Se trata de un nuevo candidato a protocolo prospectivo, no de una modificación de las observaciones históricas. El trabajo estadístico no ejecutó ningún modelo Polar, experimento B1-E ni generador de semillas finales. Todas las salidas adjuntas de validación del método son exclusivamente de desarrollo, no confirmatorias e inutilizables como evidencia final.

## Alcance científico y programa

Las decisiones de elegibilidad proceden de `B1_DECISION_RULES_v1.yaml` congelado, no del signo de las diferencias de desarrollo de P7. P5 y P6 conservan cada uno sus nueve resúmenes de variables de resultado registrados como controles de ingeniería/negativos, con 32 paquetes prospectivos de ingeniería por piloto. Su elegibilidad para utilidad positiva y especificidad positiva del emparejamiento permanece en falso. Esta cantidad de repeticiones de ingeniería no respalda una afirmación poblacional con potencia estadística establecida. P7 conserva sus nueve variables de resultado registradas con intervalos: dos efectos del mecanismo fuente, su interacción de contexto, cuatro diferencias pareadas ordinarias entre comparadores y dos diferencias de intervención aguda sobre la ruta. C5 sigue siendo descriptivo y C6 sigue siendo un control de consistencia algebraica.

P7 recopila los 12 episodios de cada paquete antes de contrastar cualquier hipótesis: C0, C1, ambos ciclos C2, C3, C4, C5, C6 y realizaciones intactas/con omisión de ruta en ambos contextos de intervención aguda. También conserva el programa diagnóstico de fuente de 24 eventos. P6 utiliza el mismo programa de ingeniería de 12 episodios. P5 añade el testigo del techo físico de reposición completa, identificado por separado, hasta alcanzar 13 episodios. Las 18 variables de resultado de P5/P6 no se eliminan silenciosamente ni se agrupan con P7.

## Inferencia seleccionada y márgenes conservados

La inferencia utiliza la propia diferencia pareada a nivel de paquete, con una observación por paquete independiente. Los dos ciclos de C2 se promedian dentro de ese paquete. Los trabajos, las épocas, las rutas repetidas y los diagnósticos de fuente dentro de un paquete no se tratan como réplicas independientes.

Para `N >= 2`, varianza muestral insesgada `s²`, amplitud conocida del soporte `R`, tamaño de familia `m = 9` y error familiar `alpha = 0.05`, se utiliza

```
t = log(4*m/alpha) = log(720)
h = sqrt(2*s²*t/N) + 7*R*t/[3*(N-1)]
CI = clip([sample_mean-h, sample_mean+h], known support)
```

La cota se obtiene reescalando la desigualdad unilateral y asignando error a ambas colas. No sustituye la varianza por una varianza de desarrollo. Véase [Maurer y Pontil, teorema 4](https://arxiv.org/pdf/0907.3740).

| Grupo de variables de resultado | Soporte | Delta práctico | Épsilon de equivalencia | Semiamplitud objetivo |
|---|---|---:|---:|---:|
| Utilidad e intervención aguda sobre la ruta | [-1, 1] | 1/64 | 1/128 | 1/256 |
| Mecanismo fuente | [-1, 1] | 1/2 | 1/4 | 1/8 |
| Interacción de contexto | [-2, 2] | 1/2 | 1/4 | 1/8 |

Estos son los márgenes de ingeniería existentes. Ninguno se modificó para reducir N. Una clasificación favorable de utilidad requiere `CI.upper < -1/64`; el empeoramiento requiere `CI.lower > 1/64`; la equivalencia requiere que todo el intervalo esté estrictamente dentro de `(-1/128, 1/128)`. La falta de precisión permanece como inconclusa. La equivalencia exige positivamente la inclusión del intervalo, no la falta de rechazo de un efecto cero; compárese con el [principio de dos pruebas unilaterales](https://doi.org/10.1007/BF01068419).

## Multiplicidad y requisitos para afirmaciones positivas

Los nueve intervalos P7 son simultáneos. La probabilidad de fallo de cada intervalo es como máximo `0.05/9`; la cota de unión controla cualquier fallo de cobertura en 0.05 sin requerir independencia entre variables de resultado. Toda declaración escalar incorrecta de mejorado/equivalente/empeorado implica un fallo de cobertura de su intervalo correspondiente. Exigir varias comparaciones favorables y aplicar una jerarquía únicamente elimina declaraciones de ese suceso. Esto proporciona protección fuerte del error familiar para las clasificaciones escalares y sus afirmaciones compuestas.

Las afirmaciones positivas deben superar, en este orden: instrumento y mecanismo local admisible; utilidad práctica frente a C0 y C4; especificidad del emparejamiento frente a C0/C2/C3/C4; moderación por contexto. Todas las observaciones se recopilan con N fijo aunque falle un requisito positivo anterior. Los informes negativos y de equivalencia siguen visibles. Se trata de un orden científico fijo de requisitos sobre intervalos simultáneos; **no** recicla alfa ni reivindica los valores críticos menores de un procedimiento de reciclaje de alfa. Las pruebas ordenadas pueden controlar FWER, pero importa la estructura exacta de las hipótesis; véase [Edwards y Madsen](https://doi.org/10.1002/sim.2905).

Fuente/simulada permanece en `joint_manipulation=true`. La restricción congelada de admisibilidad causal sigue activa, de modo que la elegibilidad no autoriza por sí misma una afirmación causal positiva de fuente. El diagnóstico de fuente no puede demostrar necesidad selectiva de Gamma. Los contrastes de intervención aguda sobre la ruta son distintos; ningún resultado de ruta puede subsanar requisitos fallidos de utilidad práctica o emparejamiento. Un PASS del instrumento no es un PASS causal.

**Consecuencia para la realización actual:** como los casos de prueba fuente/simulada son conjuntos, `source_causal_admissible=false`; esta jerarquía no puede emitir ninguna afirmación positiva agregada H_mechanism, H_utility, H_pairing o H_context, sean cuales sean los signos de las diferencias escalares futuras. Las salidas correctas siguen siendo not_supported/inconclusive donde corresponda, y los resultados escalares corregidos improved/equivalent/worsened/inconclusive pueden seguir informándose por separado. Esta es una limitación explícita de las afirmaciones científicas. No es una promesa oculta de que pueda obtenerse un veredicto agregado favorable bajo los casos de prueba actuales, ni un motivo para relajar una restricción o reajustar P7. Cualquier rediseño causal futuro necesitaría otro protocolo prospectivo. La finalización técnica puede coexistir con esta vía válida de resultados negativos/inconclusos.

## Planificación conservadora con desarrollo, no varianza conocida

`statistics/build_design.py` reconstruye independientemente los seis contrastes P7 de margen estrecho a partir de los 384 ensayos de calibración archivados (32 paquetes, 12 episodios cada uno), utilizando aritmética racional exacta. Verifica el conjunto completo de identidades de comparador/ciclo/contexto/lesión y coincide con el informe v1 congelado. Cada uno de los seis contrastes contiene 32 valores cero. Los ensayos originales, el mapa de identidades, la configuración seleccionada, el registro de ajuste, el cierre técnico y los resultados se referencian mediante repositorio, commit, ruta y SHA-256 en el artefacto JSON.

Para una diferencia pareada `D` en [-1,1], `E[D²] <= Pr(D != 0)`. Con cero sucesos distintos de cero en 32 paquetes de desarrollo iid, una cota superior binomial exacta unilateral, corregida por Bonferroni para seis variables de planificación, es

```
q = max(1/64, 1 - (0.05/6)^(1/32))
  = 0.13895552166407366
```

El mínimo declarado de varianza es 1/64. La varianza cero observada no se trata como cero conocido. La demostración elemental es `Pr(all 32 events absent) = (1-p)^32`: valores `p > q` producirían esa observación con una probabilidad inferior a `0.05/6`. La cota conjunta de desarrollo puede fallar con probabilidad como máximo 0.05. Si alguna cantidad de sucesos de desarrollo fuera distinta de cero, la implementación fijada utiliza el techo conservador 1 para el segundo momento, en lugar de ajustar un sustituto conveniente.

Para una muestra futura independiente de paquetes, Hoeffding aplicado a `D² in [0,1]`, seguido de una cota de unión para seis variables de resultado, proporciona con probabilidad al menos 0.95, **condicionada a cotas válidas del momento de desarrollo y a una ley de paquetes sin cambios**,

```
s² <= Vplan(N)
Vplan(N) = N/(N-1) * min(1, q + sqrt(log(6/0.05)/(2*N)))
```

La desigualdad utiliza `s² <= N/(N-1)*mean(D²)` y nunca supone que la media desconocida sea cero. El menor entero que satisface la semiamplitud de utilidad conservada, junto con las condiciones menos exigentes de peor soporte para fuente/contexto, es N = 138,688. Con ese N, `Vplan = 0.14311106135429483` y el radio planificado es `0.0039062395446212454`. Con N−1 es `0.003906254618612961`, superior al `0.00390625` requerido. No existe un límite elegido por conveniencia computacional. La alternativa conservadora de la regla para el peor segundo momento requeriría 878,007 paquetes.

La garantía conjunta de planificación de desarrollo y futuro es de **al menos 90%**, utilizando una cota de unión conservadora para dos errores del 5%; no es una promesa incondicional del 95% de precisión. La precisión futura condicional es al menos del 95%. La cobertura inferencial familiar permanece en al menos 95% independientemente de que la cota de planificación sea correcta, porque el CI final siempre utiliza la varianza muestral final real. N, adaptado con información de desarrollo, se fija antes de la muestra final independiente.

Este es un diseño de precisión. Condicionado a cotas válidas de momentos, combinar los sucesos de precisión del 95% y cobertura del 95% proporciona una garantía conservadora del 90% de clasificación para medias suficientemente separadas, con requisitos estrictos de inclusión del intervalo para la equivalencia. No promete potencia en la frontera del margen práctico ni para toda distribución acotada de alta varianza. Los resultados sintéticos de potencia que aparecen a continuación son específicos de cada escenario.

La información de planificación solo se transfiere a la misma tarea P7, las mismas políticas seleccionadas y la misma ley iid de paquetes. La certificación de recursos debe verificar que el comportamiento no cambia. Un cambio de tarea o política invalida esta derivación antes de la congelación final. Si un intervalo futuro no alcanza su objetivo de precisión, debe conservarse como inconcluso: no ampliar N, cambiar un margen, sustituir una semilla ni reemplazar la varianza por un mínimo en ese CI.

## Comparación de candidatos

| Candidato | Decisión y motivo |
|---|---|
| Bernstein empírico de muestra fija | Seleccionado: la inferencia de medias acotadas con muestra finita admite la varianza pareada sin supuestos gaussianos. |
| CI t/normal pareado | No seleccionado para las garantías primarias: la t exacta requiere diferencias gaussianas; la validez por TCL es asintótica y no proporciona esta garantía de muestra finita. |
| Permutación/aleatorización pareada | Compartir semillas por sí solo no implica intercambiabilidad de signos ni etiquetas de tratamiento aleatorizadas. Las pruebas de una hipótesis nula fuerte no contrastan automáticamente una hipótesis nula débil de media/margen con efectos heterogéneos; véase [Chung y Romano](https://arxiv.org/abs/1304.5939). |
| Métodos exactos/binomiales | Utilizados únicamente para el indicador de valor distinto de cero entre paquetes durante la planificación. Una fracción de pérdida de 192 trabajos no es Binomial(192,p), porque los trabajos comparten estado e historia. |
| Equivalencia/no inferioridad | La equivalencia utiliza la inclusión de CI simultáneos con épsilon sin modificar. La no inferioridad por sí sola no establecería superioridad de utilidad ni necesidad del emparejamiento. |
| Holm, pruebas cerradas, secuencia fija, reciclaje | Potencialmente válidos con hipótesis elementales especificadas. Se seleccionaron intervalos simultáneos y requisitos restrictivos para conservar informes rigurosos negativos y de equivalencia después de fallar requisitos positivos. |

## Validación estadística

La validación actual contiene 40 condiciones con 20,000 repeticiones sintéticas cada una. Prueba N = 138,688 y una condición de estrés de 32 observaciones, coordenadas de variables de resultado independientes y perfectamente dependientes, nueve escenarios de leyes de utilidad que incluyen ambas fronteras de equivalencia y un caso de reescalado del soporte de contexto. Se muestrean estadísticos suficientes multinomiales exactos de leyes acotadas declaradas. El simulador no aproxima las muestras mediante una distribución normal ni importa ninguna tarea Polar, controlador o API de semillas finales.

| N = 138,688; coordenadas independientes | Cobertura familiar | Clasificación escalar correcta | Las nueve alcanzan la precisión |
|---|---:|---:|---:|
| Efecto cero | 0.99980 | 1.00000 | 1.00000 |
| Frontera del margen práctico | 0.99990 | 0.00000 | 1.00000 |
| Frontera positiva de equivalencia | 0.99995 | 0.00000 | 1.00000 |
| Frontera negativa de equivalencia | 1.00000 | 0.00000 | 1.00000 |
| Efecto favorable | 0.99985 | 1.00000 | 1.00000 |
| Efecto adverso | 0.99990 | 1.00000 | 1.00000 |
| Colas extremas raras acotadas | 0.99865 | 0.00000 | 0.00000 |
| Ley discreta de fracciones de trabajos | 1.00000 | 1.00000 | 1.00000 |
| Mezcla de varianzas heterogéneas | 0.99880 | 1.00000 | 1.00000 |
| Frontera de contexto con amplitud 4 | 0.99835 | 0.00000 | 1.00000 |

Entre todas las condiciones, la cobertura familiar mínima fue 0.99835 y la máxima probabilidad observada de declaración falsa fue 0.001 (20/20,000 en el caso de frontera de contexto). La frecuencia de mejora falsa en la frontera práctica de utilidad fue 1/20,000. El JSON registra la incertidumbre binomial exacta de Monte Carlo para cada tasa. Cero errores observados no implica probabilidad de error cero. La varianza de colas intensas supera la cota de planificación y, correctamente, incumple el requisito de precisión; no provoca un CI menor mediante el uso de la varianza de desarrollo.

Se aplica una comprobación de regresión exacta de cola superior Binomial(0.05) a la falta de cobertura y a tres tasas de declaraciones falsas de cada condición, con presupuesto de falsas alarmas de QA `0.01/(4*40)`. La demostración analítica es la garantía primaria; la simulación comprueba estas implementaciones y estas distribuciones. El artefacto original de QA sintético más pequeño y sus instantáneas de diseño y fuentes permanecen archivados como revisión 1. La revisión 2 añade las pruebas de frontera negativa y reescalado de amplitud de revisores independientes y la comprobación de regresión del error nominal; no modifica N ni el método de inferencia.

## Reproducción y requisitos pendientes

Ejecute desde el repositorio de código:

```
python -B -m unittest b1.v1_1.statistics.test_statistics -v
python -B -m b1.v1_1.statistics.simulate --smoke
```

Hay 11 comprobaciones estadísticas unitarias/de contrato, que incluyen la reducción exacta de datos archivados, la minimalidad de N fijo, la incertidumbre con varianza cero, el reescalado del soporte, el rechazo de entradas faltantes, el comportamiento no vinculante con alta varianza, la elegibilidad de controles negativos, las restricciones de causalidad conjunta y los hashes de fuentes. El diseño guardado se reproduce sin escribir. La simulación estadística completa solo puede escribirse en una ruta nueva; la salida archivada nunca se sobrescribe silenciosamente.

La viabilidad de recursos y la conservación se certifican por separado. Este candidato no autoriza B1-E. El prerregistro externo, la congelación final del código, la configuración y la conservación, la custodia independiente de semillas, la entropía final y la autorización explícita de recursos y ejecución siguen siendo requisitos futuros. `ready_for_b1e_confirmatory_run=false`.
