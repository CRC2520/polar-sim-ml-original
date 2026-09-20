# R9: contrato prospectivo de confirmación

Fecha de diseño: 2026-09-20. Criterios primarios acordados antes de los pilotos
de controladores y de las semillas finales. La ejecución final requiere además
congelar fuente, protocolos, configuración y sus hashes en un commit verificable.
Este registro de trabajo versionado no se presenta como preregistro externo.

R9 es una nueva realización acotada. Conserva los resultados R8, incluidos sus
seis hipótesis no apoyadas. Los cambios posteriores de arquitectura no alteran
retrospectivamente aquellos resultados. Ningún endpoint R9 mide conciencia
subjetiva, comprensión moral ni validez de un catálogo filosófico.

## 1. Unidad independiente y particiones

La unidad inferencial es una **semilla completa**, con sus adquisiciones,
episodios de selección, dominios de evaluación e intervenciones. Pasos,
dominios, cabezas, aristas y episodios dentro de una semilla no cuentan como
réplicas independientes.

- Desarrollo de controladores: 950101, 950102 y 950103.
- Finales: **952001–952080, n=80**, sin exclusiones ni reemplazos selectivos.
- Fixtures físicos y matemáticos usan otras etiquetas y no instancian una
  campaña confirmatoria. Las pruebas matemáticas del módulo no llaman al
  simulador.
- Stream bootstrap descriptivo: 953900. Es un stream de remuestreo, no una
  semilla de entrenamiento ni una observación científica adicional.

El pareamiento se conserva dentro de cada semilla mediante tapes exógenos
fijos. Cuando la política cambia, la trayectoria endógena puede cambiar; no se
afirma igualdad de estados físicos después de intervenir sobre decisiones.
Los cuatro controladores adquieren sus propias trayectorias. Compartir tapes
no significa compartir acciones, buffers factuales o resultados.

## 2. Endpoints y prevención de compensaciones por mortalidad

Cada evaluación nativa dura **320 pasos**. Las recompensas tienen escala
analítica [0,1] definida en `ENVIRONMENTS_DESIGN.md`; nunca se normalizan con el
mínimo, máximo, percentiles o resultados de las semillas finales.

Para una semilla, dominio y variante se conservan por separado:

1. `reward_mean`: suma de recompensa observada dividida por 320.
2. `alive_fraction`: suma del indicador vivo **postransición** dividida por 320.

La muerte es absorbente: los pasos muertos tienen recompensa cero y vida cero,
y el episodio continúa hasta el horizonte fijo. No se divide el retorno por
pasos supervivientes, no se truncan las colas muertas y no se rellenan trazas
incompletas como si fueran muertes observadas. `episode_endpoints` rechaza
trazas parciales, resurrecciones o recompensas en pasos muertos.

Un éxito por condición exige **simultáneamente**:

\[
 R_{full}-R_{comparador}\geq\delta_R,\qquad
 A_{full}\geq0.80,\qquad
 A_{full}-A_{comparador}\geq-0.005.
\]

El retorno no puede compensar una violación de cualquiera de las dos guardas
de vida. Los márgenes y el piso son decisiones prácticas del diseño, no
constantes psicológicas. No equivalen a ausencia de daño individual ni a
supervivencia terminal. Se publicarán también terminalidad, restricciones y
distribuciones completas como diagnósticos separados.

Se comparan valores sin redondear. Los márgenes son inclusivos; no se añaden
tolerancias después de ver resultados para convertir pérdidas en éxitos.

## 3. Familia confirmatoria fija: seis hipótesis

Una semilla gana una hipótesis únicamente si cumple el criterio completo en
**todas** sus condiciones. No se promedian dominios o comparadores para
compensar una condición fallida.

| ID | Contraste contra `full` | Dominios obligatorios | Margen de retorno |
|---|---|---|---:|
| `REL` | `noCross`: lesión aguda de acoplamientos entre pares | `ecology_train` y `ecology_delay9` | .02 |
| `GATE` | Tanto `constant_gate` como `permuted_gate` | Ambos dominios ecológicos para cada control: cuatro condiciones | .01 |
| `TENSION` | `scalar`: controlador disperso con tensión escalar, adquirido nuevamente bajo su propia política | Ambos dominios ecológicos | .01 |
| `CONTENT` | `permuted_content`: contenidos tipados permutados antes de entregarse a los dos consumidores | Ambos dominios ecológicos | .01 |
| `GENERIC` | `dense`: rival denso adquirido bajo su propia política | Ambos dominios ecológicos | .01 |
| `TRANSFER` | `dense`: el mismo rival propio, sin aprendizaje en las nuevas familias | `inventory_transfer` y `thermal_transfer` | .01 |

Todas las filas añaden el piso absoluto .80 y tolerancia relativa .005 de
vida. El criterio de `GATE` no puede cumplirse superando únicamente al control
más débil. Las permutaciones deben estar definidas y versionadas antes de los
finales, sin seleccionar realizaciones que produzcan el efecto deseado.

`CONTENT` prueba la contribución de la correspondencia de los contenidos
entregados a dos consumidores dentro de esta arquitectura. Las lesiones
individuales de consumidores, memoria y revisión de metas son **secundarias**;
no se convierten en parte de la hipótesis primaria después de ver resultados.

## 4. Presupuesto, adquisición propia y evaluación congelada

Se entrenan por separado cuatro controladores: `full` disperso y vectorial,
`dense` vectorial, `scalar` disperso, y `fixed` vectorial con soporte fijo.
Cada uno tiene la misma asignación de features y coeficientes, número de
episodios, iteraciones de ajuste y candidatos de selección. Cada uno recibe
sus propias acciones y experiencias efectivamente observadas.

Presupuesto por controlador y semilla:

- **8 episodios de entrenamiento × 320 pasos.** Ajuste del predictor cada dos
  episodios, 80 iteraciones por acción. Tres cabezas Q —recurso, energía y
  tarea— con acción sucesora común calculada con su agregado Q, alpha=.1 y
  gamma=.95. Ese bootstrap Q no incorpora el readout predictivo completo del
  actor nativo; se declara la diferencia algorítmica frente a realizaciones
  anteriores. El último paso del horizonte no añade bootstrap al target Q.
  Entrenamiento usa gate provisional .5 y epsilon=.2.
- **2 episodios propios de calibración × 320 pasos.** Q y predictor ya están
  congelados. El episodio 200 usa margen provisional .1 y produce el margen
  residual empírico .90. El episodio 201 usa ese margen calibrado y aporta
  exclusivamente los datos para ajustar los coeficientes del gate contextual
  según beneficio predictivo. Ambos usan gate provisional .5 y epsilon=.2.
  Son conjuntos propios separados; no se describe el ajuste como conjunto
  sobre ambas calibraciones ni se afirma política idéntica entre ellas.
- **7 episodios únicos de selección × 320 pasos**, uno por candidato, con el mismo
  tape exógeno de validación y clones del mismo checkpoint previo. Se congelan
  Q, predictor, coeficientes del gate y márgenes; sólo cambia el modo candidato.
  Los candidatos son contextuales con umbrales 0, .001 y .01, apagado y
  constantes .25, .5 y 1. El candidato contextual final se elige entre cuatro
  opciones: apagado y los tres contextuales. `constant_gate` también se elige
  entre cuatro: apagado, .25, .5 y 1. Se comparte la evaluación del apagado;
  no se ejecutan ocho episodios ni se da mayor búsqueda a una de las familias.

Este presupuesto **8+2+7** se aplica a los cuatro controladores: 5 440 pasos
por controlador antes de la evaluación, 21 760 por semilla entre los cuatro.

La selección maximiza recompensa entre candidatos con vida media al menos
.80. Si ninguno cumple el piso, se prioriza vida y después recompensa.
Desempates y orden de candidatos se fijan en la configuración serializada,
sin reglas dependientes del resultado final. El mismo presupuesto y reglas
se aplican a los rivales. El orden contextual es apagado, 0, .001, .01; el
constante es apagado, .25, .5, 1. La igualdad de los objetivos favorece la
primera opción de ese orden, como implementa `select_modes`.

La incertidumbre utilizada para tensiones y contexto del gate es causal y
variable desde el entrenamiento: media de los tres márgenes sensoriales más
RMSE de los últimos dieciséis errores predictivos ya observados. La ventana
se inicializa a cero al comenzar cada episodio. Cada error compara la
observación actual con una predicción emitida en la decisión anterior; no
utiliza el resultado todavía desconocido de la acción actual. La calibración
200 modifica el término global de margen; el contexto que recoge 201 conserva
esa variación observada, en lugar de usar una incertidumbre constante.

La puerta se aplica tanto a la primera pasada con tensión anterior como a
la pasada final del predictor. `noCross` y la puerta apagada ponen a cero la
contribución entre pares en ambas pasadas. Las predicciones por acción,
incluidas las predicciones de polos que se compararán en el paso siguiente,
se emiten antes de `environment.step`; la transición sólo retiene la
predicción previamente emitida para la acción realmente ejecutada.

El selector offline utiliza recompensa factual y la fracción de vida medida
por el evaluador para imponer el piso. Este uso de supervisión de viabilidad
en selección se declara explícitamente. Los predictores y la política paso a
paso reciben sensores públicos y recompensa; no reciben las colas latentes,
estado físico sin ruido, fases exógenas o futuros del tape. No se confunden
ambas fronteras de información ni se afirma aprendizaje sin esa supervisión
de selección.

Durante cada episodio de selección, la memoria del workspace puede actualizarse
con observaciones pasadas propias, pero la memoria del candidato validado no
se transfiere al checkpoint final seleccionado. Se conserva el estado base
producido por entrenamiento y calibración. Las siete validaciones no son siete
episodios adicionales de entrenamiento encadenados.

En evaluación se congelan todos los parámetros aprendidos y Q, con epsilon=0.
Se permiten las actualizaciones episódicas especificadas de memoria y metas
a partir del pasado observado; no ajustes de pesos con resultados de test.
Las intervenciones usan clones emparejados y el mismo aparato de actualización
episódica, excepto cuando precisamente se lesiona ese aparato.

La diferencia entre una lesión aguda (`noCross`, permutaciones) y un rival
reentrenado (`scalar`, `dense`, `fixed`) se conserva en todas las conclusiones.
Una lesión no mide el mejor agente que podría entrenarse sin el componente.

Igualar slots, coeficientes o iteraciones **no demuestra igual capacidad
efectiva**. El escalar suma los cuatro componentes de tensión de cada par y
coloca esa suma en el primer slot; los otros tres se fijan a cero. No duplica
la suma en cuatro columnas, evitando cambiar la contracción ridge mediante
columnas repetidas. Conserva la anchura de 48 features y asignación de 784
coeficientes/interceptos, pero pierde información y puede tener menor rango
efectivo. La dispersidad y el soporte también cambian la complejidad efectiva.
Se informa la asignación y el rango observado; no se atribuye un beneficio
exclusivamente a una semántica filosófica.

## 5. Transferencia y límites del dominio

La adquisición y la selección ocurren sólo en `ecology_train`.
`ecology_delay9` cambia el retraso ecológico. `inventory_transfer` y
`thermal_transfer` son familias de dinámica diferentes definidas antes de los
finales. No se usan resultados de esas familias para elegir pesos, gates,
hiperparámetros, candidatos, márgenes o la arquitectura ganadora.

Los tres pilotos de controladores evalúan exclusivamente `ecology_train` y
`ecology_delay9`. `inventory_transfer` y `thermal_transfer` permanecen
reservados para la evaluación final posterior al freeze. Sus fixtures de
física no ejecutan controladores entrenados y no se usan para elegir políticas.

La afirmación de transferencia es cero-shot respecto del aprendizaje de
parámetros, no ausencia de estado episódico. Actualizar memoria con el pasado
propio dentro del episodio es parte del algoritmo congelado y debe exponerse
también en el rival. Se conservan todos los resultados de los dos dominios
nuevos, aunque uno perjudique la conjunción.

Son entornos sintéticos internos distintos. Su diversidad no constituye una
replicación externa ni demuestra transferencia universal. Los tests de física
y observabilidad son contratos de ingeniería, no ensayos de utilidad del
controlador en los dominios reservados.

## 6. Inferencia, multiplicidad y potencia prospectiva

Para cada hipótesis se obtiene un vector de 80 booleanos en el orden fijo de
las semillas. Se contrasta unilateralmente
H0: Pr(una semilla cumple el criterio completo) ≤ .5 mediante la cola binomial
exacta. Los **seis** p-valores se ajustan conjuntamente por Holm con error
familiar .05. No se elimina una hipótesis porque su gate esté apagado, su
lesión sea nula, no alcance el margen o resulte adversa.

Con n=80, el umbral conservador de la primera etapa de Holm requiere **52
éxitos**. La cola nula para 51 es .00915805, superior a .05/6; para 52 es
.00484142. La potencia en ese umbral es:

| Probabilidad verdadera de satisfacer toda la conjunción por semilla | Potencia conservadora |
|---:|---:|
| .60 | 21.311% |
| .65 | 55.120% |
| .70 | **86.331%** |
| .75 | 98.337% |
| .80 | 99.946% |

La probabilidad de esta tabla corresponde al éxito **conjunto** de la
hipótesis, incluidos los dos dominios, las guardas y los dos controles de
`GATE`; no a ganar aisladamente un endpoint o dominio. Holm puede rechazar con
un umbral menos conservador según los demás resultados. No se promete 80% de
potencia para efectos menores o probabilidades conjuntas desconocidas.

Los tres pilotos permiten diagnosticar implementación y presupuesto; no
estiman con precisión la potencia real ni justifican reducir márgenes. Toda
modificación de diseño antes de los finales se registra y exige nueva fuente
congelada. No se aumenta n después de observar la significación.

Se publican diferencias por semilla, dominio y control, medias e intervalos
bootstrap pareados de 20 000 remuestreos al 95%, mediante
`paired_mean_interval`. Estos intervalos son descriptivos, marginales y sin
ajuste de multiplicidad; no sustituyen a Holm si éste falla. Una mayoría de
éxitos tampoco garantiza media favorable: pérdidas grandes deben permanecer
visibles, junto con las distribuciones de mortalidad y retorno.

## 7. Completitud, fallos y secundarios

Un outcome científicamente válido de colapso conserva su cero. Un archivo
ausente, no finito o incompleto impide cerrar la campaña; no se transforma
automáticamente en fracaso observado ni se excluye del denominador. Una
recuperación determinista conserva los originales y no aporta una semilla
nueva. Los hashes, CRC y reconstrucción independiente preceden al cierre.

Los análisis de `fixed`, `block_viability`, `block_resources`, `noMemory`,
`frozenMemory`, `noGoalRevision`, `observation_shock` y `dense_shock`, aristas,
efectos de pulso y diagnósticos de predicción son secundarios. Se rotulan
como tales y no amplían ni reemplazan la familia confirmatoria de seis.
Un test de contenido, memoria o metas no valida por sí mismo conciencia,
autobiografía, autonomía normativa o el significado psicológico de sus nombres.

Antes de finales: fuente y configuración completas, semántica exacta de cada
permutación y lesión, candidatos/desempates, seeds y budgets, tests de
observabilidad, propios buffers y cobertura de las condiciones. Después:
80 semillas completas, integridad de datos, reconstrucción de los seis vectores,
replays prospectivos y publicación de todos los resultados adversos o nulos.

## 8. API ejecutable

`statistics.py` implementa este contrato sin invocar entornos ni escribir
resultados. `compile_from_metrics` recibe
`data[seed][domain][variant][metric]`, con métricas `reward_mean` y
`alive_fraction`. Acepta claves de semilla enteras o strings decimales y
rechaza duplicados, datos ausentes, rangos inválidos y cambios de n/familia.
Los alias de nombres de dominio, variante o métrica sólo adaptan etiquetas
del runner; no alteran comparadores, márgenes ni condiciones.

`test_statistics.py` comprueba la inferencia contra una implementación de
referencia, los márgenes inclusivos, no compensación por mortalidad, conjunción
de dominios/controles, presupuestos n y familia, ausencia de descarte de datos
y bootstrap por semillas. Sus fixtures son números sintéticos, no experimentos.

## Referencias primarias de métodos

- [SciPy: binomtest](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html),
  alternativa unilateral y contraste binomial exacto.
- [R: p.adjust](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html),
  procedimiento Holm y control familiar; referencia al artículo de Holm (1979).

Estas referencias sustentan los procedimientos generales. No validan los
márgenes, las guardas, la arquitectura o los constructos elegidos para R9.
