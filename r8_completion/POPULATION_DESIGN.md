# R8: regulación Q, identificación de p* y rutas físicas/sociales

Documento prospectivo nuevo. P1/P2 y su motor se conservan sin modificar. El
motor de herencia es `collective/bridge_v2/engine.py`, SHA256
`48330c1a88a2312182bb488dae0096e2ccf674c6fe2b0ca7ddf6fa9b8a29dccc`.
La evidencia P1 justificó estos cambios, pero no se combina con las muestras R8.

## Presupuesto, semillas y límites de adaptación

Toda corrida poblacional mantiene 20 generaciones, 400 pasos/generación, seis
grupos de 20 agentes y ventana de evaluación de las últimas cinco generaciones.
Las 30 semillas finales serán 941001–941030, pareadas por todas las condiciones
de este módulo. Se ejecutarán únicamente después de publicar el freeze conjunto
de fuente y calibraciones y de la autorización del agente principal. El CLI
verifica ese freeze, la lista exacta de semillas y los artefactos de calibración.

Desarrollo reservado: entrenamiento de arbitraje 940001–940008; validación
940011–940014; diagnóstico posterior sin retuning 940015–940020; búsqueda de
soporte de p* 940031–940035. Las pruebas de contrato usan 940081–940091.
El ajuste en validación nunca usa semillas finales. Las escalas y tablas de
arbitraje se congelan como `CALIBRATION_FROZEN.json`; la presión y grilla de
p* se congelan como `THRESHOLD_SELECTION.json`. No se cambia ninguno para
rescatar resultados finales. Los regímenes externos Q no se usan para ajustar
el arbitraje ni seleccionar su regularización.

Máximo final: Q48 celdas, p*22 celdas y A30 celdas, cada una con 30 semillas.
Son hasta 3000 corridas poblacionales; las evaluaciones reproducidas para
auditoría no se cuentan como nuevas observaciones independientes.

## R1: separar escala, sucesor y arbitraje

Se cruzan dos factores: sumar Q en sus escalas originales o normalizar cada
cabeza, y usar máximos independientes o la misma acción sucesora para todos
los bootstraps. No se cambia gamma. Se añaden un arbitraje global aprendido,
uno por estado, la misma tabla reasignada a otros estados y el control ganador
P1 que elimina Qv de la política conservando su aprendizaje.

| Brazo | Actor | Sucesor |
|---|---|---|
| `legacy_full` | Qt+Qv+Qg original | Máximo de cada cabeza |
| `shared_successor` | Qt+Qv+Qg original; control genérico P1 | Acción común |
| `normalized_independent` | Ventajas Q normalizadas, pesos iguales | Máximo de cada cabeza |
| `normalized_common` | Ventajas Q normalizadas, pesos iguales | Acción común |
| `global_common` | Pesos globales aprendidos | Acción común |
| `state_common` | Tabla aprendida de 20 estados × 3 pesos | Acción común |
| `shuffled_common` | Misma tabla con asignación de estados permutada | Acción común |
| `qv_policy_off` | Qt+Qg; Qv continúa aprendiendo | Máximo de cada cabeza |

Todos conservan las tres tablas Q de 20×4 por agente, los mismos pasos de
aprendizaje y las mismas cintas aleatorias. La tabla estatal tiene 60 entradas
frente a tres pesos globales: no se afirma igualdad de grados de libertad
entre estos dos modelos. El control permutado sí conserva exactamente las
60 entradas, datos, ajuste y coste de consulta, y elimina su correspondencia
contextual. El comparador global recibe el mismo conjunto factual y el mismo
presupuesto de diez candidatos de calibración; el genérico P1 y Qv-off
permanecen visibles para evitar una comparación exclusivamente interna favorable.

La normalización usa ventajas `Q−media_de_acciones(Q)` y su RMS por cabeza,
estimado únicamente en el entrenamiento. Un desplazamiento común de valores
por acción no decide la política. Todos los pesos son no negativos y suman
tres; se controla la suma de pesos y se mantiene la temperatura original .05.
La normalización sí cambia la dispersión de logits respecto del legado; no
se afirma igualdad de esa dispersión entre brazos.

### Aprendizaje factual del arbitraje

Un controlador de desarrollo usa exploración uniforme con epsilon=.05. Se
registran solamente decisiones exploratorias de agentes vivos: Q y estado se
capturan **antes** de la acción; el target es `.5×media_alive_propio +
.5×media_recurso_públicamente_observado`, ambos en los siguientes 40 pasos
efectivamente recorridos. Se usa la observación ruidosa del recurso, recortada a
[0,1] y normalizada por capacidad; nunca el stock oculto usado para evaluar.
No hay etiquetas de acciones contrafactuales ni oráculos del entorno.

Se centran predictores y target por estado. Un ajuste ridge no negativo aprende
tres coeficientes globales. Cada estado ajusta sus coeficientes con contracción
hacia los globales. Estados con menos de 12 observaciones o sin los cuatro
valores de acción observados usan el fallback global. Los coeficientes se
normalizan a suma tres. Global y estatal eligen independientemente entre diez
candidatos: tres ajustes NNLS con lambda16,64,256 y siete constantes simplex
`[3,0,0]`, `[0,3,0]`, `[0,0,3]`, `[1,1,1]`, `[1.5,1.5,0]`, `[1.5,0,1.5]` y
`[0,1.5,1.5]`. En la familia estatal las constantes son tablas de veinte filas
iguales, por lo que puede seleccionarse legítimamente un fallback sin adaptación
contextual. Ambas familias consumen exactamente los mismos datos, diez
configuraciones y semillas de validación, bajo las dos reglas de transmisión.
El objetivo de selección es `.5×tiempo_vivo+.5×recurso_públicamente_observado`;
se conserva también el recurso verdadero como auditoría, pero no se usa para
elegir. Empates a doce decimales favorecen constantes y después el nombre de
candidato. No se seleccionan modelos por el conjunto final ni por regímenes
externos. La evaluación final sigue usando el stock verdadero como endpoint
externo, separado de las observaciones disponibles al controlador.

Se conserva una versión de desarrollo v1 que comparaba únicamente tres ajustes
NNLS y seleccionaba con el recurso verdadero. Reveló un global aprendido débil
([0,3,0]) y se declaró antes de cualquier semilla final. La revisión v2 amplía
simétricamente ambas familias con las siete constantes y usa recurso observado
para seleccionar; reutiliza exactamente las mismas filas factuales de
entrenamiento. V1 y sus pilotos quedan como desarrollo expuesto. Sólo el
artefacto de `population_development_v2` se incorpora al freeze final. Si
ambas familias eligen la misma constante, no se atribuye adaptación contextual.

API compartida con la arquitectura integrada:
`fit_arbitration(qheads[N,3,4], state_index[N], actions[N], returns[N],
is_uniform_probe=None, ridge=64.0)` y
`arbitrate_q(qheads[...,3,4], state_index, calibration, mode)`.
El estado es `bin_vitalidad*5+bin_recurso_observado`, igual al motor preservado.

### Regímenes y dos afirmaciones primarias

Los ocho brazos se ejecutan bajo transmisión `success` y `conformity`, en:
ID original; metabolismo .4; regeneración media .12/amplitud .05. La composición
inicial es .5. El diagnóstico posterior de desarrollo usa solamente ID; los
dos regímenes externos se evalúan por primera vez con las semillas finales.

La unidad inferencial es la semilla. Para cada regla se promedian los tres
regímenes; después se exige que el criterio se cumpla en ambas reglas:

- **Q1:** `normalized_common−legacy_full`, mejora del tiempo medio vivo≥.01.
- **Q2:** `state_common−global_common`, mejora del recurso medio≥.01 y pérdida
  de tiempo medio vivo no mayor a .005. Es utilidad ecológica condicionada a
  no inferioridad de supervivencia, no una afirmación de superioridad primaria
  en supervivencia.

Estos criterios describen robustez del **promedio** de tres regímenes, no de
cada régimen por separado. Se publican todos los resultados por régimen y los
otros controles. El tiempo vivo es la fracción de los 400 pasos en que el agente
permanece vivo después de la transición, promediada sobre agentes, grupos y
últimas cinco generaciones; no es supervivencia terminal ni probabilidad de
viabilidad poblacional.

El agente principal combina Q1/Q2 con cuatro afirmaciones integradas mediante
pruebas binomiales exactas unilaterales sobre P(semilla cumple)> .5 y Holm,
familia fija de seis. Las medias y bootstrap pareado de 2000 remuestreos se
presentan como descriptivos. Con n=30 y familia seis, la primera etapa
conservadora de Holm usa alpha=.05/6 y requiere 22 éxitos. Las potencias exactas
son .094011, .431518, .871349 y .997980 para probabilidades verdaderas de
satisfacer el margen de .60, .70, .80 y .90, respectivamente. Esto no garantiza
potencia para efectos menores: el diagnóstico de desarrollo ID no estima
directamente la potencia del criterio final de tres regímenes y no permite
bajar márgenes ni alterar el conjunto final.

## R5: localizar soporte antes de estimar un umbral

Piloto A/base bajo ambas transmisiones, p=0,.1,…,1 y metabolismo .3,.45,.6.
Cinco semillas emparejadas por todas las celdas. Una presión es elegible si
ambas reglas tienen viabilidad≤.2 en p0 y≥.8 en p1, con un único cruce observado
de probabilidad .5. Se elige la presión con media de p* más próxima a .5;
empates favorecen la menor presión. No se usa significancia ni un bootstrap
favorable para seleccionar el piloto.

La grilla final incluye 0,1 y siete puntos con separación .05 alrededor de
la media de umbrales piloto redondeada a .05, recortando fuera de [0,1]. Si los
umbrales de ambas reglas distan más de .3, se conserva la grilla completa .1.
Si no existe presión elegible, se fija metabolismo .3 y grilla0–1 solamente
como diagnóstico; no se fuerza la existencia de un umbral.

Viabilidad por semilla: media final de tiempo vivo≥.8 y recurso≥.2. La estimación
usa un único cruce bajo→alto del indicador de probabilidad empírica≥.5. Pequeñas
reducciones que no cruzan .5 no prueban multiplicidad de umbrales; una reversión
alto→bajo impide identificar uno único. No se impone isotonicidad ni se extrapola.

Se publican bandas exactas Clopper–Pearson marginales y simultáneas Bonferroni
sobre todos los puntos de cada perfil. Un bracket respaldado exige un punto
bajo con límite superior<.5 y uno alto con límite inferior>.5, en ese orden.
La identificación fiable exige además cruce dentro del soporte y al menos80%
de los bootstrap pareados con cruce único. Los draws inválidos se cuentan; los
IC de puntos interiores se rotulan condicionales. Toda conclusión se limita
a la presión y grilla seleccionadas, no al dominio original en general.

## R6: factorial de ingreso físico y puntaje de transmisión

Se implementa el 2×2 `copy_score_equal`×`resource_pool` bajo selección por éxito,
en p=.35,.50,.65. Son cuatro brazos con transmisión endógena operativa y cuatro
con mapas de donantes fijados desde el control sin tratamiento de la misma
semilla. Los mapas incluyen donante/receptor individuales, reemplazos de grupos
y mutaciones. El brazo fijo reutiliza identidades de donantes, no resultados
de una condición tratada. Se añaden controles base/score_equal bajo conformidad.

El factorial endógeno estima efectos totales, efectos simples e interacción.
Los mapas fijos controlan las decisiones de selección social: los efectos de
pool allí son efectos controlados respecto de esa intervención. No se presentan
como efectos directos/indirectos naturales de una mediación completa; contenidos
aprendidos transmitidos y otras respuestas pueden cambiar. Al fijar los mapas,
score_equal debe producir exactamente las mismas trayectorias físicas que el
control del mismo pool: es un control negativo, no evidencia del efecto social
operativo. Bajo conformidad, los puntajes también deben ser irrelevantes.

Se separan fitness bruto, puntaje de copia, ingreso físico, supervivencia y
recursos. Pool iguala ingreso por instante entre agentes vivos y conserva el
total; no garantiza igualdad del ingreso acumulado durante vidas de distinta
duración. Los contrastes A son estimaciones causales descriptivas dentro del
simulador; no añaden pruebas primarias a la familia fija de seis.

## Integridad, reproducibilidad y ejecución

Todo artefacto se serializa en memoria, se publica con archivo exclusivo, fsync y
enlace atómico, y se verifica con SHA256/CRC antes de `COMPLETE.json`. No se
sobrescriben destinos existentes. Cada celda conserva outcomes y transmisiones
de todas las generaciones, endpoints, estado final completo y mapas de donantes
cuando corresponden. Las trazas completas se reconstruyen determinísticamente
desde las cintas por semilla/generación del motor preservado.

Ejemplos desde la carpeta que contiene `r8_completion` y `collective`:

```bash
python -m r8_completion.population --part calibration --phase pilot --output r8_results/population_development --workers 4
python -m r8_completion.population --part calibration --phase pilot --output r8_results/population_development_v2 --training-source r8_results/population_development --workers 4
python -m r8_completion.population --part Q --phase pilot --output r8_results/Q_pilot_v2 --calibration r8_results/population_development_v2/CALIBRATION_FROZEN.json --workers 4
python -m r8_completion.population --part threshold --phase pilot --output r8_results/threshold_pilot --workers 4
python -m r8_completion.population --part A --phase pilot --output r8_results/A_pilot --workers 4
```

Los comandos finales usan `--phase final`, nuevos destinos y los artefactos
congelados. El programa rechaza una ejecución final sin freeze verificable o
con calibración/selección distintas de las congeladas. Los endpoints negativos,
inconclusos y adversos se conservan completos.
