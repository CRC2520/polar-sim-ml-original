# R8.I.1 — agente integrado: diseño prospectivo

Nueva realización experimental del agente; conserva las fuentes y resultados
P1/P2/P5/P6. Los nombres filosóficos son etiquetas de operaciones del simulador,
no constructos psicológicos validados ni evidencia de experiencia subjetiva.

## Física y experiencia propia

Se conserva la física de BridgeEngine: grupos de ocho agentes, dos grupos,
recurso compartido, extracción proporcional al recurso, vitalidad individual,
regeneración logística con historia y fase, muerte absorbente dentro del episodio.
Cada episodio contiene 400 transiciones consecutivas. Los parámetros ID conservan
la configuración Bridge; OOD cambia únicamente el retraso de regeneración a dos o
nueve pasos. Los controles de cada dominio usan exactamente el mismo entorno y
tapes exógenos. Reiniciar un episodio no produce anchors clonados ni etiquetas
contrafactuales para entrenamiento: cada agente observa sólo la acción que realizó.

La política es la misma realización algorítmica para todos los individuos, con
Q/visitas y memoria propia por individuo; los estimadores se comparten mediante un
pool explícito de sus experiencias públicas. No acceden a recurso verdadero,
crecimiento, fase, acciones no observadas o futuros durante la decisión.

## Componentes de un solo agente

1. Qt/Qv/Qg tabulares, mismos rewards individuales/colectivos del motor, anulando
   la penalización ecológica hardcoded y su prior. Los tres targets TD utilizan
   **la misma acción sucesora** elegida por la política integrada.
2. Arbitraje Q de `r8_completion.population`: escalas RMS adquiridas en train y
   pesos condicionados por el estado público (vitalidad×recurso). Ajuste sólo
   mediante acciones de exploración uniformes y retornos factuales posteriores.
3. Memoria de las últimas 16 observaciones propias de recurso/vitalidad/acción.
   Produce niveles retrasados, pendientes y variabilidad; no recibe estados ocultos.
4. Predictor ridge de consecuencias observadas (recurso público y vitalidad propia)
   a horizontes 1, 4 y 12. Sus etiquetas provienen de la misma trayectoria y no
   atraviesan límites de episodios. Una lista explícita de variables permitidas
   separa datos públicos de registros diagnósticos privados.
5. Ocho pares de activaciones no negativas, 16 canales con coactivación permitida;
   su mapeo operacional se explicita debajo.
6. Red `LearnedNetwork` ajustada sólo con (polos, tensión, acción, polos observados
   siguientes). Predice polos por acción. El score utiliza estas predicciones
   mediante un readout aprendido del retorno factual, por lo que cambiar el
   acoplamiento puede cambiar decisiones del mismo agente.
7. Salvaguarda de viabilidad que compara acciones con la predicción aprendida de
   energía. La aplica también noEcology: la ablation ecológica retira sólo uso de
   consecuencias del recurso. Evitar el resultado adverso de P1 es una hipótesis
   evaluada, no una propiedad asumida de esta salvaguarda.

El valor de conservar recursos y el umbral de viabilidad son elecciones del
diseñador. La magnitud/signo de consecuencias y pesos se aprenden; no se afirma
que el agente descubra autónomamente el objetivo moral o ecológico.

## Ocho pares, operaciones medibles

| Etiqueta operacional actual | Canal positivo operacional | Canal negativo operacional |
|---|---|---|
| Poder / Vulnerabilidad | Vitalidad actualmente disponible | Riesgo de agotamiento estimado desde tendencia propia |
| Placer / Dolor | Magnitud de ingreso/reward propio reciente | Magnitud de pérdida de vitalidad/reward reciente |
| Integración / Fragmentación | Disponibilidad observada del recurso compartido | Variabilidad temporal de ese recurso |
| Control / Rendición | Intensidad de la acción propia previa | Frecuencia de pausas propias recientes |
| Deseo / Límite | Déficit respecto al objetivo local de vitalidad | Escasez respecto al objetivo del recurso compartido |
| Libertad / Orden | Novedad de la visita al estado | Persistencia de acciones recientes |
| Preservación / Transformación | Retención de reserva energética durante la memoria | Cambio de la acción propia respecto a su historia |
| Reconocimiento / Autenticidad | Cumplimiento del objetivo externo de reserva común | Cumplimiento del objetivo local de energía |

Estos canales no se construyen imponiendo p−=1−p+. Una persona puede considerar
que los nombres filosóficos no describen estas operaciones; el experimento sólo
valida las operaciones y las conexiones que realmente implementa.
Esta lista procede del mapeo operacional actual; no se atribuye como lista literal
al manuscrito de 2025, que sólo explicita algunos nombres y usa N1…N8.

La tensión es el error observado entre los polos actuales y la predicción que se
hizo previamente para ellos, más 0.2·p+·p−. En el arranque se usa persistencia como
predictor. El agente nunca compara contra polos futuros aún no observados.

## Crédito temporal en trayectorias continuas

El control conserva exactamente las mismas trayectorias físicas, estados,
memoria preacción y etiquetas futuras de adquisición. Desplaza únicamente la
acción acreditada, cinco eventos de exploración uniforme muestreados atrás en la secuencia
de cada individuo dentro del episodio, con cierre circular. Mantiene exactamente
el multiconjunto de acciones aleatorizadas por individuo/episodio. El tiempo
físico entre estos eventos varía: no son cinco pasos del entorno, ni otra demora
de regeneración. Sólo estas acciones de exploración uniforme entrenan los
predictores y el arbitraje; cada una produjo una trayectoria factual propia.
La evaluación predictiva usa episodios separados y los dos predictores reciben
exactamente las mismas observaciones/acciones/etiquetas factuales de test.

## Entrenamiento, intervenciones y comparación

Los modelos y la red se ajustan entre cuatro bloques de tres episodios nativos, con Q
actualizándose durante esos episodios. Las elecciones de gate o regularización
utilizan únicamente segmentos train/valid separados; las semillas finales no
autorizarán consultar episodios de test durante selección. Gate cero está permitido.

Evaluación con todos los parámetros aprendidos congelados y condiciones pareadas:
full, noEcology, noMemory, noNetwork, crédito desplazado, coordenadas equivalentes
y comparador JointRidge con la misma información, 1536 parámetros de red, readout
propio de 17 coeficientes, guardas y presupuesto de selección. Redes fija y
aleatoria se incluyen como lesiones secundarias del mismo agente, con el gate
seleccionado para el modelo aprendido; no son competidores optimizados. Las
intervenciones sobre el mismo agente congelado estiman necesidad/uso causal de
componentes y se distinguirán de comparaciones entre políticas reentrenadas.
JointRidge comparte las experiencias y Q adquiridas por full: compara la regla
de ajuste de un predictor y su readout bajo una política de adquisición común,
no dos sistemas completos reentrenados con trayectorias propias diferentes.
`noNetwork` elimina el acoplamiento entre pares, conservando la predicción local
y su readout. Un gate seleccionado igual a cero implica ausencia de uso de W/K
en esa evaluación; no se atribuye a la red un efecto que se ha apagado.
El control de coordenadas retiene diferencia e intensidad de ambos canales y
debe conservar la decisión; no se usa como demostración de superioridad.
`generic_flat` es además una implementación vectorial equivalente, distinta del
rival JointRidge. `noMemory` retira la ventana de 16 observaciones; mantiene el
estado recurrente mínimo de la predicción previa usado para tensión y las Q.

La selección intra-semilla usa sólo el episodio ID 500, separado de train y test.
Se evalúan doce combinaciones λ ecológico∈{0,2,8} × gate de red∈{0,.25,.5,1} para
full y otras doce para JointRidge. Se elige máximo recurso público sujeto a
supervivencia ≥la opción (0,0)−.005; los empates prefieren menor λ y luego gate.
Siempre hay una opción factible por definición (0,0); esto no garantiza que
esa política sobreviva ni que el resultado se transfiera a test. El umbral
de energía es .12 y la guarda resta el percentil 90 del error absoluto de energía
a un paso medido sobre el propio ajuste train. Es una heurística con residuo
in-sample, no un intervalo predictivo calibrado o una garantía de viabilidad.
La regularización ridge es .1 para consecuencias,
.1 para red, 1 para readout y 64 para el arbitraje compartido.

## Inferencia y restricciones

Desarrollo: semillas 940101–940199. Final prospectivo: 942001–942030, prohibido
ejecutar antes del freeze y autorización de root. Se reportan todos los dominios,
incluidos resultados adversos. Los endpoints quedan fijados con el revisor antes
del freeze: crédito requiere B−C≥max(.10·B,1e−4), donde B es MSE de acción desplazada
y C es MSE alineado, media igual de los seis outputs R/capacidad y E/máximo a
horizontes 1/4/12, sobre probes propios de test. Utilidad ecológica requiere
recursos +.02 con supervivencia ≥noEcology−.01. La ventaja integrada frente a
JointRidge requiere supervivencia +.01 y recursos ≥comparador−.02. El efecto de red
requiere recursos +.01 y supervivencia ≥noNetwork−.005. Cada éxito por semilla
exige conjunción en ID, OOD demora2 y OOD demora9. Intervalos de medias
son descriptivos; tests de signos/binomial por semilla y Holm corresponden a los
claims confirmatorios acotados, no a cada métrica exploratoria.
Test usa un episodio independiente (700) por dominio y variante. La supervivencia
es tiempo-agente vivo postransición, promedio 400 pasos×16 individuos. El recurso
es stock postransición/capacidad, promedio 400 pasos×2 grupos; sólo la evaluación
accede al stock verdadero. La precisión usa un episodio disjunto (650), con los
mismos tres dominios, y sólo las filas de exploración uniforme aún vivas al origen.
La utilidad nativa y la selección (500/700) usan ε=0: todas las acciones de agentes
vivos respetan el conjunto de acciones factibles predichas. Adquisición y diagnóstico
predictivo (650) usan ε=.15 con probes uniformes no restringidos por la guarda;
se registran por separado probes y violaciones. Ninguna prueba usa valores futuros
verdaderos para decidir. Los errores del predictor pueden causar pérdida real de
viabilidad aun cuando se respete el conjunto factible predicho.

Todos los datos fuente/resultados previos permanecen intactos. Los nuevos IO deben
ser atómicos y validar el CRC de cada contenedor. Se conservan observaciones,
acciones, recompensas antes/después, estados diagnósticos, targets, parámetros y
hashes suficientes para reconstrucción y replay independiente.

## Revisión de desarrollo, sin selección posterior de endpoints

Se ejecutaron 940151–940153 en `results/r8_integrated_pilot_v2`, 103.34 s en total
con OPENBLAS_NUM_THREADS=1. Se preserva la fuente exacta usada en ese piloto. Los
tres replays y las equivalencias de acción por coordenadas fueron exactos. Se
retuvieron los resultados nulos y adversos sin cambiar los márgenes prospectivos.
Los gates full elegidos fueron (λ,g)=(2,0),(0,.25),(8,.5); por tanto el primer seed
no usa acoplamiento W/K y el segundo no usa preferencia ecológica explícita.
Las mejoras predictivas medias frente a crédito erróneo fueron pequeñas o
negativas en estos tres seeds; no se interpreta el piloto como evidencia positiva.

Las comprobaciones mecanísticas cubren paridad física con Bridge durante 14 pasos
en los tres retrasos, Q con sucesor común, targets propios a sus horizontes, ausencia
de señales latentes en train, recuentos de acción preservados tras subsampling,
cronología de tensión, coactivación posible de los ocho pares, uso efectivo de la
red en scores, guarda nativa, congelación y checkpoint/replay sin pickle.
La versión final sólo añade metadatos/checks de publicación y un identificador
estable de versión respecto de ese piloto; no cambia dinámica o selección.

Los resultados finales quedan sujetos a freeze y autorización explícita de root.
Cada ejecución final verifica la fuente congelada antes y después; cada semilla
compara sus fuentes al inicio/final y sólo publica COMPLETE.json tras volver a
validar los hashes originales y CRC de sus artefactos.
