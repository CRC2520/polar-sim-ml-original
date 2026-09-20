# P1.E.1 — consecuencia ecológica aprendida y asignación temporal de crédito

Extensión experimental nueva del motor tabular `collective/bridge_v2/engine.py`.
El motor y los resultados anteriores se mantienen intactos. Este experimento no
implementa la arquitectura completa de ocho polaridades ni demuestra conciencia.

## Hipótesis y significado de costo

El diseñador elige valorar positivamente recursos futuros y fija λ=3. El modelo
aprende la magnitud y signo de la consecuencia de sus acciones. No aprende por sí
solo que conservar recursos sea deseable. No se interpreta como valencia emergente.
El costo de una acción se define como la diferencia entre el recurso futuro medio
predicho con acción 0 y con esa acción, horizonte H=12. Se conserva su signo; no se
impone por construcción que consumir tenga costo positivo.

El estimando mecanístico es el efecto de la acción focal sobre el recurso medio de
los siguientes 12 pasos con las acciones posteriores del grupo fijadas a 1, salvo
agentes muertos. Incluye dinámica ecológica y muerte; excluye compensación mediante
una política adaptativa de los otros agentes. La utilidad en interacción nativa se
evalúa aparte y no se infiere de la precisión de este estimando.

## Adquisición factual y control temporal

Por semilla: 128 estados iniciales independientes (`anchors`), 32 ensayos con
reinicio controlado por estado, una acción focal aleatoria por ensayo, 8 asignaciones
de cada acción en orden aleatorio. La observación inicial de recurso y vitalidad
propia es exactamente la misma entre los 32 reinicios de un anchor. Las fases
ecológicas y estados iniciales son iguales dentro de él; el ruido público futuro
es independiente por ensayo. El estado verdadero, fase y acciones de otros agentes
nunca son variables del predictor. El entrenamiento ve 4096 trayectorias factuales
con una sola acción focal realizada por trayectoria; los cuatro resultados
contrafactuales pareados se reservan a evaluación experimental.

La intervención `shifted_credit` atribuye al ensayo i el resultado del ensayo i−1
del mismo anchor, con cierre circular. Mantiene exactamente las mismas X, el mismo
multiconjunto de Y por contexto inicial, todos los estados/transiciones físicos,
ruido, acciones y presupuesto de aprendizaje. Sólo cambia la llave de asignación
acción→consecuencia. Es un desfase de crédito entre ensayos reiniciados, no una
modificación de la demora física de regeneración ni una prueba de memoria temporal
continua. Las acciones son balanceadas sin reemplazo; por ello el control desplazado
puede producir una pequeña anticorrelación (−1/31 en expectativa), no independencia
perfecta. No se asume que deba estimar costo exactamente cero.

Cada semilla tiene anchors independientes de entrenamiento, test ID y test OOD.
Los streams 301, 401 y 501 separan estos conjuntos. Se retienen todos los pasos de
una trayectoria en un solo conjunto. El OOD cambia crecimiento medio/amplitud a
0.10/0.04 y rango inicial de recursos a 0.10–0.55; los controles dentro de cada
dominio comparten exactamente ese mismo entorno. No se atribuye al crédito temporal
ninguna comparación entre entornos ID y OOD.

## Modelos y controles

Todos los brazos con head usan 4096 ensayos, igual ruido, las mismas variables
públicas, H=12, ridge=0.01 y 20 coeficientes. El predictor ecológico es ridge con
acción categórica × [1,r,r²,v,rv]. Es un algoritmo convencional de ajuste de
consecuencias; darle el nombre POLAR no prueba exclusividad.

* `aligned`: asignación correcta de la consecuencia; costo aprendido activo.
* `no_head`: mismo modelo adquirido y mismo cómputo, gate cero en decisión.
* `shifted_credit`: misma capacidad, datos y cómputo con asignación temporal errónea.
* `generic`: ridge polinomial de grado total ≤3 en (r,v,a), 20 coeficientes; igual
  información, cantidad de parámetros, muestras y algoritmo, distinta base.
* `immediate`: misma arquitectura de 20 coeficientes y muestras, entrenada con
  recurso público del paso 1. Diagnostica qué aporta horizonte mayor.
* `no_action`: predictor diagnóstico sin acción, 5 coeficientes; no es control de
  capacidad. Su costo marginal es cero. Se registra error absoluto y causal.
* Oráculos: diferencia causal real a H=12 y pérdida física inmediata a H=1,
  exclusivamente para evaluación. Nunca alimentan decisiones o entrenamiento.

Se verifica además equivalencia algebraica con una política convencional que suma
el recurso futuro predicho al score y resta un baseline común por estado. Es una
comprobación de consistencia, separada del comparador polinomial flexible.

## Evaluación nativa e identificación

Motor original con `restraint_cost=0` y `initial_restraint_prior=0`: desaparece la
penalización ecológica hardcoded de los rewards, también el prior heredado. El
controlador tabular base se entrena durante 10 episodios de 300 pasos. Sus Q y
visitas se clonan idénticas en todos los brazos y quedan congeladas al evaluar.
Cada brazo ejecuta 2 episodios ID y 2 OOD de 300 pasos con tapes comunes. El score
es qt+qv+qg−λ·costo aprendido. Los roles funcionales de qt/qv/qg se conservan; no hay
evolución o transmisión entre grupos en este bloque.

La comparación aligned/no_head identifica contribución incremental del uso del
predictor bajo esta política. Aligned/shifted identifica la asignación de crédito
de estos ensayos, manteniendo ecología y presupuesto. El comparador generic
examina dependencia de parametrización con el mismo número de coeficientes; no
garantiza igual condición numérica o exactamente igual complejidad funcional.

Métrica primaria de mecanismo: MSE de efecto causal (acciones 1..3) por seed,
frente a cero costo y crédito desplazado. Métrica primaria de utilidad: fracción
de tiempo-agente vivo. Secundarias: recurso medio, acción media y puntuación
0.5·supervivencia+0.5·recurso (también elección del diseñador). Se reporta siempre
ID/OOD por separado y diferencias de las 30 semillas con bootstrap descriptivo de
10000 remuestreos; no se exige éxito para completar el experimento ni se declara
una hipótesis confirmada con una sola mejora descriptiva. Contrastes adicionales
son exploratorios sin corrección por multiplicidad.

## Reproducción y presupuesto

Piloto: 930101–930103. Final: 933001–933030, sin modificación de parámetros tras
ver resultados finales. Por seed: 4096 ensayos de adquisición, 1024 trayectorias
contrafactuales de diagnóstico (512 por dominio), 10 episodios de entrenamiento
tabular y 20 episodios nativos de evaluación. Se conservan observaciones, acciones,
estados verdaderos diagnosticados, tapes, modelos, Q congeladas, resultados por
seed, hashes, condiciones y replay completo de al menos un episodio por seed.

Ejecución desde este directorio:

```
python ecology.py --phase pilot --output RUTA_NUEVA
python -m unittest -v tests_ecology
python ecology.py --phase final --output RUTA_NUEVA
python ecology.py --verify-only --output RUTA_EXISTENTE
```

Los archivos con sufijo `DIAGNOSTIC` son evaluación/registro y quedan excluidos de
`public_training_data` mediante una lista explícita de variables permitidas. El
replay reconstruye datos físicos con semillas y compara transiciones bit a bit.
