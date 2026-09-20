# R8 — afirmaciones operacionales y confirmación prospectiva

Estado: criterios cerrados antes de las semillas finales; sujetos al freeze común
de fuentes, diseños y artefactos de desarrollo.
Responsable: revisión independiente. Este documento fija qué permitirían concluir
los resultados; no certifica que las hipótesis hayan sido apoyadas.

## 1. Alcance y unidad de evidencia

Las ocho correcciones del diseño no se convierten artificialmente en ocho tests.
La familia confirmatoria contiene **seis** hipótesis, identificadas abajo. Los
contratos de implementación, la identificación de umbrales y los análisis
descriptivos responden preguntas diferentes.

La unidad independiente es la **semilla completa**, con todos sus agentes,
grupos, pasos, episodios, condiciones y ajustes internos. Ninguno de estos
elementos se cuenta como réplica independiente. Cada contraste utiliza las mismas
semillas y tapes exógenos en sus condiciones; una trayectoria que diverge porque
cambia la política conserva el pareamiento exógeno, no una física endógena idéntica.

Los modelos poblacionales se calibran con semillas de desarrollo y sus Q se
actualizan dentro de cada realización final. El agente integrado debe declarar
por separado sus datos de adquisición, selección y evaluación dentro de cada
semilla. Compartir experiencias públicas entre individuos de una semilla es
aprendizaje agrupado explícito, no aprendizaje aislado de cada individuo. No se
permite ajustar hiperparámetros con resultados de evaluación final ni agrupar
semillas finales para elegir una configuración ganadora.

P1/P2 conservan su condición de evidencia previa, incluidos los resultados
adversos de supervivencia, los comparadores genéricos superiores y los umbrales
no identificados. R8 prueba una realización modificada; no vuelve favorables
retrospectivamente aquellos resultados.

## 2. Familia confirmatoria fija

Sea Δ el endpoint del primer brazo menos el del comparador. Cada desigualdad se
evalúa con valores sin redondear y el margen es inclusivo. Un win exige satisfacer
**todas** las condiciones de la fila en esa semilla.

| ID | Contraste | Criterio práctico por semilla | Conjunción |
|---|---|---|---|
| `Q1` | `normalized_common` − `legacy_full` | Δ fracción de tiempo vivo ≥ 0.01 | Ambas reglas de transmisión, después de promediar los tres regímenes |
| `Q2` | `state_common` − `global_common` | Δ recurso ≥ 0.01 y Δ tiempo vivo ≥ −0.005 | Ambas reglas, con la misma media de tres regímenes |
| `E_credit` | Crédito correcto frente a acción temporalmente desplazada | B − C ≥ max(0.10 B, 0.0001), donde B es MSE desalineado y C alineado | Cada uno de ID, OOD con retraso 2 y OOD con retraso 9 |
| `E_utility` | Integrado completo − `noEcology` | Δ recurso ≥ 0.02 y Δ tiempo vivo ≥ −0.01 | Cada uno de los tres dominios |
| `T_utility` | Integrado completo − `noNetwork` | Δ recurso ≥ 0.01 y Δ tiempo vivo ≥ −0.005 | Cada uno de los tres dominios |
| `I_generic` | Integrado completo − `generic_joint` | Δ tiempo vivo ≥ 0.01 y Δ recurso ≥ −0.02 | Cada uno de los tres dominios |

Los márgenes son decisiones prácticas del protocolo, no umbrales derivados de
una teoría psicológica. `Q2`, `E_utility` y `T_utility` estudian conservación bajo
una tolerancia explícita de viabilidad; su éxito no se redactará como mejora
demostrada de supervivencia. `Q1` e `I_generic` sí exigen una mejora de tiempo vivo.
La tolerancia de tiempo vivo es relativa al comparador; no impone una viabilidad
absoluta mínima. Puede cumplirse incluso si ambos brazos funcionan mal. Sólo el
análisis separado de p* utiliza umbrales absolutos de viabilidad.
La salvaguarda energética resta un percentil del error absoluto de entrenamiento
a la predicción inmediata. Es una heurística aprendida con objetivos impuestos;
no un intervalo de predicción calibrado ni una garantía probabilística de
viabilidad, especialmente al transferir a otro retraso físico.

Para población, `survival_time_fraction` promedia la fracción de pasos vivos
postransición en las últimas cinco generaciones: 400 pasos, seis grupos y veinte
agentes por grupo en la configuración principal. El recurso se divide por la
capacidad y se promedia en el mismo tramo. Los tres regímenes predefinidos son ID,
metabolismo 0.4 y regeneración media 0.12/amplitud 0.05. Se promedian con pesos
iguales **dentro** de cada regla `success` y `conformity`; después se exige la
conjunción de ambas. El alcance primario es esa media de regímenes, no una mejora
en cada régimen por separado. Supervivencia terminal y probabilidad de viabilidad
son endpoints distintos y no sustituyen tiempo vivo.

Para el integrado se evalúa un episodio de 400 pasos por dominio: tiempo vivo
postransición promediado sobre 16 agentes y recurso verdadero postransición
dividido por capacidad 160, promediado sobre dos grupos. El recurso verdadero se
usa sólo como endpoint del evaluador; entrenamiento y selección utilizan recurso
público observado. Los dominios nunca se promedian entre sí para satisfacer una conjunción. En
`E_credit`, los seis outputs son recurso público normalizado y energía propia
normalizada a horizontes 1, 4 y 12, con peso igual por output y horizonte. Ambos
predictores se puntúan sobre los mismos probes uniformes de la trayectoria
reservada del agente completo, por dominio. La adquisición y el episodio de
diagnóstico predictivo usan probes uniformes con epsilon 0.15; la selección y
utilidad nativa usan epsilon 0 y conservan la elección Gumbel entre acciones
factibles. Son episodios separados: selección 500, predicción 650 y utilidad 700.
El
suelo absoluto 0.0001 evita declarar mejora práctica porque un denominador casi
nulo produzca un porcentaje grande.
La media de seis outputs puede mejorar por energía inmediata aunque no mejore
el recurso retrasado. Se conserva y muestra la matriz por horizonte/variable;
`E_credit` por sí solo no identifica aprendizaje específico del retardo. Los
targets factuales recogen consecuencias bajo la continuación de la política,
incluidas sus acciones posteriores, no efectos físicos directos manteniendo
esas acciones futuras constantes.

## 3. Test, multiplicidad y precisión

Para cada ID se conserva un vector de treinta booleanos en el orden fijo de las
semillas. Se contrasta unilateralmente H0: Pr(una semilla cumple el criterio
completo) ≤ 0.5, mediante la cola binomial exacta con n = 30 y p = 0.5. Los seis
p-valores se ajustan juntos por Holm, con error familiar 0.05. No se retiran de la
familia las hipótesis desfavorables, los gates apagados ni los dominios difíciles.
La implementación común es `r8_completion/confirmation.py`.

Un empate de efecto no satisface un margen positivo. La igualdad exacta con el
margen sí satisface la desigualdad inclusiva. Una simulación válida que muere o
agota el recurso conserva su outcome, incluso si es cero. Un dato ausente o no
finito no es un éxito, y además impide declarar completa una campaña hasta
resolver su integridad. No se descartan ni reemplazan selectivamente semillas.
Una repetición determinista de recuperación documentada no aporta una réplica
nueva.

El test trata la **probabilidad de satisfacer un criterio práctico**, no la media
del efecto ni la probabilidad de conciencia. Una victoria estadística no asegura
que la media mejore: pocas pérdidas grandes podrían dominarla. Por ello se
publican los treinta contrastes, sus medias e intervalos bootstrap pareados por
semilla al 95%, además de los dominios por separado. Esos intervalos son
descriptivos, marginales y sin ajuste de multiplicidad; no se usan como tests
primarios alternativos cuando falla Holm. La ausencia de soporte no prueba
equivalencia ni ausencia del mecanismo.

Con seis hipótesis, el umbral conservador de la primera etapa de Holm coincide
con Bonferroni: al menos 22 wins de 30. La potencia exacta en ese umbral es:

| Probabilidad verdadera de cumplir todo el criterio | Potencia conservadora |
|---|---:|
| 0.60 | 9.40% |
| 0.70 | 43.15% |
| 0.80 | 87.13% |
| 0.90 | 99.80% |

Holm puede ser menos conservador según los demás resultados. Treinta semillas
no garantizan potencia del 80% para cualquier efecto. Los pilotos de desarrollo
se describen aparte, con su n pequeño y sin incorporarlos al resultado final;
estimaciones piloto no se transforman en garantías de potencia.

## 4. Contratos causales y comparadores

**Arbitraje.** RMS, pesos globales y pesos por estado se aprenden sólo con
experiencia factual y estados observables. Los retornos usan vida propia y
recurso público observado, no crecimiento, fase o stock latente. El comparador
global recibe los mismos datos y presupuesto de selección; barajar una tabla de
pesos es una ablación contextual, no el único rival. Deben seguir visibles
`shared_successor` y `qv_policy_off`, ganadores o comparadores relevantes previos.
El arbitrador por estado almacena 20 × 3 pesos, frente a tres del global: `Q2`
evalúa el paquete de condicionamiento y capacidad adicional, no un efecto puro
de usar contexto con idéntica capacidad. El contraste descriptivo con la misma
tabla permutada conserva parámetros y aísla el uso de su asignación contextual,
sin convertir esa lesión posterior en un rival reentrenado.

La calibración poblacional v2 compara diez candidatos por familia: tres ajustes
NNLS con regularización 16/64/256 y siete constantes simplex prefijadas. Las
constantes también están disponibles como veinte filas iguales en la familia
por estado. Ambas seleccionan con los mismos episodios y recurso **observado**;
el stock verdadero queda para evaluación/auditoría. Puede ganar una constante:
«calibrado» incluye esta selección y no significa que todos los pesos procedan
de NNLS o que siempre exista adaptación contextual. V1 y sus pilotos, que
mostraron un global NNLS débil y usaron stock verdadero para selección, se
conservan como desarrollo expuesto. V2 reutiliza sus filas de entrenamiento,
fortalece simétricamente el espacio de candidatos y elimina aquella selección
privilegiada antes del freeze. Sólo su calibración se usa en los finales.

**Crédito temporal.** El control primario cambia únicamente la acción acreditada:
en las filas de probes uniformes de cada individuo/episodio, desplaza circularmente
las acciones cinco **eventos de probe**, conservando su multiconjunto exacto, el
estado y memoria preacción, el futuro observado real y la física. Cinco eventos
no son cinco pasos; el intervalo físico entre eventos varía y debe quedar
registrado. Se declara el cierre circular. La memoria preacción no incluye la
acción actual. Los datos factuales usados para calibrar efectos de acción proceden de probes uniformes;
no se producen etiquetas de acciones no realizadas. Desplazar todos los futuros
Y modifica también la relación estado→futuro y sólo puede ser un control
secundario de alineación temporal conjunta.

**Red.** W/K son coeficientes predictivos aprendidos en un soporte entre pares
prefijado. El gate global se selecciona en desarrollo entre 0, 0.25, 0.5 y 1,
con apagado permitido; no es activación autónoma dependiente del estado. Los
objetivos siguientes son observaciones reales, no targets generados por W/K.
Una intervención en el predictor identifica su efecto dentro del algoritmo;
no identifica aristas causales del entorno ni relaciones psicológicas reales.
Los grafos sintéticos de tests son contratos de ingeniería, no evidencia nativa.

**Rival integrado.** `generic_joint` conserva observaciones, historia, targets,
parámetros disponibles, readout, salvaguarda y presupuesto de selección. Su
predictor convencional `JointRidgePredictor` ajusta conjuntamente 24 coeficientes
por acción/output: 4 × 16 × 24 = 1536. El predictor POLAR ajusta local y residual
entre pares secuencialmente, con los mismos 1536 coeficientes y datos. Cambia la
regla de ajuste. El costo aritmético no es idéntico y se declara; igualar número
de coeficientes no prueba por sí solo equivalencia de complejidad efectiva.
Las experiencias y los Q compartidos se adquieren bajo la política completa;
el rival ajusta su predictor y readout sobre ese mismo dataset. Por tanto se
compara una regla predictiva con adquisición común, no dos agentes que aprendan
por separado bajo sus respectivas políticas. Esa asimetría limita cualquier
afirmación de superioridad general.

**Lesiones.** `noEcology`, `noNetwork` y `noMemory` intervienen sobre el mismo
agente aprendido; estiman contribución incremental bajo esa realización. No son
una comparación con la mejor política que podría reentrenarse sin el componente.
`noNetwork` pone a cero el gate entre pares y conserva el predictor local y su
readout: mide contribución del acoplamiento, no de toda predicción. `noMemory`
retira la ventana explícita de dieciséis observaciones; conserva Q, visitas y
la predicción anterior usada por tensión. No es un agente sin estado temporal.
Memoria útil exige una mejora medible bajo su intervención: persistir estados
o ejecutar una lesión no basta. La lesión de memoria es descriptiva y no añade
una séptima hipótesis confirmatoria.

## 5. Umbral p* y factorial social/físico

El piloto explora la grilla 0–1 y presiones predefinidas para buscar un intervalo
interior. La regla que selecciona presión y soporte final se congela antes de
las nuevas semillas. Un cambio de presión cambia el dominio de la pregunta y se
documenta; no repara retrospectivamente la falta de identificación en P1.

Se muestran proporciones de viabilidad y bandas binomiales simultáneas
Clopper–Pearson, ajustadas por Bonferroni dentro de cada perfil sobre todos sus
puntos. El criterio de viabilidad por semilla y el número de perfiles deben
constar en el protocolo específico. El helper adopta tiempo vivo ≥ 0.8 y recurso
≥ 0.2, ambos inclusivos. Un bracket certificado requiere un punto
inferior cuyo límite superior sea menor que 0.5 y otro superior cuyo límite
inferior sea mayor que 0.5. No se obtiene un punto por extrapolación. La
reversión del indicador de probabilidad ≥ 0.5, el soporte insuficiente, un cruce en el borde y la
incertidumbre que impide bracketing se reportan explícitamente. La compatibilidad
con un cruce único no demuestra monotonía de la función: se permiten descensos
que no vuelvan a cruzar 0.5. Un cruce observado único en la rejilla tampoco
demuestra unicidad entre los puntos no observados. Una interpolación, si se muestra,
es descriptiva/modelodependiente. No se calcula Δp* como estimación identificada
cuando algún componente no lo es. Estas bandas son por perfil; no se les atribuye
cobertura simultánea sobre toda la campaña.

`status=within_support` describe sólo un cruce muestral interpolable.
`reliable_identification` exige además bracket simultáneo ordenado y al menos
80% de réplicas bootstrap con cruce interior. Ese 80% es una regla diseñada de
estabilidad, no una probabilidad posterior ni garantía de cobertura. El intervalo
bootstrap condicionado a réplicas con cruce se rotula condicional y no como
intervalo de confianza incondicional al 95% para p*.

El factorial puntaje de copia × redistribución física conserva ambos factores
separables y contrasta sus cuatro combinaciones. El factorial adicional con
donantes fijados desde baseline mantiene controlada la reproducción. Bajo
donantes fijos, igualar puntajes debe ser un control negativo de esa vía; el
efecto del pooling es condicional a ese régimen de donantes. No se llama efecto
indirecto natural ni descomposición completa de mediación. Efectos, interacciones
y contrastes se reportan estimativamente, con incertidumbre, fuera de la familia
confirmatoria de seis.

## 6. Brecha 8: identidad, diferenciación y conciencia

| Nivel | Evidencia que puede obtener R8 | Límite de la afirmación |
|---|---|---|
| Estado individual persistente | Q, visitas, memoria y consecuencias propias vinculadas al mismo individuo; trazas temporales y lesiones | Individualidad operacional de registros; no identidad personal ni un yo fenomenológico |
| Integración funcional | Los módulos operan en una misma política y sus intervenciones cambian decisiones o utilidad bajo física pareada | Causalidad dentro del sistema implementado; no integración consciente demostrada |
| Diferenciación algorítmica | Diferencia de ajuste secuencial frente a JointRidge y, si pasa `I_generic`, ventaja práctica robusta en los tres dominios | Ventaja local frente a ese rival y presupuesto; no superioridad universal ni exclusividad de POLAR |
| Representación de dos polos | Coactivación y dos grados de libertad; recodificación invertible diferencia/intensidad con decisiones equivalentes | La misma información admite coordenadas convencionales; nombres filosóficos no hacen irreducible la representación |
| Conciencia subjetiva | No hay endpoint directo en esta campaña | No se afirma experiencia, dolor sentido, deseo vivido, autenticidad, autoconciencia ni estatus moral |

El control de coordenadas transforma p+,p− en diferencia e intensidad y vuelve a
reconstruir los mismos canales antes de evaluar el predictor. Su equivalencia es
un contrato matemático esperado, no evidencia estadística contra un competidor.
No implica que reentrenar ridge en coordenadas distintas conserve el resultado:
la penalización y la selección pueden depender de la parametrización.

La historia explícita se reinicia en cada episodio; Q y visitas conservan el
aprendizaje entre episodios. Los reinicios físicos, incluida la reaparición de
individuos después de una muerte dentro del episodio anterior, son reglas del
banco de pruebas. Esa continuidad de índices/parametrización no demuestra
continuidad biográfica o subjetiva.

Los ocho nombres filosóficos son etiquetas del mapeo operacional actual de
variables computadas; no se atribuyen como lista literal al manuscrito de 2025.
El reporte conserva junto a ellos sus definiciones observables. Un score de
depleción no es miedo y un error predictivo no es sufrimiento. Los objetivos de
recurso y energía, el término de coactivación y la salvaguarda de viabilidad son
decisiones del diseñador. Aprender consecuencias o conexiones no equivale a
inventar objetivos, comprender normas o desarrollar fenomenología.

Si se discuten teorías de conciencia, se distingue una analogía funcional de un
indicador validado y éste de una conclusión fenomenológica. El informe de
Butlin y colaboradores de 2023 propone propiedades computacionales derivadas de
teorías; no autoriza convertir los endpoints ecológicos de R8 en un detector de
conciencia ni extrapolar su evaluación histórica a todo sistema de 2026.

## 7. Condiciones de preparación y cierre

Antes de ejecutar finales: nombres/API y agregaciones definitivas, lista de
semillas y separación desarrollo/evaluación, criterios de viabilidad y bracket,
presupuesto de entrenamiento/selección de cada rival, orden temporal sin acceso
al futuro, control action-only, parity de física y guardas, pruebas de
equivalencia de coordenadas, contratos de IO y protocolo/hash/commit congelados.

Para cerrar ejecución: treinta semillas completas por bloque, sin exclusiones
selectivas; hashes y CRC válidos; reconstrucción independiente de endpoints y
wins desde resultados por semilla; replay preseleccionado; todos los seis tests,
dominios, comparadores relevantes y resultados adversos publicados. Una auditoría
de recuperación conserva originales y separa igualdad byte a byte de equivalencia
científica reconstruida. Completar estas tareas cierra la ejecución y la
trazabilidad. Cada hipótesis conserva después su resultado: apoyada bajo su
criterio, no apoyada o no evaluable por un fallo concreto; ninguna se declara
exitosa simplemente porque corrió el código.

## Referencias metodológicas primarias

- Agarwal et al., [Deep Reinforcement Learning at the Edge of the Statistical
  Precipice](https://arxiv.org/abs/2108.13264): importancia de incertidumbre y
  agregación con pocas realizaciones; no prescribe los márgenes particulares de R8.
- [SciPy `binomtest`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html):
  referencia de la prueba binomial y alternativas unilaterales.
- [R `p.adjust`](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html):
  implementación/documentación de Holm y control del error familiar, con referencia
  al trabajo original de Holm (1979).
- Butlin et al., [Consciousness in Artificial Intelligence: Insights from the
  Science of Consciousness](https://arxiv.org/abs/2308.08708), 2023: marco de
  propiedades indicadoras basadas en teorías, usado aquí para delimitar el alcance.
