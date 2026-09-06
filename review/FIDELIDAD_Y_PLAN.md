# Dictamen de fidelidad y plan priorizado

## 1. Contrato científico y referencias internas

No hay arquitectura completa validada ni evidencia de ventaja polar específica. Tampoco hay solamente nombres: el prototipo verifica interfaces y algunas funciones operacionales delimitadas; otras funciones carecen de utilidad suficientemente acreditada. La hipótesis arquitectónica todavía exige una especificación y un contraste más precisos.

Distinguir siempre las afirmaciones (i) mecanismo implementado, (ii) contribución causal a una función y (iii) relevancia para conciencia bajo una teoría independiente. Para registrar el estado operativo, usar cuatro campos, no un sello binario: **L1 implementación; L2 función causal; L3 utilidad práctica; L4 generalización/teoría**. L4 no equivale automáticamente a conciencia; esa inferencia conserva su protocolo independiente.

Las referencias a Ecs. 1–2 originales corresponden a los bloques `eq:architecture-original` y `eq:architecture-latent` de `sections/theory_architecture.tex`, no a números de ecuación que cambian al compilar. Su organización es:

A_i(t+1)=alpha_i(E_t) f(b_i+P_i+sum_j[W_ij+M_ij(C_t)]A_j+beta_i g_i(c_t,E_t)U_i+kappa_i C_i+delta_i Delta_i),

U_(t+1)=Lambda U_t+L A_t+R s_t; Delta_i=eta_i[U_i-theta_i]_+.

El caso actual con dos canales independientes refina la representación, pero RLS+MPC NO ejecuta literalmente estas dos ecuaciones. `K_eff=J^T Q` y `W_eff=-J^T Q J` son sensibilidades derivadas de un predictor y un objetivo: no se deben identificar retroactivamente con W, K, M(C) o U originales.

Se preservan las decisiones reportadas: Study 1 exploratorio y su trade-off del estimador; Study 2 `suspend_exclusive_polar_advantage_claim`; Study 3 `no_confirmed_network_advantage`, incluida la dependencia del panel activo respecto a desarrollo/generadores; P1 con 2/4 contrastes prácticos; P2 con funciones específicas y costo de revisión; estado del prototipo `engineering_verified_with_partial_empirical_support`. Nada en este dictamen recalcula o mejora esas conclusiones.

## 2. Tabla de fidelidad

En la columna de nivel, un resultado de una versión no se transfiere automáticamente a otra. Realiza significa realizar una función especificada, no validar el alcance completo de la capa.

| Componente original / anclas | Realización examinada | Estado L1 / L2 / L3 / L4 | Relación con el mecanismo original | Lesión o control ya existente | Pendiente y claim que perdería respaldo |
|---|---|---|---|---|---|
| Motor de tensiones: acceso a polos; G01–G03 | Canales independientes, orientación e intensidad; luego estados p, v, u | L1 verificado; L2 separación y perturbaciones; L3 no acredita necesidad universal de p; L4 abierto | Aproxima la organización; p/v/u introduce una hipótesis de continuidad adicional respecto al escalar A | Recodificación, intercambio de polos y ablación de continuidad | Comparar continuidad sin cambiar estimador, caché o memoria. Sin mejora pertinente, p queda como estado auditable, no como capacidad necesaria |
| Red W y modulación M(C); G02–G05 | K tau configurado; estimación acoplada y MPC en otra realización | Red: L1/L2 propagación; L3 Study 3 no confirma ventaja. Predictor: L1 verificado; L3 Study 2 útil vs lesión, P1 no pasa su umbral. L4 abierto | K tau aproxima una vía; predictor+MPC sustituye la regla regulatoria e introduce una hipótesis de ingeniería distinta | K/W off, topologías alternativas, lesión de efectos cruzados en planning only | No fusionar resultados entre mecanismos. Si el prior alternativo gana, no atribuir especificidad a la organización nominada |
| Ct: contenido y M(C); G07 | P1 transmite metas/prioridades a planner, memoria y reporter con cortes de ruta | L1 sí; L2 transmisión mínima intervenible; L3 no acredita todavía la función exigente de selección entre especialistas; L4 abierto | Aproxima la disponibilidad de contenido; allocator de recursos es una función diferente | Cortes selectivos de consumidores; no_broadcast | Consumidores con procesamiento propio, tarea que requiera usos distintos, sham de igual cómputo/banda y bypass directo. Sin efecto específico, no acreditar integración sensible al contenido |
| Ut: trazas, gU, Delta y reconsolidación; G08 | Memoria por cue, compuertas, borrado y versiones | L1 sí; L2 intervenciones; L3 utilidad acotada de recuerdo respaldada; L4 interpretación como Ut completo abierta | Aproxima retención/liberación; sustituye la recurrencia latente por registros discretos | Memoria off, borrar/editar registros, compuertas con estado de caché declarado | Separar pérdida del registro, bloqueo de recuperación y revocación de copias. Sin efecto bajo controles adecuados, no sostener necesidad del submecanismo particular |
| Et: regulación/overrides; G09 | Permisos, precedencia, obligaciones de demostración, razones y halt/review | L1 y casos funcionales delimitados; L3 no acredita moralidad; L4 abierto | Aproxima regulación institucional y sustituye multiplicadores por restricciones de decisión | Aprobación/prohibición y obligaciones incompatibles | Registrar incumplimientos, no llamar satisfechas obligaciones imposibles. Un caso que falle invalida esa regla de compliance, no toda ética posible |
| Motivación y prioridades; G06/G10 | Metas suministradas, compromisos persistentes, prioridades efectivas | L1/L2 sí; L3 prioridades respaldadas; L4 motivación endógena abierta | Realiza seguimiento de objetivos suministrados; aproxima la capa motivacional | Prioridades off; cambio de objetivo | Conservar G06 como referencia lograda. No inferir valores propios ni deseo de un peso suministrado |
| Planificación; G10 | MPC restringido de horizonte finito | L1 sí; L2 anticipación en fixture; L3 P1 no pasa criterio temporal; L4 abierto | Realiza planificación numérica; sustituye otras realizaciones sugeridas, sin superioridad por nombre | Myopic y forecast distinto | Previsión útil e inútil, retardos y cómputo comparable. Si falla, no sostener beneficio del horizonte ensayado |
| Automodelo: predicción; G05/G11 | Ganancias Study 1; efectos por pares Study 2; Bhat acoplada P1 | L1 sí; L2/L3 dependientes de versión; no equivalencia universal | Realizaciones diferentes de predicción; Bhat física no es taxonomía polar | Freeze/reset, lesión de uso de efectos cruzados | Admisión de datos débiles/ruidosos. Fallo retira utilidad del estimador/política ensayados, no capacidad general de aprender |
| Automodelo: procedencia; G11 | Token actor/acción/secuencia en ciclo; inferencia de candidatos en assay P2 | L1 ambos; L2 discriminación condicional en assay; L3 integración al aprendizaje primario pendiente; L4 abierto | Realiza coherencia de metadatos; inferencia física dentro del ciclo es ajuste nuevo | Rechazo de token inconsistente; candidatos distinguibles/idénticos | Un token coherente no basta para rechazar efectos foráneos mal etiquetados. Evaluar contaminación física manteniendo metadatos plausibles |
| Automodelo: fiabilidad; G11 | Heurística del estimador antiguo y monitor empírico P2, distintos | L1 sí; P2 calibración delimitada; transferencia a otras políticas no acreditada | Aproxima fiabilidad operacional; no debe equipararse a incertidumbre residual ni autoconciencia | Calibración fuera de desarrollo, referencia de tasa base | Brier, curvas, discriminación y cobertura por política. Si falla, no llamar calibrado a ese pronóstico |
| Automodelo: respuesta; G11 | Review P2 con costo externo medido | L1/L2 la política cambia conducta; L3 desempeño adverso; L4 abierto | Hipótesis nueva de regulación ante fiabilidad, no evidencia de seguridad | Review on/off | Mantener desactivada por defecto; siguiente contraste compara decisiones con costos explícitos. Forecast correcto no salva una política perjudicial |
| IACL; G12 | Control local y ofertas escalares en P2 | L1/L2 sí; L3 coordinación limitada útil en su tarea; L4 autonomía/cooperación amplia abierta | Realiza coordinación limitada y aproxima IACL | Reparto igualitario vs ofertas | Contenido, objetivos e información heterogéneos, fallos de comunicación. No negar el control local existente ni llamarlo negociación amplia |
| Percepción/significado; G04/G13 | Canales numéricos, gramática y signos fill/drain | L1 y significado operacional delimitado; L4 categorías originales no justificadas | Realiza un adaptador operacional; sustituye categorías psicológicas como objeto medido | Gramática, signos, permutación de canales/decoder | Justificar taxonomía y prior mediante criterios independientes. Bhat no resuelve el significado de las relaciones |
| Sistema completo; G14 | Integración parcial y trazas de un ciclo | L1 y funciones L2; L3 respaldo parcial; L4 no establecido | Aproxima la arquitectura conjunta; no realización literal completa de Ecs. 1–2 | Regresión, replay, intervenciones por componente | Garantías y transferencia del sistema adaptativo. Clip/proyección no bastan; ausencia de prueba no es prueba de imposibilidad |

Fuentes internas examinadas: `integrated_polar/agent.py`, `layers.py`, `numerics.py`, `network_tension.py`, `docs/P0P2_CIERRE_GAPS.md`, `sections/theory_architecture.tex` y `sections/p0p2_limits.tex`, fijadas a las bases indicadas en README.

## 3. Una hipótesis arquitectónica, no una definición retrospectiva

**H_R propuesta:** En transferencia con cambios de acoplamiento y competencia por recursos, seleccionar desde datos relaciones entre canales con efectos condicionalmente opuestos sobre una misma variable reducirá la pérdida externa, por al menos el margen práctico registrado, frente a priors alternativos de igual información, número de parámetros, presupuesto de ajuste y expresividad combinatoria, conservando disponibilidad independiente de ambos canales.

Esta hipótesis es NUEVA respecto al pairing nominado y no rehabilita Studies 2–3. No presupone que las ocho parejas filosóficas sean una taxonomía apropiada ni que un signo opuesto constituya conflicto. La competencia/conflicto se define con restricciones y efectos conjuntos verificables. Sus resultados no resolverían el significado psicológico de las relaciones.

Operacionalización a fijar antes del ensayo arquitectónico: estimar Bhat con las mismas transiciones para todos; puntuar oposición condicional por similitud negativa normalizada de columnas de efectos sobre variables comunes; seleccionar conexiones bajo un mismo grado y límite de aristas. La selección no usa nombres. Las alternativas usan el mismo Bhat, historial, recursos y dimensiones, con soporte permutado de igual grado o agrupación por recursos. Ninguna recibe menos información. Misma cantidad de pesos libres, regularización y tuning; la localización de relaciones es el factor experimental. Aprendizaje de pesos, selección de soporte e interpretación semántica se registran como tres operaciones diferentes.

Comparadores mínimos: (1) recodificación completa signed-plus-intensity, que debe conservar conducta; (2) controlador genérico de capacidad/información comparable, que no incorpora la selección relacional propuesta; (3) prior estructural alternativo y shuffled del mismo grado; (4) lesión de planning que no utiliza la relación pero conserva Bhat e información accesible. Incluir denso o feature no lineal si la afirmación intenta abarcar superioridad organizativa más amplia; informar su distinta capacidad cuando exista. Una fórmula genérica exactamente equivalente es control de implementación, no una prueba de superioridad.

Contraste que descarta H_R dentro del alcance: no superar el margen registrado frente a cualquiera de los comparadores primarios pertinentes, o deteriorar las guardas. Si la permutación de coordenadas cambia la conducta física más allá de tolerancia, la prueba es inválida por implementación, no evidencia a favor o contra H_R. Si el soporte alternativo/shuffled o denso vuelve a ganar, se mantiene suspendida la afirmación de ventaja polar exclusiva y no se redefine «polar» para incluir al ganador.

**El próximo experimento de pasos 2–3 no prueba H_R.** Prueba admisión pertinente de evidencia al aprendizaje, requisito previo para interpretar más tarde una comparación de organización. Un éxito del gate no será etiquetado como éxito de polaridad.

## 4. Orden de trabajo y condiciones de salida

1. Precisar fidelidad e H_R (G02/G03/G04/G13/G14). Salida: una transformación explícita entre términos originales y componentes computacionales, con claims retirables.
2. Desagregar cinco factores causales (G02/G03/G05). Salida: efecto atribuible a un cambio, no a un paquete de signos+conflicto+estimador+solver.
3. Integrar procedencia y admisión del aprendizaje; separar valor/confianza/recuperación (G04/G05/G08/G11). Salida: un contraste prospectivo operacional, no ventaja polar.
4. Probar necesidad de continuidad, selección de contenido, recuerdo y anticipación (G01/G07/G08/G10); conservar G06 como referencia funcional. Salida: tareas y controles capaces de refutar utilidad, con mecanismos que puedan fallar.
5. Ampliar comparadores/dominios (G09/G12/G13/G14). RL modular vs monolítico entra aquí cuando sea pertinente al claim, con canales, memoria, observaciones, interacciones de entrenamiento y presupuesto comparables. No se afirma haberlo ejecutado.

## 5. Desagregación causal mínima

### Núcleo: diseño 2^5 sobre snapshots y transiciones comunes

| Factor | Único cambio | Se mantiene idéntico | Métrica e interpretación |
|---|---|---|---|
| S: signo | e frente a abs(e) | Coactivación, topología, Bhat y regla de decisión | Efecto externo del signo del mensaje, no de quitar simultáneamente conflicto |
| C: coactivación | Activar/desactivar el término chi*p+*p- o su derivada declarada | e y signo, routing, Bhat y solver | Utilidad del término con chi especificado por contexto; sin asumir que coactivar sea malo |
| R: simetría receptora | Dos filas receptoras distintas frente a su promedio por pareja | Mensaje y Bhat idénticos; mismo número de productos computados, salida redundante al sink | Utilidad de diferenciar recepción; la ablación reduce uso expresivo, por lo que no sustituye al comparador estructural igual-capacidad |
| E: estimación | Estimador diagonal frente a acoplado sobre exactamente las mismas transiciones | Familia de regla de decisión, pérdidas, constraints y número de iteraciones | Efecto del modelo estimado; no cambiar también a inversión directa |
| D: regla de decisión | Inversa regularizada frente a optimización restringida | Mismo snapshot de Bhat, mismo target, constraints finales y recursos | Efecto de decidir distinto con el mismo conocimiento |

Primero ramificar desde estados clonados sin aprendizaje posterior para estimar efectos locales; después un ensayo de políticas cerradas, separado, permite divergencia de estados y aprendizaje. No interpretar ambos como la misma estimación causal. Todas las combinaciones y las interacciones se retienen. Señales cero, soporte diagonal y condiciones en que la relación propuesta es falsa son controles, no casos a descartar por rendimiento desfavorable.

### Memoria: tres operaciones independientes

Usar factores registro presente/borrado, recuperación habilitada/bloqueada y copia activa presente/revocada. El contraste de borrado compara ambos brazos con el MISMO estado de caché. Revocar caché no borra el registro; cerrar la recuperación no revoca copias ya entregadas. Editar memoria produce una versión nueva; actualizar una copia activa requiere un evento explícito separado. Los brazos combinados del factorial no autorizan atribuir su diferencia conjunta a un solo factor.

### Contenido Ct

Ya existe transmisión mínima; endurecerla con dos consumidores de procesamiento propio y salidas diferentes, por ejemplo planner de acción y selector de medición diagnóstica con presupuesto fijo. Contrastar varias metas/residuos/contextos candidatos y mantener al menos dos rutas. Cortar planner deja memoria y reporter intactos; cortar el otro consumidor debe afectar su uso específico. El consumidor lesionado ejecuta el mismo cómputo sobre un paquete sham del mismo tamaño. No se conserva idéntico contenido útil en la ruta que se interviene: se conservan información externa disponible, timing, trabajo computado y ancho de banda. Añadir bypass directo sin selección con información equivalente para distinguir integración de mera disponibilidad de la meta.

Retirar cue externo, retirar target externo, cerrar retrieval y revocar copia activa son ensayos diferentes. Para el test sin cue, cargar el contenido antes, retirar la clave explícitamente y comprobar un cambio de acción frente a una copia editada o revocada; no llamarlo recuperación espontánea si simplemente sigue activo el caché.

### Automodelo

Freeze de Bhat no resetea p ni memoria. Reset de Bhat no vacía caché. Lesión de cross-effects en planning no apaga el estimador: solo cambia qué términos utiliza. Introducir feedback foráneo con token correcto para probar la atribución física, además del rechazo de metadatos inconsistentes ya existente. Incluir candidatos indistinguibles y ninguna fuente explicativa; no exigir adivinar una procedencia no identificable.

### Invariancia

Intercambiar polos implica transformar estados, targets, permisos, costos, memoria, caché, Bhat, routing y decoder. D'=D*P^{-1}, Bhat'=Bhat*P^{-1}; W'=PWP^{-1}; K'=PK*T_tau^{-1} si las features también cambian. No comparar una permutación de p con decoder fijo como si fuese recodificación. Registro de max-discrepancia de acciones/efectos con ruido físico común.

## 6. Etiquetas, figura y clasificación de exigencias

Mantener el título histórico y añadir el subtítulo propuesto: «Propuesta arquitectónica y evidencia delimitada de control interpretable; conciencia bajo protocolo independiente». No renombrar el mecanismo RLS+MPC como principio descubierto. La sección de conciencia conserva su contrato independiente por etiqueta estable; no confiar en que siempre sea el número 9.

Figura 1 y Tabla 1: mantener todos los componentes originales, pero asignar estado a FUNCIONES y FLECHAS, no pintar toda una capa como demostrada. Tres tintas y códigos redundantes: P propuesto (gris), I implementación verificada (azul), F función delimitada con evidencia (verde, siempre con versión/tarea). El estado de utilidad práctica va en una columna separada (+, no acreditada, adversa), de forma que una política review pueda ser I y tener efecto causal observado sin recibir un sello de mejora. Nunca una flecha sólida convierte la arquitectura completa en ejecutada; no incluir por defecto una categoría de conciencia demostrada.

Eliminar del lenguaje de éxito HGI/INC/tau, el carácter sagrado del anillo y cualquier regla de consenso hacia un punto medio. Degradar las ocho etiquetas a categorías propuestas cuya taxonomía y utilidad estructural no se justificaron empíricamente en este programa. Review es una política con costo. Una expresión nombrada sin camino a una variable usada se marca como telemetría o se elimina; no atribuirle función causal.

Clasificar por separado:
- Exigido para corresponder al mapa original: contenido/selectividad Ct, trazas/gates Ut, regulación Et, motivación e IACL con los alcances que el original declara. No afirmar realizados sus alcances completos.
- Necesario para el prototipo actual: contratos de datos, órdenes de update, separación de fuentes, contabilidad de costos, estados inválidos y pruebas de intervención. En un deployment externo, requisitos de autenticación/safety dependerán del análisis de amenazas y riesgos.
- Ampliación opcional: VAE como compresor, un LLM, criptografía cuando no hay amenaza de suplantación, negociación rica, nuevos sensores y demostraciones globales adicionales. Un VAE no es condición de toda memoria.
- Fuera de la secuencia de ingeniería: inferencia de experiencia fenoménica, ASI o ley física de polaridad. Sus teorías y medidas no son intercambiables con un bug de la interfaz de feedback.

## 7. Riesgos de autoengaño y regla de revisión

Vigilar redefinir polaridad para absorber shuffled/dense ganadores; confundir acceso al dato con necesidad de un workspace; confundir residual con probabilidad; atribuir éxito del estimador de estado a continuidad p; confundir señal disponible y utilizada; usar confianza para reducir un valor correcto sin una hipótesis de política; seleccionar subgrupos que favorecen un grafo; fusionar coeficientes aprendidos con significado semántico; restaurar el mapa completo y tratarlo como sistema ejecutado; confundir no superar umbral con efecto nulo; evaluar solo rechazo de contaminación y no falso rechazo de feedback propio tras cambios reales; declarar un baseline «igual» por parámetros sin contabilizar memoria, observaciones, entrenamiento y cómputo.

Cada revisión debe preguntar: ¿qué dato refutaría la afirmación y en cuál nivel? Un éxito de L1 no se promueve por retórica a L3; un fracaso de L3 no borra L1/L2 válidos ni refuta L4 en todo dominio.

## Conclusión

El problema principal no es únicamente la ausencia de capas, sino la falta de correspondencia suficientemente precisa entre la hipótesis polar original, sus realizaciones computacionales y las afirmaciones respaldadas por los experimentos. Los estudios descritos no acreditan una ventaja específica del emparejamiento polar nominado ni de la propagación inter-polar configurada. El prototipo integrado sí demuestra algunas funciones operacionales delimitadas, pero no establece todavía la necesidad de todos sus componentes, su generalización o el alcance completo de las capas propuestas. La equivalencia con formulaciones genéricas impide atribuir diferenciación a las coordenadas por sí solas; no excluye investigar una contribución basada en organización, aprendizaje o integración, siempre que se formule de manera falsable y se contraste con alternativas pertinentes. La siguiente etapa debe precisar esa contribución, aislar sus mecanismos y conservar separadas la evidencia de control, la evidencia arquitectónica y las hipótesis sobre conciencia.

Referencias metodológicas primarias: Guo et al. (2017), On Calibration of Modern Neural Networks, PMLR 70:1321–1330; Nosek et al. (2018), The preregistration revolution, doi:10.1073/pnas.1708274114; Dulberg et al. (2022), Modularity benefits reinforcement learning agents with competing homeostatic drives, arXiv:2204.06608. Precedentes, no comparaciones ejecutadas por este dictamen.
