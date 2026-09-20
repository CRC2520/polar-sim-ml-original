# R9 — contrato de diseño de los ocho ajustes

Estado: diseño de una **nueva realización limitada**, actualizado contra
agent.py, config.py, runner.py, NETWORK_DESIGN.md y WORKSPACE_DESIGN.md antes
del freeze R9. No contiene resultados ni certifica utilidad. No modifica las
fuentes, los criterios ni las conclusiones de R8; tampoco sustituye ahora el
manuscrito canónico. El contrato estadístico será **PREREG_R9.md**: este
documento no duplica sus semillas, márgenes, agregaciones o decisiones.

R9 estudia un agente con observación nativa de tres componentes
**[resource, energy, demand]** y cuatro acciones. Los nombres representan
variables y objetivos del banco, no afectos, conceptos morales ni conciencia.
No se implementa la arquitectura cognitiva general completa de 2025.

## 1. Correspondencia exacta de los ocho ajustes

| Ajuste | Contrato concreto de R9 | Comprobación e interpretación |
|---|---|---|
| 1. Niveles de afirmación | Separar ejecución, efecto causal en el algoritmo, utilidad externa, transferencia y afirmación sobre conciencia. | Un test de rutas o replay verifica ejecución; una lesión puede identificar uso; una mejora exige el endpoint y margen previamente declarados. Ningún nivel se convierte automáticamente en el siguiente. |
| 2. H_REL / H_CAT / H_TRANSFER | H_REL contrasta una organización computacional especificada. H_CAT requeriría un mapa catálogo–proceso independiente y fijado previamente. H_TRANSFER evalúa la realización congelada en otra familia con un mapa de observaciones/acciones declarado. | R9 puede contrastar utilidad relacional y transferencia acotadas. No descubre por ello el catálogo filosófico ni establece fenomenología. Las tres familias son construidas por el equipo: no son validación externa independiente. |
| 3. Soporte aprendido frente a pesos | Ajustar modelos predictivos por acción con regularización dispersa; registrar tanto los coeficientes como sus ceros/soporte. Comparar con soporte fijo y ajuste denso declarados. | Un cambio de peso dentro de soporte fijo no basta para afirmar descubrimiento estructural. El soporte obtenido es de asociación predictiva entre entradas y outputs definidos, no un grafo causal del entorno ni el regulador original de 2025. |
| 4. Tensión vectorial | Conservar cuatro componentes por par: mismatch respecto al pronóstico anterior, coactivación ponderada, oposición predicha de acciones y escala empírica de error. | Los cuatro están disponibles antes de actuar. La coactivación no identifica competencia; la oposición predicha no identifica incompatibilidad causal; la escala de error no garantiza cobertura probabilística. El rival escalar resume los cuatro en un slot y rellena los otros tres con cero. |
| 5. Compuerta contextual con apagado explícito | Un indicador binario abre la ruta cuando el beneficio predictivo estimado supera el umbral seleccionado. El mismo gate actúa en ambas pasadas; existe modo off exacto. | Comparar con cuatro candidatos constantes, incluido off compartido, y con contexto permutado. No se exige una sigmoide. Apagado verificado no demuestra utilidad del mecanismo inactivo ni aprendizaje causal del entorno. |
| 6. Contenido, memoria y revisión de metas en el mismo agente | Un mensaje tipado transporta contenido propio a consumidores distintos. La historia, memoria de consecuencias y metas persistentes participan en esa misma decisión nativa. | CONTENT contrasta permutación del contenido entregado a ambos consumidores. Lesiones individuales de consumidores, memoria y revisión de metas son secundarias y separadas. Ancho de interfaz conservado no equivale a contenido útil conservado. |
| 7. Descenso proximal correcto | Minimizar una pérdida explícita mediante gradiente con signo negativo y umbral proximal; no penalizar el intercepto salvo declaración contraria. | Verificar gradiente, objetivo, escala de paso, ceros inducidos, parámetros finitos y descenso bajo las hipótesis matemáticas especificadas. La reducción de pérdida de ajuste no demuestra mejor control. |
| 8. Estabilidad, rivales justos y familias nuevas | Declarar límites de estados, coeficientes, ganancias y acciones; evaluar sensibilidad y perturbaciones. Rivales adquieren su propia experiencia con presupuestos comparables. Usar familias ecology, inventory y thermal separadas. | Distinguir una garantía numérica/local de supervivencia real. Contabilizar datos, parámetros, ajustes, pasos y tiempo; conservar resultados adversos y familias por separado. No presentar tres nombres para una misma ley física como tres familias independientes. |

Las filas 1–2 fijan qué puede afirmarse; las filas 3–7 especifican mecanismos;
la fila 8 organiza los controles y el alcance. No son ocho teorías separadas
ni obligan a fabricar ocho tests de superioridad. Sus hipótesis estadísticas
concretas y relaciones con las intervenciones se fijan en PREREG_R9.md.

## 2. Ciclo de un solo agente y orden temporal

1. Recibir la observación permitida $o_t=(r_t,e_t,d_t)$.
2. Comparar la observación con el pronóstico emitido anteriormente, actualizar
   el buffer de errores pasados y calcular el contexto y gate. Una primera
   pasada usa la tensión anterior; de sus predicciones se obtiene la tensión
   actual para una segunda pasada. El mismo gate multiplica la contribución
   cruzada en las dos pasadas; no se consulta un futuro factual.
3. Entregar la observación al workspace. Los targets pendientes a horizontes
   1, 4 y 12 vencen cuando llega su observación propia correspondiente; no
   atraviesan reinicios. Actualizar memoria/historia, revisar metas y construir
   el mensaje tipado con procedencia y antigüedad.
4. Los dos consumidores usan ese mensaje. Pronósticos de red, Q normalizados,
   consecuencias recordadas y pesos de metas llegan al score y al filtro de
   acciones del mismo agente. El filtro combina predicción inmediata y
   memoria de energía con márgenes heurísticos explícitos.
5. Seleccionar y registrar una de las cuatro acciones; ejecutar una única
   transición. Conservar observación, acción, efectos y estado interno
   suficientes para reconstruir qué información causó la decisión.

El workspace usa una historia de 16 pasos y mensajes
inmutables. **StateEstimate** contiene observación, pendientes, dispersión y
cue derivado del contexto visible; **ActionConsequences**, estimaciones
factuales por acción/horizonte y recuentos; **GoalState**, prioridad, pesos,
edad y presiones. Los consumidores de viabilidad y recurso son computaciones
distintas con entrada común declarada. ActionConsequences entrega estimaciones
absolutas de las tres observaciones, con persistencia cuando no hay experiencia.
Los cambios asociados a una acción a horizontes 4 y 12 incluyen las acciones
posteriores y el ambiente: no son efectos marginales aislados. Nombrar dos
funciones no demuestra que ambas dependan del contenido.

Las metas pertenecen a un repertorio diseñado: supervivencia, recurso y tarea.
Su persistencia, revisión y abandono pueden probarse como operaciones; no
implican que el agente invente fines morales. Del mismo modo, pisos normativos
de energía/recurso y reglas de prioridad son impuestos por el diseñador.
Comunicación entre consumidores de software no es negociación entre agentes
socialmente autónomos.

La reconsolidación operacional identifica una entrada existente recordada
desde el origen del evento, nueva evidencia con error absoluto medio al menos
.08 y una actualización retenida a tasa .6, registrada antes/después. No es
una equivalencia con reconsolidación biológica. Las metas usan un repertorio
diseñado y una histéresis: permanencia mínima de tres observaciones y candidata
durante dos observaciones antes de sustituir la prioridad bajo su regla de
presiones. La memoria explícita se distingue de parámetros aprendidos,
pronósticos anteriores y demás estado retenido por cada lesión.

## 3. Aprendizaje, soporte y límites de las garantías

La entrada predictiva X tiene 48 componentes: 16 canales operacionales y
32 componentes de tensión, cuatro para cada uno de ocho pares. Y tiene cuatro:
recurso siguiente, energía siguiente, demanda siguiente y recompensa factual
de la acción ejecutada. No se atribuyen ocho dimensiones independientes o
constructos psicológicos a los ocho pares.

Para una acción $a$, sea $A_a=[1,Z_a]$ el diseño normalizado sólo con datos de
entrenamiento y $Y_a$ las cuatro etiquetas factuales propias. El objetivo es

$$
J_a(b_a,B_a)=\frac{\|A_a\Theta_a-Y_a\|_F^2}{2n_a}
 +\frac{\rho}{2}\|B_a\|_F^2+\lambda\|B_a\|_1,
\qquad \Theta_a=(b_a,B_a).
$$

Con gradiente $G_a$ de la parte suave, el intercepto hace descenso ordinario y
los coeficientes usan

$$
B_a^{k+1}=\operatorname{soft}(B_a^k-\eta G_{B,a}^k,\eta\lambda),
\qquad b_a^{k+1}=b_a^k-\eta G_{b,a}^k.
$$

La operación soft es $\operatorname{sign}(x)\max(|x|-u,0)$. Para datos,
normalización y pérdida fijos, un paso $0<\eta\leq1/L_a$, con $L_a$ cota del
máximo autovalor de la Hessiana suave, permite justificar descenso proximal
del objetivo convexo. La implementación exige al menos dos muestras por acción,
usa intercepto y fija $L_a=\lambda_{\max}(H_a)(1+10^{-12})$,
$\eta=1/L_a$. Ejecuta exactamente 80 pasos por acción; no afirma alcanzar el
óptimo con ese presupuesto. La comprobación de batch fijo no garantiza convergencia
de un aprendizaje online con distribución cambiante, óptimo con una cantidad
finita de pasos, calidad fuera de distribución ni estabilidad del agente.

El soporte predictivo aprendido se define por los coeficientes no nulos bajo
una convención numérica congelada. Los rivales denso y fijo usan pérdidas,
entradas, targets y presupuestos explícitos, con las restricciones que los
definen; se informa la diferencia de soporte efectivo. Si se añade una
proyección, su conjunto y efecto sobre el argumento proximal se declaran:
no se hereda automáticamente la garantía al componer operaciones arbitrarias.
En el modo fixed actual se restringen coordenadas a un subespacio fijo, una
restricción compatible con ese objetivo proximal. El clipping posterior del
actor sobre pronósticos y acciones es una operación distinta.

Los cuatro modelos por acción almacenan 784 coeficientes, incluidos 16
interceptos, con hasta 768 pendientes. El soporte fijo permite 384 pendientes
y su máscara se fija sin consultar etiquetas. El rival scalar suma las cuatro
tensiones por par en el primer slot y pone cero en los otros tres: conserva
48 slots y 784 coeficientes asignados, pero tiene menor dimensión y capacidad
efectiva. No duplica columnas para simular igualdad de capacidad. Los recuentos
efectivamente activos y las diferencias de cómputo se informan por separado.

La compuerta contextual implementada es
$g_t=\mathbf{1}[\theta^\top c_t>\delta]$, con
$c_t=(1,r_t,e_t,d_t,\overline{\mathrm{mismatch}}_t,u_t)$ y
$\delta\in\{0,.001,.01\}$. El modo off fija exactamente $g_t=0$.
Los coeficientes estiman por ridge el beneficio predictivo factual
MSE(local) menos MSE(completo) en la acción observada durante el episodio 201;
no aprenden directamente una garantía de utilidad nativa. El gate elegido
se aplica tanto al pronóstico preliminar con tensión anterior como al
pronóstico con tensión actual. noCross pone cero en ambas pasadas para impedir
que la ruta cruzada reaparezca indirectamente por la tensión.

Las cuatro tensiones son: media de error absoluto del par respecto del
pronóstico previo; $0.2p^+p^-$; media entre acciones 1, 2 y 3 de
$\max(0,-\Delta_a^+\Delta_a^-)$ respecto del pronóstico de la acción 0; y
$u_t$. El coeficiente .2 y la referencia acción 0 son decisiones del diseño.
La incertidumbre operacional es
$u_t=\operatorname{mean}(m_{r,e,d})+
\sqrt{\operatorname{mean}(\mathrm{errores}_{16\times3}^2)}$:
márgenes empíricos del episodio 200 más RMSE de los últimos 16 errores
observados de pronósticos previamente emitidos, con buffer inicialmente cero.
No utiliza etiquetas futuras del paso actual y no es un intervalo calibrado
con cobertura garantizada.

Clipping, ganancia acotada y máscara de acciones permiten demostrar sólo sus
propiedades algebraicas. Respetar una máscara derivada de predicciones no
asegura satisfacer límites físicos reales cuando el predictor se equivoca.
Si ninguna acción es admisible, el fallback se registra como tal, nunca como
acción certificadamente segura. Un argumento de contracción necesitaría
hipótesis sobre todo el operador cerrado, incluido entorno, memoria,
compuerta y cambios de parámetros; no se infiere del signo del gradiente.

## 4. Comparación, familias y transferencia

Ecology, inventory y thermal tendrán ecuaciones, semántica de acciones,
perturbaciones y objetivos especificados por separado. Su interfaz común
de tres observaciones y cuatro acciones facilita comparación, pero no prueba
identidad causal entre sus estados. El mapa entre dominios y cualquier
normalización se congela antes de evaluar transferencia.

Los cuatro tipos full, dense, scalar y fixed recogen experiencia bajo
**sus propias políticas** con iguales
oportunidades de interacción, validación y ajuste. Un dataset común puede ser
un diagnóstico secundario de la regla de ajuste, claramente separado de la
comparación completa. Se registran presupuestos y soportes efectivos; igualar
arrays almacenados no equivale a igualar capacidad ni coste aritmético.

Las intervenciones sobre un checkpoint congelado estiman dependencia de esa
realización. Los rivales reentrenados estiman rendimiento bajo otra oportunidad
de aprendizaje. No intercambiar ambos estimandos. El pareamiento conserva
semillas y perturbaciones exógenas, no exige que dos políticas produzcan una
misma trayectoria física endógena.

La transferencia a inventory y thermal es **zero-shot respecto a parámetros**:
sus observaciones y resultados no participan en ajuste de pesos, normalización,
hiperparámetros o selección de gates. La actualización episódica de memoria
sigue el algoritmo congelado, con el mismo permiso para el rival. Cualquier
adaptación adicional necesitaría declarar su presupuesto y cambiar el alcance
del contraste. Que los tres generadores tengan leyes diferentes constituye
amplitud interna del banco; no sustituye un generador externo o una replicación
por otro equipo.

### Presupuesto y fronteras de selección vigentes

Cada uno de los cuatro tipos usa ocho episodios de entrenamiento de 320 pasos,
con ajuste cada dos episodios, seguidos por dos episodios de calibración
separados. Entrenamiento y calibración emplean gate provisional .5 y
exploración .2; los probes uniformes pueden sobrepasar la máscara y se registran
por separado. Q aprende sólo en entrenamiento; el sucesor común se elige con
el agregado Q, no con todo el score predictivo del actor.

El episodio 200 estima márgenes por output como percentiles .90 del error
absoluto factual. El 201, con esos márgenes ya fijados, ajusta el gate con otro
conjunto de transiciones propias. Q y predictor están congelados en ambos.
La escala de error no se obtiene del propio test ni de futuros inaccesibles.

La selección usa siete modos únicos con clones del mismo checkpoint y el tape
del episodio 300. Hay cuatro candidatos contextuales (off y umbrales 0,
.001, .01) y cuatro constantes (off, .25, .5, 1), con off compartido. Ambos
conjuntos eligen por la misma regla de viabilidad y recompensa. Los estados de
memoria adquiridos por cada candidato de validación no se acumulan entre
candidatos ni se incorporan al checkpoint elegido. El piso de vida medido
por el evaluador se usa explícitamente en esta selección offline; la política
paso a paso no recibe las variables físicas latentes.

Los pilotos evalúan sólo ecology_train y ecology_delay9. Inventory y thermal
permanecen reservados hasta el freeze final, sin selección de parámetros o
arquitectura a partir de sus resultados. En evaluación quedan congelados Q,
predictores, normalización, márgenes y coeficientes del gate; el workspace
continúa actualizando su memoria y metas con el pasado propio según el
algoritmo fijado. Zero-shot se refiere a parámetros sin ajuste en las nuevas
familias, no a prohibir estado episódico observable.

## 5. Cierre prospectivo y preservación

Antes de finales: congelar código, diseños, semillas, splits, targets, orden
temporal, normalización, familias, comparadores, presupuestos, máscaras,
fallbacks, endpoints y regla estadística. Registrar qué episodios fueron
desarrollo y toda revisión motivada por ellos. Los números finales no
modifican retrospectivamente una definición, margen o familia de hipótesis.

El cierre de ejecución requiere integridad de artefactos, reconstrucción de
endpoints y replays declarados, además de resultados para todas las semillas
y condiciones, incluidas muertes, gates cerrados y efectos adversos. La
unidad inferencial y multiplicidad siguen PREREG_R9.md, no el número de
pasos ni de mensajes. Un test que pasa cierra su contrato concreto; los
resultados de utilidad, H_REL y H_TRANSFER conservan después su propio veredicto.
H_CAT y conciencia fenomenal permanecen fuera de lo establecido por este
diseño. No se reescriben R8 ni el manuscrito hasta una revisión posterior
explícita basada en la evidencia realmente obtenida.
