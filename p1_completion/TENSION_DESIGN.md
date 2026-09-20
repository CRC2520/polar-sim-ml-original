# P1: propagación explícita de tensión; protocolo antes del final

Estado: extensión experimental 2026-09-20. Fuente preservada: `CRC2520/polar-sim-ml-original@2d2d51120409ad39559e1311897bd6b99bbf748e`, `network_tension.py` (blob `037e7a4464372b3e4e5067502326e2387a893470`) y dependencias exactas en `source_network_v01`. No se modifican los estudios congelados ni se afirma integrar todas las capas propuestas. Los ocho tipos y dos canales por tipo son los de la fuente; sus nombres son etiquetas operativas, no constructos psicológicos validados.

## Hipótesis y ecuación recuperada

`τ = mean_pole(abs(clip(d/G,0,1)−q)) + χ q+ q−`; `proposal = q + η(d/G−q) + η(Wq + Kτ)`. La factibilidad heredada aplica cotas, permisos y presupuesto. W/K son parámetros configurados y fijos, no topología aprendida. La primera hipótesis es que una intervención interna localizada produce efectos mediados por estas rutas aun cuando el presupuesto no vincula. La segunda, independiente, es que añadirlas reduce error externo en tareas preexistentes que no se generan a partir de W/K.

## Intervenciones, controles y semillas

Piloto: 930201–930203. Final reservado: 934001–934030. Cada semilla contiene pulsos positivos de 0.2 en cada uno de 16 canales (8 polaridades) y ruido exógeno común σ=0 o 0.02, 12 pasos. Estado inicial aleatorio [0.15,0.45], objetivo original mantenido y perturbado únicamente por la secuencia común de ruido. Ambos miembros de cada pareja comparten estado restante, observaciones, presupuesto=16 no vinculante y χ=0.5. Memoria y adaptación de ganancias se desactivan sólo en la prueba de mecanismo. La intervención es `do(q[source,pole] += 0.2)` al inicio; no se cambia el ambiente ni el objetivo.

Grafo predeclarado: ciclo dirigido 0→1→…→6→0 y tipo 7 aislado; en cada arista W conserva polo con pesos 0.12/0.08 y K recibe tensión con pesos +0.18/−0.14. Es una configuración experimental nueva del mecanismo exacto, no una red descubierta. Se registran el grafo, sus cambios y la semilla de permutación.

Controles: red completa; cero W/K; W solo; K solo; lesión de arista 0→1 en W/K; aristas invertidas; permutación conjunta de tipos conservando grados, pesos y dispersidad; coordenadas signo/intensidad; cálculo genérico sobre vector plano algebraicamente equivalente; recurrente convencional con mismo W/K y memoria/modelo de capacidades; mediador τ fijado paso a paso al valor de la pareja sham con W intacto. El último permite separar una vía K de la propagación residual por W; no mide una mediación natural en un organismo.

## Desempeño externo

Se reutilizan sin cambios `polar.tasks.make_trial` para `switching_memory` y `gain_resource_shift`, split heldout, 96 pasos y 3 unidades×8 tipos. Cada variante recibe exclusivamente `observation(frame)`; gains/oracle/target oculto permanecen en el evaluador. Se comparte el ruido de feedback entre variantes. W/K se repite por unidad; los targets originales nunca reciben las matrices para construirse. χ=0 en estas tareas porque no suministran incompatibilidad contextual; no se inventa conflicto. Se comparan red, cero-red, W/K separados, permutación, inversión, coordenadas, vector plano y recurrente con red; se incluyen recurrente sin red y lesión de arista.

El rival recurrente conserva parámetros, estado, topología, memoria, aprendizaje y restricciones; sólo cambia la recurrencia base por el gradiente proyectado heredado. El bloque de red tiene igual costo de cálculo; el factor peso/ganancia añade aritmética en la recurrencia y por ello no se afirma igualdad exacta de FLOPs. Se registra tiempo de ejecución descriptivo, sin usarlo para seleccionar modelos. El vector plano es control de consistencia, no rival independiente.

## Métricas y decisión predeclarada

Mecanismo: diferencia pareada por canal y tiempo; primer instante que supera 1e-10; coincidencia con distancia dirigida; efectos fuera de rutas; amplitud/cambio de signo; efecto de lesión y de clamp τ; reproducibilidad; igualdad de observaciones; presupuesto; error máximo de equivalencia 1e-12. La evidencia es una consecuencia del mecanismo implementado, no emergencia, conciencia ni exclusividad.

Desempeño: MSE ponderado y regret contra el oráculo factible del mismo episodio, costo, violaciones y recuperación sostenida. Dos contrastes primarios (una tarea cada uno): regret(red)−regret(cero-red); 30 unidades independientes son las semillas, no pasos/canales. IC bootstrap pareado 97.5% por tarea (Bonferroni aproximado para dos contrastes), 10 000 remuestreos fijos. Mejora práctica requiere límite superior <−0.005 en ambas tareas y cero violaciones; daño requiere límite inferior >+0.005. Todo otro resultado es mixto/no respaldado. Resto de contrastes secundarios: IC95% descriptivos. No se reoptimiza después del piloto.

Las trazas NPZ contienen entradas exógenas, intervención, estado antes/después, objetivos efectivos, tensión, términos de red, propuestas, acciones y feedback. Los resúmenes deben poder reconstruirse sin ejecutar los controladores; se conserva hash de fuente e inputs y no se permite sobrescribir salida existente. El final requiere un registro previo cuyo contenido y hash quedan copiados al resultado. Puede cerrarse el trabajo P1 aunque la hipótesis de ventaja resulte negativa.
