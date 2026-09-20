# Auditoría piloto de tensión P1

Piloto 930201–930203, salida utilizable `results/p1_tension_pilot_v2_20260920`. La primera salida conserva tres archivos raw completos, pero interrumpió el resumen al serializar un escalar `numpy.int64`; se corrigió exclusivamente la conversión JSON y se añadió una prueba. No se seleccionaron parámetros a partir del desempeño.

El segundo piloto termina 1 056 pares de intervención (25 344 pasos de control), 66 episodios externos (6 336 pasos), 14 pruebas nuevas y 14 pruebas upstream. Verificación desde raw: todas las filas/resúmenes se reconstruyen exactamente y el error máximo de las ecuaciones es 1.11e-16. El campo `source_unchanged=false` de esa verificación piloto corresponde a añadir `verify()` al runner una vez iniciada la corrida; el algoritmo de simulación, semillas, tareas y criterio permanecieron iguales. El final debe congelar la fuente actual y exigir `source_unchanged=true`.

Red completa: 504 de 504 destinos esperados alcanzados; ningún efecto fuera de rutas ni llegada antes de la distancia dirigida. El presupuesto nunca vincula. Al eliminar W/K la respuesta en otras polaridades es exactamente cero. Al fijar τ a la trayectoria sham, el efecto cruzado L1 promedio baja de 0.0788214 a 0.0300591 y permanece la vía W. El control vectorial plano es idéntico; el máximo error de coordenadas alternativas es 1.55e-15. Esto confirma el funcionamiento de rutas programadas.

En los dos benchmarks anteriores a la extensión, la diferencia de regret `full−off` es +0.0006577 y +0.0009526 (descriptivo, sólo tres semillas). No satisface la mejora práctica predeclarada. No se ajusta la red para invertir este resultado. No se observan violaciones de restricciones.

El control genérico recurrente coincide con la recurrencia base en el mecanismo aislado (ganancias=pesos=1); informa una comparación distinta sólo en las tareas con adaptación/ponderaciones. El benchmark oráculo es factible y minimiza el objetivo determinista de la fuente; el ruido y recorte de efectos pueden producir regret negativo por muestra sin implicar un error de implementación.

Alcance: ni el grafo ni los ocho tipos son descubiertos. Estas dos tareas no validan la red en todo entorno con dependencias cruzadas o desalineadas; permiten evaluar su utilidad en tareas previas sin que W/K construya las respuestas correctas. Un resultado negativo deja abierta la investigación de estructura aprendida y utilidad transferible, pero cierra la comprobación técnica definida para este bloque.
