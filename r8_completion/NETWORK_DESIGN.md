# R8: red predictiva aprendida dentro del agente

Estado: diseño de desarrollo, previo a las corridas finales. P1-T queda intacto. Esta es una implementación nueva, `learned-network-0.1`, que aprende pesos predictivos W/K y permite seleccionar contribución nula. No reemplaza retrospectivamente la red reguladora configurada de P1-T y no convierte sus resultados negativos en positivos.

## Contrato integrado

`LearnedNetwork(actions=4, seed=..., ridge=0.1)` expone:

- `fit(poles[N,8,2], tension[N,8], action[N], next_poles[N,8,2], split='train') -> self`.
- `predict_all(poles[...,8,2], tension[...,8], variant='learned', gate=λ) -> [...,4,8,2]`.
- `operational_tension(poles, desired, chi=0)` calcula `mean(abs(desired-poles),pole) + chi*p_plus*p_minus`.
- `state_dict()` y `from_state_dict()` permiten guardar/reproducir todos los parámetros.

La implementación integrada calcula la tensión preacción con la predicción emitida en el paso anterior frente a los polos ahora observados; χ=0.2 será una hipótesis operacional explícita del agente, no incompatibilidad psicológica validada. Inicialmente la predicción es persistencia. Los objetivos de entrenamiento son los polos efectivamente observados en t+1 después de la acción propia; nunca se usa el futuro como entrada de t. Cada candidato de acción recibe su propia predicción de polos, que entra en las características del actor aprendido. No se añade una recompensa diseñada para preferir una red.

## Modelo, capacidad e identificación

Los canales p+ y p− son independientes en [0,1], admiten coactivación y no se exige p−=1−p+. Se conservan los ocho nombres fuente, con correspondencias operacionales condicionadas a las señales ecológicas declaradas por el agente. Un nombre no valida un constructo psicológico.

Para cada acción a y canal receptor i, primero se estima por ridge un modelo local de los dos canales de su propio par más intercepto:

`local_a,i(p) = b_a,i + L_a,i * standardized(p_own_pair)`.

Después se ajusta el residuo factual `next_p_i - local_a,i` con todos los otros canales y las tensiones de los otros pares:

`network_a(p,τ) = W_a @ (p - mean_train_p) + K_a @ (τ - mean_train_τ)`;

`next_hat_a = clip(local_a(p) + λ * network_a(p,τ), 0, 1)`.

Cada W tiene dimensión 16×16 con bloques propios nulos; cada K tiene dimensión 16×8 con el propio par excluido. Las variables se centran y escalan con entrenamiento solamente; los coeficientes publicados W/K se devuelven a unidades originales. Se registra cobertura de las cuatro acciones, dispersión de cada característica y condición de los sistemas regularizados. El soporte denso entre pares está fijado; ridge aprende pesos, no descubre topología ni causalidad ambiental. La estructura identifica relaciones predictivas condicionadas por la cobertura y colinealidad de observaciones, no ocho conceptos psicológicos.

Hay 192 parámetros locales y 1 344 parámetros entre pares para cuatro acciones; capacidad persistente de coeficientes: 1 536 escalares. Los comparadores nativos deben recibir las mismas observaciones, memoria, características y presupuesto de entrenamiento/selección; disponer de un control plano equivalente no satisface por sí solo ese requisito.

## Selección y comparadores

La regularización se fija en 0.1 antes de finales. La contribución global λ∈{0,0.25,0.5,1} se selecciona exclusivamente con utilidad nativa de desarrollo y restricciones de viabilidad del protocolo integrado. El valor cero es admisible. No es una puerta autónoma que cambie por estado. Los comparadores reciben igual cantidad de candidatos y datos de desarrollo; el empate favorece λ menor. Una selección negativa o nula es un resultado válido.

Las variantes `off`, `fixed` (anillo P1), `random` (soporte y norma global de pesos igualados), `scrambled` (reordenación de pares conservando pesos) y `learned` comparten el mismo modelo local aprendido. Son controles de contribución y correspondencia, no todos rivales aprendidos de igual capacidad efectiva. `generic_flat` implementa la misma función sin etiquetas semánticas: es consistencia algebraica. La inversión de coordenadas signo/intensidad conserva los dos grados de libertad y debe reproducir las predicciones.

El rival efectivo `JointRidgePredictor` está implementado con la misma API. Cada acción/salida ajusta conjuntamente los 16 canales observados, siete tensiones de otros pares e intercepto: 24 coeficientes por salida/acción, 1 536 en total. Recibe exactamente los mismos datos y normalización. Se diferencia del ajuste secuencial local→residuo; no es una simple reparametrización. En la corrida integrada recibe el mismo actor, memoria, cabezas de consecuencias y presupuesto de desarrollo. El costo aritmético de resolver un sistema 24×24 frente a dos sistemas 3×3 y 21×21 difiere y debe registrarse sin afirmar igualdad exacta de FLOPs.

## Comprobación y alcance

Se comprueban aislamiento de propios pares, lesiones de W/K, intervención localizada manteniendo τ/observaciones restantes, actualización síncrona, retardo multisal­to en un grafo unitario controlado, equivalencia de coordenadas, serialización, rechazo de ajuste en test y selección de λ=0 cuando la red perjudica desarrollo. Las pruebas unitarias con sistemas sintéticos conocidos no constituyen la prueba de beneficio: la utilidad se mide en el ambiente nativo del agente con targets factuales no generados a partir de W/K.

Desarrollo reservado: semillas 940201–940299. La decisión de la raíz es probar la contribución funcional de la red **dentro del bloque integrado 942...**; no se ejecutará una campaña final 943... adicional para duplicar ese contraste. El error predictivo de siguiente observación es secundario. Ningún final se ejecuta sin congelación de la raíz. La familia global usa seis afirmaciones primarias, éxitos por semilla frente al margen predeclarado, contraste binomial exacto de probabilidad de éxito ≤0.5 y ajuste Holm; los intervalos de medias son descriptivos. Los márgenes de desempeño nativo, potencia y episodios ID/OOD se fijan en el protocolo integrado antes de finales. λ=0 seleccionado en desarrollo implica ausencia de ganancia atribuible a la red, no un fallo de ingeniería ni una reclasificación favorable. Fuentes P1-T y sus conclusiones permanecen congeladas.
