# R9: soporte predictivo disperso y tensión vectorial

Diseño de una implementación nueva. R8 y sus conclusiones permanecen intactos. Este módulo no reproduce el regulador inter-polar original: predice observaciones siguientes y recompensa factual mediante relaciones entre características observadas. Los coeficientes, sus signos y los ceros seleccionados no identifican por sí solos relaciones causales ambientales ni constructos psicológicos.

## Contrato

```python
model = SparseActionModel(
    n_features=48, n_outputs=4, mode='sparse',
    l1=0.01, ridge=0.01, iterations=80,
)
model.fit(X, own_action, Y, split='train')
forecast = model.predict_all(X_current)       # [..., 4 actions, 4 outputs]
local = model.predict_local_all(X_current, local_mask)  # same shape
parts = model.predict_components(X_current, local_mask)
portable = model.state_dict()                # JSON-native, no object arrays
restored = SparseActionModel.from_state_dict(portable)
```

`fit` recibe X de forma [n,d], acciones enteras propias en {0,1,2,3} y Y [n,k]. Exige al menos dos observaciones factuales por acción; no completa acciones sin cobertura mediante estados verdaderos, simuladores contrafactuales ni oráculos. `predict_all` acepta también batches multidimensionales. El entrenamiento se rechaza si `split` no es `train`; la separación real de episodios y procedencia corresponde al orquestador.

Contrato del agente R9 acordado con la raíz:

- X contiene 16 polos observables más 32 componentes de tensión, en ese orden. Los componentes se aplanan por par: cuatro valores de cada uno de ocho pares.
- Y contiene reserva siguiente, energía propia siguiente, demanda siguiente y recompensa factual de la acción que acaba de ejecutarse. Los tres primeros son observaciones posteriores, no estado oculto.
- La predicción actual sólo recibe X. Y aparece únicamente después de ejecutar la acción y pasa a ser etiqueta de entrenamiento; nunca es entrada de la política del mismo paso.
- Las salidas son lineales sin clipping interno. El actor es responsable de la proyección física que corresponda y debe registrar su efecto; no se afirma estabilidad del sistema cerrado a partir del ajuste convexamente regularizado.

## Objetivo y descenso proximal

X se centra y escala con la media y desviación estándar del conjunto de entrenamiento completo, sin usar desarrollo o test. El piso de escala es 10⁻⁶, independiente de los resultados. Los objetivos permanecen en sus unidades observacionales; el orquestador debe declarar sus escalas. No se normaliza cada nueva consulta con sus propios datos.

Para una acción a, sea A=[1,Z], Θ=[b;B] y n_a su número de transiciones factuales. Se minimiza:

\[
J_a(\Theta)=\frac{1}{2n_a}\|A\Theta-Y\|_F^2
 +\frac{\rho}{2}\|B\|_F^2+\lambda\|B\|_1.
\]

El intercepto b no se penaliza. Las salidas se suman en el error de Frobenius; no se divide adicionalmente por k. El objetivo público del modelo es la media de J_a sobre las cuatro acciones, con peso igual por acción. Las funciones aprendidas por acción se ajustan independientemente.

Con D=diag(0,1,…,1), G=AᵀA/n_a y C=AᵀY/n_a:

\[
H=G+\rho D,\qquad L=\lambda_{\max}(H)(1+10^{-12}),\qquad \eta=1/L,
\]

\[
\widetilde\Theta=\Theta-\eta(H\Theta-C),\qquad
b' = \widetilde b,\qquad
B'=\operatorname{sign}(\widetilde B)
\max(|\widetilde B|-\eta\lambda,0).
\]

Se inicializa B=0 y b como media de Y para cada acción. Se ejecutan exactamente 80 pasos por acción, sin parada temprana ni reajuste del presupuesto según resultados. El multiplicador conservador de L evita una pequeña subestimación numérica de la constante de Lipschitz. Los gradientes usan estadísticas suficientes G y C; la pérdida registrada se comprueba también directamente contra los residuos finales.

La proyección del modo fijo pone en cero las coordenadas excluidas por una máscara binaria fija. Es una restricción a un subespacio de coordenadas compatible con este objetivo convexo y la cota de paso; no es una proyección arbitraria de seguridad física. La garantía de descenso del objetivo de ajuste no implica estabilidad de la interacción del agente con su entorno.

## Controles y presupuesto

| Modo | Penalización L1 efectiva | Soporte permitido |
|---|---:|---|
| `sparse` | λ configurado | Todas las características; el proximal selecciona ceros exactos |
| `dense` | 0 | Todas las características |
| `fixed` | 0 | max(1,floor(d/2)) características por salida y acción |

La máscara fija se genera mediante PCG64, semilla de configuración 1729, antes de observar datos; no depende de la semilla experimental, etiquetas, semántica de los polos ni resultados. Para d=48 permite exactamente 24 entradas por salida y acción. Su soporte no se iguala retrospectivamente al número de coeficientes que retenga el modelo disperso.

Los tres modos reciben los mismos tipos de observaciones, dimensiones, arquitectura de salida y presupuesto de pasos. Realizan productos matriciales densos de las mismas dimensiones, incluso si muchos coeficientes son cero. Para d=48 y k=4 almacenan 784 coeficientes incluyendo los 16 interceptos; el máximo de pendientes es 768 y el soporte fijo permite 384. Ejecutan 320 pasos de gradiente y 324 evaluaciones de objetivo por ajuste. Las actualizaciones usan matrices H de 49×49 y Θ de 49×4.

Se registra capacidad permitida y efectiva por separado: el ajuste disperso, el denso y el fijo no tienen necesariamente el mismo número de parámetros activos ni la misma capacidad estadística. El cómputo de soporte/umbral añade operaciones elementales, por lo que no se afirma igualdad exacta de FLOPs o tiempo total. Si se comparan políticas con adquisición propia, sus transiciones realizadas pueden diferir, aunque el presupuesto y acceso informativo sean iguales; el orquestador debe distinguir ese contraste del entrenamiento con datos comunes.

## Lesión local sin reajuste

`local_mask[d,k]` es binaria y común a las cuatro acciones. La lesión usa:

\[
\widehat Y^{local}_a(X)=b_a+Z(B_a\odot M),\qquad
\widehat Y^{cross}_a(X)=Z(B_a\odot(1-M)).
\]

Conserva medias, escalas e intercepto entrenados. En X igual a la media de entrenamiento, ambos modelos predicen b exactamente; no se introduce un desplazamiento al borrar características centradas. La suma local+cross reproduce el predictor completo salvo redondeo aritmético. No se reajustan coeficientes, y la función no modifica el estado.

La raíz definió la máscara operacional de R9: reserva usa par 2; energía par 0; demanda par 1; recompensa pares 1 y 3. Cada salida conserva los dos canales y los cuatro componentes de tensión de sus pares asignados. Los bloques de tensión empiezan en índice `16+4*pair`. Es una asignación explícita del diseño, no un descubrimiento semántico de pares propios.

## Cuatro componentes de tensión

```python
tau = tension_vector(
    poles, previous_prediction, predicted_poles_all, uncertainty,
    coactivation_weight=0.2, n_pairs=8,
)
```

Se admite cualquier número positivo de pares; ocho es el contrato integrado. Para p de forma [...,pairs,2], la salida es [...,pairs,4], con orden fijo:

1. `mismatch`: media de |p_t−forecast_t| sobre los dos canales; el forecast fue emitido antes de recibir p_t. En el paso inicial el orquestador declara el predictor de persistencia empleado.
2. `coactivation_weighted`: χ·p⁺·p⁻, con χ∈[0,1], por defecto 0,2. Es coactivación ponderada; no se identifica con competencia por recursos ni incompatibilidad psicológica.
3. `predicted_opposition`: media sobre acciones 1,2,3 de max(0,−Δ_a⁺Δ_a⁻), donde Δ_a es el forecast actual bajo acción a menos el forecast actual bajo acción 0. Mide cambios predichos en sentidos opuestos respecto a esa acción de referencia; no prueba efectos causales ni imposibilidad de satisfacer objetivos simultáneamente.
4. `uncertainty`: escala no negativa de error predictivo suministrada por el orquestador, calibrada con observaciones propias anteriores de entrenamiento/desarrollo. No es la dispersión entre acciones, que puede reflejar controlabilidad. El campo `residual_rmse_` del predictor es un diagnóstico in-sample y no reemplaza esa calibración.

Las entradas p, forecast previo y forecasts de candidatos deben estar codificadas en [0,1]. La incertidumbre conserva su escala explícita y no se recorta silenciosamente. El predictor posterior normaliza las 48 características con entrenamiento solamente.

El agente ejecuta dos pasadas: una primera predicción usando la tensión anterior, codificación de los forecasts de observaciones como pares mediante historia pública, y construcción de la tensión actual para la segunda pasada. La misma compuerta contextual multiplica la contribución cruzada en ambas pasadas; off y noCross eliminan las dos rutas actuales. Se aplica a todos los comparadores, sin iterar hasta convergencia ni consultar futuros factuales. El episodio 200 calibra la incertidumbre a partir de errores sobre transiciones propias; el episodio 201 ajusta la puerta con transiciones distintas. La selección posterior de su modalidad se realiza en el episodio 300. La especificación completa, incluida la conservación del forecast polar emitido antes de observar el siguiente estado, pertenece al orquestador.

Un array no certifica su procedencia temporal. Las trazas deben guardar el forecast emitido, observación recibida, identificación de la acción, parámetros congelados y partición usada para calibrar. Un coeficiente aprendido desde transiciones observacionales sigue siendo predictivo salvo un diseño adicional de identificación causal.

## Estado y comprobaciones

El checkpoint JSON incluye configuración, máscara permitida, coeficientes, interceptos, normalización, curvas de objetivo de las cuatro acciones, pasos Lipschitz, diagnóstico residual y SHA256 de los arrays de entrenamiento. La restauración valida dimensiones, finitud, escalas, soporte determinista y ausencia de coeficientes prohibidos. No usa pickle. `parameter_digest()` permite comprobar que predicciones y lesiones no alteran parámetros.

Las pruebas unitarias verifican descenso en los tres modos; signo de los coeficientes; ceros exactos por proximal; igualdad sparse con λ=0 y dense; soporte fijo independiente de etiquetas; conservación del centro/intercepto; serialización exacta; normalización sólo de entrenamiento; cobertura mínima por acción; ausencia de argumentos de futuro en predicción; y definición separada de los cuatro componentes. Son contratos sobre datos sintéticos deterministas, no experimentos piloto ni demostraciones de utilidad.

No se ejecutarán pilotos o finales de este módulo fuera de la autorización y congelación de la raíz. Ningún cambio en este módulo reinterpreta resultados R8 anteriores.
