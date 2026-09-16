# B1-SC-D4 v1.0 — Transición tardía de adquisición de rutas

**Estado: DESIGN_FROZEN_NOT_EXECUTED.**

## 1. Linaje y límite inferencial

D4 sigue al cierre inmutable de D3 (`7842ef2933edf4800d59f06af79f0368cf03cd9d`). D3 cerró con integridad 32/32, `H_INFO_UTILITY=false`, `H_INFO_CAUSAL=false` y `NO_REPRODUCIBLE_PARTITION_INDUCED_S6_NECESSITY`. D4 no altera esa conclusión.

La observación de que D3 tuvo competencia LOCAL-S6 únicamente en los bloques 6–7 es retrospectiva y solo genera la hipótesis D4. Esos bloques **no se seleccionan, no se reutilizan, no se ponderan, no se estratifican y no se usan para derivar seeds o snapshots D4**. D4 usa 12 bloques nuevos e independientes.

## 2. Pregunta

¿La heterogeneidad observada en D3 se explica por una transición de adquisición de comunicación S6 durante el último cuarto del entrenamiento, o la disponibilidad temporal de las rutas no produce un mecanismo reproducible?

## 3. Manipulación estructural

Todas las condiciones usan exactamente la misma arquitectura `D3RoutingActor`, información `LOCAL_PARTITIONED`, soporte potencial S6 `100110`, crítico global 93-D y el mismo presupuesto PPO. Solo cambia el gate funcional `lambda` de las tres rutas activas S6.

Ventanas congeladas sobre índices de transición cero-based:

- temprana: `[0, 786432)`;
- tardía: `[786432, 1048576)`.

El checkpoint de 786.432 se evalúa **antes** de activar la ventana tardía.

Tres condiciones por bloque:

1. `LOCAL-S6-OFF`: lambda activo = 0 temprano y 0 tardío;
2. `LOCAL-S6-LATE`: lambda activo = 0 temprano y 1 tardío;
3. `LOCAL-S6-ALWAYS`: lambda activo = 1 temprano y 1 tardío.

No se cambia máscara, observación, recompensa, hiperparámetros, presupuesto ni familia de política.

## 4. Replicación e inicialización

- 12 bloques D4 frescos;
- 3 condiciones por bloque;
- 36 fits;
- 1.048.576 transiciones nativas por fit;
- 37.748.736 transiciones totales.

Se requieren 12 snapshots canónicos nuevos, uno por bloque. Dentro de cada bloque, las tres condiciones comparten exactamente los mismos bytes trainables iniciales. Está prohibido reutilizar snapshots, pesos, checkpoints o seeds de D3.

El umbral de replicación es 9/12, conservando prospectivamente la proporción 75% del gate 6/8 usado en D3.

## 5. Diagnóstico temporal

Checkpoints diagnósticos, sin selección ni early stopping:

`262144, 524288, 786432, 851968, 917504, 983040, 1048576`.

Se ejecutan 32 episodios deterministas emparejados por checkpoint y bloque. Los cuatro checkpoints del último cuarto permiten observar si aparece una transición después del switch sin redefinir retrospectivamente su momento.

## 6. Endpoint y panel causal

A 1.048.576 se ejecutan 64 episodios emparejados.

Modo configurado:

- OFF se evalúa con rutas activas anuladas;
- LATE y ALWAYS se evalúan con las rutas S6 activas.

Además, para LATE y ALWAYS se ejecuta un panel causal independiente de 64 episodios por régimen:

- intact/configured;
- permanent_all_active_lesion desde decisión 0 a terminal;
- sham;
- inactive_edge_control sobre 0→2.

Para el análisis de scaffold se evalúan también LATE y ALWAYS con lesión permanente en el panel endpoint emparejado. No hay reentrenamiento durante evaluación.

## 7. Gates primarios

Se conservan los márgenes históricos: loss `130/30`, displacement `1.0`, unidad primaria = bloque independiente, bootstrap 10.000 solo descriptivo.

### H_D4_LATE_TOTAL

Contraste `LOCAL-S6-LATE configured` vs `LOCAL-S6-OFF configured`. Pasa si:

1. el agregado cumple ventaja práctica;
2. >=9/12 bloques cumplen ventaja práctica;
3. LATE es competente frente a no-action en >=9/12 bloques.

Este gate pregunta si abrir rutas solo en el último cuarto es suficiente para producir una transición reproducible total.

### H_D4_EARLY_PREPARATION

Contraste `LOCAL-S6-ALWAYS configured` vs `LOCAL-S6-LATE configured`. Pasa si:

1. el agregado cumple ventaja práctica;
2. >=9/12 bloques cumplen ventaja práctica;
3. ALWAYS es competente frente a no-action en >=9/12 bloques.

Este gate pregunta si la exposición temprana prepara un basin al que LATE por sí sola no llega.

### H_D4_LATE_SCAFFOLD

Contraste `LOCAL-S6-LATE permanent lesion` vs `LOCAL-S6-OFF configured`, ambos ejecutados sin rutas activas. Pasa si el agregado cumple ventaja práctica, >=9/12 bloques cumplen ventaja práctica y la política LATE lesionada es competente en >=9/12 bloques.

Este gate identifica un efecto de aprendizaje que persiste después de retirar las rutas.

### H_D4_ONLINE_LATE y H_D4_ONLINE_ALWAYS

Para cada condición tardíamente activa se compara intact vs lesión permanente. Cada gate requiere deterioro práctico agregado, >=9/12 bloques con deterioro práctico y controles sham/inactive exactos en 12/12 bloques.

## 8. Adjudicación explicativa

La explicación primaria usa solo `H_D4_LATE_TOTAL` y `H_D4_EARLY_PREPARATION`:

- true / false → `LATE_ROUTE_WINDOW_SUFFICIENT_FOR_REPRODUCIBLE_TRANSITION`;
- true / true → `LATE_ROUTE_WINDOW_SUFFICIENT_WITH_EARLY_PREPARATION_BENEFIT`;
- false / true → `EARLY_ROUTE_PREPARATION_REQUIRED_FOR_REPRODUCIBLE_TRANSITION`;
- false / false → `NO_REPRODUCIBLE_ROUTE_TIMING_EXPLANATION`.

`H_D4_LATE_SCAFFOLD`, `H_D4_ONLINE_LATE` y `H_D4_ONLINE_ALWAYS` son modificadores mecanísticos preespecificados y no cambian esa tabla primaria. Si falla integridad, el estado es `INVALID_OR_INCONCLUSIVE_D4_ADJUDICATION`.

## 9. Anti-cherry-picking

D4 debe aplicar las tres condiciones a los 12 bloques nuevos completos. Está prohibido:

- importar los bloques 6–7 de D3 como estrato;
- seleccionar inicializaciones por similitud con D3;
- reemplazar seeds por desempeño;
- excluir bloques por competencia;
- mover el switch de 786.432 tras observar D4;
- elegir un checkpoint ganador;
- cambiar 9/12 después de resultados.

## 10. Frontera de ejecución

Este freeze no instala runner, no genera snapshots, no crea START_REQUEST y no autoriza ejecución. El orden posterior será: implementación de gates temporales → QA → 12 snapshots byte-frozen → doble verificación → runner/preflight → autorización separada → ejecución científica.

B1-E permanece cerrado y no existe namespace `final`.
