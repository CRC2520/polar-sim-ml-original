# B1-SC-D3 v1.0 — Partición de información y necesidad estructural

**Estado: DESIGN_FROZEN_NOT_EXECUTED.**

Este documento congela un experimento prospectivo de desarrollo que sigue a B1-SC-D2-R1. No reinterpreta ni reejecuta B1-S, B1-SC, D2, D2-R1 ni B1-E. No autoriza entrenamiento científico, no crea semillas finales B1-E y no convierte éxito técnico en evidencia científica.

## 1. Motivación

D2-R1 cerró con integridad aprobada, `H_TRAIN_D2_R1=false`, `H_ONLINE_D2_R1=false` y adjudicación `NO_REPRODUCIBLE_S6_MECHANISM_UNDER_D2`.

Una limitación arquitectónica permanece: en B1-S/B1-SC/D2 cada nodo del `RoutingActor` recibe la misma observación global raw de 93 dimensiones. El soporte `G0=000000` puede, por tanto, producir acciones competentes sin rutas explícitas porque cada nodo ya dispone directamente de toda la información global.

D3 cambia una sola dimensión conceptual: **la disponibilidad de información para el actor**. La dinámica nativa, espacio de acción, recompensa, familia de política, topologías G0/S6, presupuesto y métricas históricas permanecen fijados.

## 2. Pregunta principal

> ¿Las rutas explícitas S6 adquieren utilidad y dependencia causal reproducibles cuando la información de ejecución se distribuye entre los tres nodos y deja de estar globalmente disponible para cada nodo?

D3 no prueba todavía una polaridad semántica del manuscrito. Prueba una condición previa más estrecha: necesidad estructural de comunicación explícita bajo información distribuida.

## 3. Observación nativa y partición

`NativeEnv.flatten()` produce 93 dimensiones como concatenación ordenada de tres observaciones nativas de 31 dimensiones:

- nodo 0 / `walker_0`: índices `[0:31)`;
- nodo 1 / `walker_1`: índices `[31:62)`;
- nodo 2 / `walker_2`: índices `[62:93)`.

Se congelan dos modos:

### GLOBAL_SHARED

Cada nodo recibe el vector raw completo de 93 dimensiones, igual que en B1-S/B1-SC/D2.

### LOCAL_PARTITIONED

Cada nodo recibe un vector de 93 dimensiones con **solo sus 31 dimensiones locales conservadas** y las otras 62 dimensiones puestas exactamente a cero.

Esto mantiene `E: Linear(96,16)` y, por tanto, el mismo número de parámetros, las mismas formas tensoriales y la posibilidad de inicialización trainable emparejada. El `one-hot` de identidad de nodo permanece sin cambios.

No existe canal lateral de observación global hacia el actor. El crítico permanece centralizado con las 93 dimensiones globales para aislar la manipulación en la información de ejecución del actor.

## 4. Soportes

Se usan únicamente dos soportes preespecificados:

- `G0 = 000000`;
- `S6 = 100110`.

Orden de aristas:

`(0→1, 0→2, 1→0, 1→2, 2→0, 2→1)`.

S6 activa:

- `0→1`;
- `1→2`;
- `2→0`.

No hay búsqueda de máscaras.

## 5. Diseño factorial

Cuatro condiciones:

1. `GLOBAL-G0`;
2. `GLOBAL-S6`;
3. `LOCAL-G0`;
4. `LOCAL-S6`.

Cada condición se ejecutará en 8 bloques emparejados.

Total planificado:

- 4 condiciones;
- 8 bloques;
- 32 fits;
- 1.048.576 transiciones nativas por fit;
- 33.554.432 transiciones nativas totales.

No hay búsqueda de hiperparámetros, selección adaptativa de checkpoint, early stopping por desempeño ni normalización online.

## 6. Inicialización prospectiva

Antes de una futura ejecución científica se deberán generar y versionar **ocho snapshots canónicos de parámetros trainables**, uno por bloque.

Dentro de cada bloque, las cuatro condiciones compartirán exactamente los mismos bytes de parámetros trainables de:

- `E`;
- `A`;
- `B`;
- `D`;
- `log_std`;
- crítico `Value`.

El buffer no entrenable `support` se deriva de la condición (`G0` o `S6`) y no forma parte del snapshot trainable emparejado.

La reconstrucción por seed durante el run científico estará prohibida. El optimizador no podrá crearse ni ejecutar el primer paso antes de validar el snapshot.

## 7. H_INFO_UTILITY — utilidad estructural bajo partición

Contraste primario:

`LOCAL-S6` vs `LOCAL-G0`.

Estimandos:

`loss_advantage_local = L_LOCAL_G0 - L_LOCAL_S6`

`displacement_advantage_local = D_LOCAL_S6 - D_LOCAL_G0`

Ventaja práctica si:

- `loss_advantage_local > 130/30` y `displacement_advantage_local >= -1`; o
- `displacement_advantage_local > 1` y `loss_advantage_local >= -(130/30)`.

`H_INFO_UTILITY` pasa solo si:

1. el agregado de los 8 bloques cumple ventaja práctica;
2. al menos 6/8 bloques cumplen ventaja práctica;
3. `LOCAL-S6` es competente frente al no-action witness en al menos 6/8 bloques.

La unidad primaria de replicación es el bloque.

## 8. H_INFO_CAUSAL — dependencia online bajo partición

Solo sobre los 8 modelos `LOCAL-S6` finales y congelados.

Intervención primaria:

- lesión permanente de las tres rutas activas S6 desde decisión 0 hasta terminal.

Controles:

- intact;
- sham exacto;
- lesión de arista inactiva `0→2` (índice 1).

Estimandos:

`loss_harm_local = L_lesion - L_intact`

`displacement_harm_local = D_intact - D_lesion`

Deterioro práctico usa exactamente los umbrales históricos de D2.

`H_INFO_CAUSAL` pasa solo si:

1. el agregado cumple deterioro práctico;
2. al menos 6/8 bloques cumplen deterioro práctico;
3. sham e inactive-edge control son exactos bajo la tolerancia congelada.

La ventana 9..40 puede conservarse solo como diagnóstico secundario y no reemplaza la lesión permanente primaria.

## 9. Control GLOBAL

`GLOBAL-S6` vs `GLOBAL-G0` se ejecuta con el mismo presupuesto y emparejamiento para contextualizar el cambio respecto de la arquitectura histórica de información compartida.

Este contraste es secundario. D3 **no exige demostrar ausencia de efecto global** para aceptar H_INFO_UTILITY/H_INFO_CAUSAL, porque un resultado nulo no constituye equivalencia.

La diferencia local-vs-global podrá reportarse descriptivamente. Cualquier test de interacción confirmatorio requeriría un protocolo futuro separado o márgenes de equivalencia preespecificados.

## 10. Competencia

Se conserva el witness no-action y los umbrales históricos:

- `loss_improvement > 130/30`;
- `additional_displacement >= 1`.

La competencia no sustituye los contrastes estructurales primarios.

## 11. Evaluación

Por condición y bloque:

- checkpoints diagnósticos: 262.144, 524.288, 786.432, 1.048.576;
- 32 episodios diagnósticos por checkpoint;
- endpoint primario: 64 episodios emparejados;
- panel causal `LOCAL-S6`: 64 episodios por régimen.

La política se evalúa determinísticamente.

## 12. QA anti-fuga de información

Antes de cualquier ejecución científica deben pasar al menos estas propiedades:

1. `GLOBAL_SHARED` replica el contrato histórico de observación compartida;
2. `LOCAL_PARTITIONED` conserva exactamente 31 dimensiones por nodo y pone a cero las otras 62;
3. las formas de `E`, `A`, `B`, `D` y el número de parámetros trainables son idénticos entre modos global/local para un mismo soporte;
4. con `LOCAL-G0`, la salida de acción de un nodo es invariante a perturbaciones aplicadas exclusivamente a observaciones de otros nodos;
5. con `LOCAL-S6`, los estados pre-mensaje `h_i` dependen solo de la observación local de `i`; cualquier influencia remota debe atravesar rutas explícitas;
6. el crítico continúa recibiendo las 93 dimensiones globales y no se reutiliza como canal hacia el actor;
7. G0 no recibe mensajes;
8. las lesiones solo afectan las rutas declaradas;
9. los namespaces de seeds están separados;
10. ninguna QA o microfit cuenta como evidencia científica.

## 13. Semillas de desarrollo

Root D3:

`a01253416f7e869e98b94dd321607b3826eebadb395048a580c652f1a22b62bf`

Splits permitidos:

- `training`;
- `diagnostic`;
- `endpoint`;
- `causal`;
- `sham`;
- `bootstrap`;
- `qa`.

Las identidades emparejadas por bloque excluyen la condición cuando los valores deben ser iguales entre las cuatro condiciones.

No existe namespace `final` ni B1-E.

## 14. Adjudicación D3

Con integridad aprobada:

- `H_INFO_UTILITY=true`, `H_INFO_CAUSAL=true` → `PARTITIONED_INFORMATION_STRUCTURAL_DEPENDENCE_SUPPORTED`;
- `true, false` → `PARTITIONED_TRAINING_UTILITY_WITHOUT_ONLINE_DEPENDENCE`;
- `false, true` → `PARTITIONED_ONLINE_DEPENDENCE_WITHOUT_TRAINING_UTILITY`;
- `false, false` → `NO_REPRODUCIBLE_PARTITION_INDUCED_S6_NECESSITY`.

Si falla integridad: `INVALID_OR_INCONCLUSIVE_D3_ADJUDICATION`.

Ninguna adjudicación D3 autoriza automáticamente B1-E.

## 15. Frontera B1-E

Permanece:

- `B1E_disposition=ON_HOLD_FOR_CURRENT_STRUCTURAL_CONFIRMATION`;
- `ready_for_b1e_protocol_design=false`;
- `ready_for_b1e_freeze=false`;
- `ready_for_b1e_confirmatory_run=false`;
- `B1E_executed=false`;
- `final_seeds_generated=false`;
- `H_CAT=NOT_EVALUABLE`;
- `H_TRANSFER=NOT_EVALUATED`.

## 16. Frontera de ejecución

Este freeze de diseño no instala un runner científico ni autoriza entrenamiento.

Antes de ejecutar D3 se requieren, en orden:

1. implementación del actor particionado;
2. QA anti-fuga;
3. generación byte-frozen de ocho snapshots trainables;
4. persistencia y doble verificación independiente de snapshots;
5. full preflight;
6. nuevo `START_REQUEST.json` inmutable autorizado explícitamente.
