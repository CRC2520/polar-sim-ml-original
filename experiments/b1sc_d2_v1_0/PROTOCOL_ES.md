# B1-SC-D2 v1.0 — Andamiaje de aprendizaje vs dependencia online

**Estado: DESIGN_FROZEN_NOT_EXECUTED.**

Este documento congela un nuevo experimento de desarrollo mecanístico derivado de B1-SC v1.0. No modifica, reinterpreta ni reejecuta B1-SC v1.0, CP-NB, B1-S v1.1 ni B1-E. No autoriza entrenamiento, evaluación científica ni creación de semillas finales B1-E.

## 1. Motivación

B1-SC v1.0 mostró un patrón específico en `S6=100110`:

- competencia reproducible en 8/8 bloques;
- cambios locales de acción al lesionar las rutas;
- ausencia de deterioro externo reproducible bajo la lesión primaria preespecificada 9..40;
- ausencia de gate de utilidad estructural reproducible frente a G0.

Este patrón es compatible con al menos dos mecanismos que B1-SC v1.0 no separa:

1. **andamiaje de aprendizaje / inductive bias**: S6 ayuda a encontrar una política competente durante entrenamiento, pero las rutas dejan de ser necesarias para ejecutar la política ya aprendida;
2. **dependencia online no detectada por ventana corta**: las rutas sí son funcionalmente necesarias, pero una lesión de 32 decisiones (9..40) fue insuficiente para revelar el déficit.

B1-SC-D2 separa prospectivamente ambos mecanismos sin buscar nuevas máscaras.

## 2. Preguntas e hipótesis

### H_TRAIN-D2 — efecto causal durante aprendizaje

Con arquitectura, presupuesto, semillas iniciales y semillas ambientales emparejadas, entrenar S6 con sus rutas funcionalmente activas produce una ventaja práctica reproducible respecto de entrenar la misma arquitectura S6 con las rutas funcionalmente apagadas durante todo el aprendizaje.

Condiciones:

- `S6-ON`: máscara `100110`; `lambda_train=1` para las rutas activas;
- `S6-OFF-TRAIN`: misma máscara `100110`, mismos parámetros potenciales e inicialización emparejada; `lambda_train=0` durante todo el aprendizaje y durante su evaluación endpoint.

Estimandos primarios:

`loss_advantage_train = L_OFF - L_ON`

`displacement_advantage_train = D_ON - D_OFF`

Existe ventaja práctica si:

- `loss_advantage_train > 130/30` y `displacement_advantage_train >= -1`; o
- `displacement_advantage_train > 1` y `loss_advantage_train >= -(130/30)`.

H_TRAIN-D2 pasa únicamente si:

1. el agregado de los 8 bloques cumple la regla de ventaja práctica; y
2. al menos **6/8 bloques** cumplen individualmente la regla.

### H_ONLINE-D2 — dependencia durante ejecución

Sobre los modelos `S6-ON` ya entrenados y congelados, eliminar todas las rutas activas **desde la decisión 0 hasta terminal** produce deterioro externo reproducible frente a la misma política intacta.

Estimandos:

`loss_harm_permanent = L_permanent_lesion - L_intact`

`displacement_harm_permanent = D_intact - D_permanent_lesion`

Existe deterioro práctico si:

- `loss_harm_permanent > 130/30` y `displacement_harm_permanent >= -1`; o
- `displacement_harm_permanent > 1` y `loss_harm_permanent >= -(130/30)`.

H_ONLINE-D2 pasa únicamente si:

1. el agregado de los 8 bloques cumple la regla de deterioro práctico; y
2. al menos **6/8 bloques** cumplen individualmente la regla; y
3. los controles preespecificados pasan.

## 3. Arquitectura y soporte

Se usa exclusivamente `RoutingActor` con máscara:

`S6 = 100110`

Orden de aristas congelado:

`(0→1, 0→2, 1→0, 1→2, 2→0, 2→1)`.

Por tanto, las rutas activas S6 son:

- `0→1`;
- `1→2`;
- `2→0`.

No se buscan nuevas máscaras ni se selecciona una topología con base en D2.

La entrada permanece **raw compartida de 93 dimensiones**. B1-SC-D2 no modifica todavía la disponibilidad global de información; la partición/localización de observaciones queda para un experimento posterior separado.

## 4. Entrenamiento

Diseño factorial mínimo:

- 2 condiciones: `S6-ON`, `S6-OFF-TRAIN`;
- 8 bloques independientes emparejados;
- 16 fits totales;
- 1.048.576 transiciones nativas por fit;
- total planificado: **16.777.216 transiciones**;
- input: `raw`;
- sin normalización de observaciones;
- sin normalización de recompensa;
- sin búsqueda de hiperparámetros;
- sin selección de checkpoint;
- sin early stopping por desempeño;
- misma familia PPO/RoutingActor y perfil fijado en la base B1-SC;
- políticas entrenadas desde cero para D2; no se reutilizan pesos finales de B1-SC v1.0.

El emparejamiento exige que, dentro de cada bloque, `S6-ON` y `S6-OFF-TRAIN` compartan:

- inicialización de parámetros;
- semillas de entornos de entrenamiento por slot/contador;
- semillas de muestreo estocástico de política;
- paneles de evaluación.

La identidad de condición queda excluida de las semillas emparejadas que deban ser iguales entre ON y OFF.

## 5. Checkpoints diagnósticos

Se guardan, sin selección ni detención, los checkpoints:

- 262.144;
- 524.288;
- 786.432;
- 1.048.576 transiciones.

En cada checkpoint se evalúan **32 episodios diagnósticos emparejados por bloque y condición** en un split independiente `diagnostic`.

Métricas diagnósticas:

- loss;
- desplazamiento;
- fracción de caída;
- ciclos/longitud de episodio.

Estos checkpoints son secundarios y descriptivos. El único endpoint primario de H_TRAIN-D2 es **1.048.576**.

## 6. Evaluación endpoint del efecto de entrenamiento

A 1.048.576 transiciones se evalúan **64 episodios emparejados por bloque y condición** en un split `endpoint`, con acciones deterministas.

La unidad primaria de replicación es el **bloque de entrenamiento independiente**. Los episodios están anidados y no se tratan como réplicas independientes.

Además se calcula, como diagnóstico secundario, la competencia de cada condición frente a un `no-action witness` emparejado usando los mismos umbrales históricos:

- `loss_improvement > 130/30`;
- `additional_displacement >= 1`.

La competencia individual de ON u OFF no sustituye el contraste causal primario ON-vs-OFF.

## 7. Contraste online permanente

Solo los 8 modelos `S6-ON` del endpoint final entran al contraste online.

Sobre un split causal separado se ejecutan **64 episodios emparejados por bloque** con:

1. `intact`: rutas S6 activas durante todo el episodio;
2. `permanent_all_active_lesion`: las tres rutas activas S6 anuladas desde decisión 0 hasta terminal;
3. `sham`: operador aplicado sin alterar rutas;
4. `inactive_edge_control`: lesión permanente de la arista inactiva preseleccionada `0→2` (índice 1).

Los controles `sham` e `inactive_edge_control` deben ser idénticos al intacto dentro de tolerancia numérica de `1e-8` en la salida determinista del actor para estados idénticos y no mostrar divergencia atribuible al operador.

Las políticas quedan congeladas: no hay reentrenamiento durante causalidad.

## 8. Replicación secundaria de la ventana B1-SC

En el mismo panel causal se incluye:

`window_9_40_all_active_lesion`

que anula las rutas activas solamente para decisiones base-cero 9..40 inclusive.

Este contraste es **secundario** y sirve para comprobar compatibilidad con el null causal de B1-SC v1.0. No puede reemplazar ni redefinir H_ONLINE-D2.

## 9. QA de equivalencia funcional

Antes de cualquier ejecución científica debe probarse:

1. con `lambda=0`, la contribución `r_j` es exactamente cero;
2. `S6-OFF-TRAIN` tiene el mismo forward que un `RoutingActor("000000")` con pesos compartidos en `E`, `B`, `D` y mismo estado, dentro de tolerancia `1e-8`;
3. los gradientes de `E`, `B`, `D` de esa equivalencia coinciden dentro de tolerancia;
4. `A` no recibe gradiente funcional desde la pérdida cuando `lambda=0`;
5. ON y OFF parten de pesos idénticos por bloque;
6. el operador de lesión permanente solo altera las tres rutas activas S6;
7. la lesión inactiva `0→2` no altera el forward;
8. los splits `training`, `diagnostic`, `endpoint`, `causal`, `sham`, `bootstrap`, `qa` son disjuntos por namespace.

Ningún microfit QA constituye evidencia científica.

## 10. Semillas y compromiso prospectivo

Root de desarrollo:

`ae302b8be2005f914c9e1998303f9508c64c600cd140dff324c5d600be034f4f`

Splits permitidos:

- `training`;
- `diagnostic`;
- `endpoint`;
- `causal`;
- `sham`;
- `bootstrap`;
- `qa`.

Derivación congelada:

`seed32 = uint32(first_8_bytes(SHA256(seed_root_sha256 || "|" || split || "|" || canonical_json(identity)))) mod 2^32`

donde `canonical_json` usa UTF-8, claves ordenadas y separadores compactos.

Para semillas emparejadas ON/OFF, `identity` no incluye la condición. Para artefactos o procesos que deban diferir por condición, la condición debe incluirse explícitamente y documentarse.

No existe namespace `final` ni namespace B1-E.

## 11. Análisis

- unidad primaria: bloque independiente;
- efectos de episodios agregados primero dentro de cada bloque;
- bootstrap percentil sobre bloques;
- 10.000 remuestreos;
- IC descriptivo 95%;
- no se usa `p<0.05` como gate primario;
- no se elimina ningún bloque por desempeño;
- no se reemplazan semillas por resultados adversos;
- un fallo técnico se conserva y no autoriza automáticamente una réplica sustitutiva.

## 12. Adjudicación

Se combinan exclusivamente los dos gates primarios:

| H_TRAIN-D2 | H_ONLINE-D2 | Estado |
|---|---|---|
| PASS | FAIL | `TRAINING_SCAFFOLD_SUPPORTED` |
| PASS | PASS | `TRAINING_AND_ONLINE_DEPENDENCE_SUPPORTED` |
| FAIL | PASS | `ONLINE_DEPENDENCE_ONLY_SUPPORTED` |
| FAIL | FAIL | `NO_REPRODUCIBLE_S6_MECHANISM_UNDER_D2` |

Si fallan controles, integridad, emparejamiento o completitud, el estado es:

`INVALID_OR_INCONCLUSIVE_MECHANISTIC_ADJUDICATION`.

Un resultado favorable no demuestra una propiedad polar específica, consciencia, H_CAT, H_TRANSFER ni validez universal de S6.

## 13. Linaje y separación de B1-SC v1.0

El diseño D2 se congela después de observar la evidencia de admisión y causal primario de B1-SC v1.0, run `34934384067`, attempt 1, autorizado por commit `5ae02be4a8d80485e41898a2c4e8b9c466959b76`.

Artefactos de motivación registrados:

- `b1sc-admission`, artifact ID `10394834164`, digest `sha256:8a49fde7819a3a1a71de2401d925f812c9a90172cdb45476718886097b79fc0c`;
- `b1sc-primary-aggregate`, artifact ID `10398070329`, digest `sha256:3b367132c8ff1b3e03ce83c34f4e5183dc3e1a47ba98a2f6aea8261e6714e302`.

El cierre posterior de B1-SC v1.0 no puede modificar retrospectivamente este freeze D2. Los resultados históricos permanecen intactos.

## 14. Frontera de ejecución

Al congelar este diseño:

- `execution_authorized=false`;
- `workflow_installed=false`;
- `start_request_exists=false`;
- `training_started=false`;
- `evaluation_started=false`;
- `results_exist=false`.

La implementación, QA, workflow y ejecución requieren autorizaciones posteriores explícitas.

## 15. B1-E

`B1E_disposition=ON_HOLD_FOR_CURRENT_STRUCTURAL_CONFIRMATION`

`ready_for_b1e_protocol_design=false`

`ready_for_b1e_freeze=false`

`ready_for_b1e_confirmatory_run=false`

`B1E_executed=false`

`final_seeds_generated=false`

`H_CAT=NOT_EVALUABLE`

`H_TRANSFER=NOT_EVALUATED`
