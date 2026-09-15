# B1-SC v1.0 — Competencia estructural y rutas causales

**Estado: DESIGN_FROZEN_NOT_EXECUTED.**

Este documento congela un nuevo estudio de desarrollo estructural. No reejecuta B1-S v1.1, no modifica CP-NB y no es B1-E. Ningún workflow de B1-SC, solicitud de inicio o entrenamiento forma parte de este freeze.

## 1. Pregunta

Después de CP-NB, la pregunta deja de ser si la tarea puede aprenderse con referencias pertinentes. El nuevo contraste pregunta si, entre controladores entrenados con una receta competente, una topología explícita de rutas aporta:

1. competencia reproducible;
2. dependencia causal externa de sus rutas;
3. utilidad práctica frente a un RoutingActor competente sin rutas explícitas (`G0=000000`).

Competencia, dependencia causal y utilidad estructural son afirmaciones separadas.

## 2. Hipótesis

**H_REL-SC.** Al menos una estructura relacional explícita y competente contiene rutas cuya lesión controlada produce un déficit externo reproducible y cuya organización aporta utilidad práctica frente a `G0`, manteniendo información, familia de actor y presupuesto comparables donde corresponda.

La alternativa sustantiva admite varias posibilidades: estructuras no competentes; rutas computacionalmente usadas pero externamente redundantes; dependencia de rutas sin ventaja frente a G0; o varias estructuras funcionalmente equivalentes sin ganador único.

No se evalúa H_CAT ni H_TRANSFER. No se asignan significados filosóficos a los nodos del entorno.

## 3. Panel congelado

Orden de aristas de `RoutingActor`: `(0→1, 0→2, 1→0, 1→2, 2→0, 2→1)`.

| Rol | Máscara | Interpretación experimental |
|---|---|---|
| G0 | `000000` | referencia principal competente sin rutas |
| GD | `111111` | referencia densa |
| S1 | `000001` | ruta histórica `2→1` |
| S2 | `000100` | dirección inversa `1→2` |
| S3 | `000101` | reciprocidad `1↔2` |
| S4 | `100100` | cadena `0→1→2` |
| S5 | `011001` | ciclo `0→2→1→0` |
| S6 | `100110` | ciclo inverso `0→1→2→0` |
| PPO | — | comparador genérico competente |

SAC no forma parte del panel porque CP-NB no alcanzó competencia en ninguna de sus condiciones ensayadas. Esta exclusión es prospectiva para B1-SC y no afirma incapacidad general de SAC.

## 4. Entrenamiento

- entrada: `raw`;
- presupuesto único: **1.048.576 transiciones nativas por ajuste**;
- **8 bloques independientes** por rol;
- **72 entrenamientos** en total;
- sin normalización de observaciones ni recompensa;
- sin búsqueda adicional de hiperparámetros;
- sin selección de checkpoint;
- sin parada anticipada por desempeño.

Los RoutingActor mantienen la familia y perfiles fijados en la base científica. PPO mantiene su perfil genérico. Las semillas pertenecen a un namespace de desarrollo B1-SC nuevo y están comprometidas antes de ejecutar.

## 5. Gate de competencia

Cada bloque se evalúa sobre 64 episodios de admisión y un `no-action` emparejado. La unidad de replicación primaria es el entrenamiento independiente; los episodios están anidados.

Se conserva la conjunción:

`loss_improvement = L_no-action - L_model > 130/30`

y

`additional_displacement = D_model - D_no-action >= 1`.

Un rol se admite a causalidad únicamente si:

1. el agregado de los ocho bloques supera ambos umbrales; y
2. **al menos 6 de 8 bloques** superan individualmente la conjunción.

La regla 6/8 es nueva para B1-SC y no reinterpreta CP-NB ni B1-S v1.1.

## 6. Contraste causal primario

Las políticas admitidas se congelan. No hay reentrenamiento durante causalidad.

Sobre un split causal separado, con 64 episodios emparejados por bloque, se compara la política intacta con una lesión de **todas las rutas activas** durante las decisiones **9 a 40 inclusive**, usando índices base cero. Son 32 decisiones consecutivas.

Definiciones:

`loss_harm = L_lesion - L_intact`

`displacement_harm = D_intact - D_lesion`.

Hay deterioro práctico si se cumple una de estas condiciones y la otra métrica no mejora más allá de su margen:

- `loss_harm > 130/30` y `displacement_harm >= -1`; o
- `displacement_harm > 1` y `loss_harm >= -(130/30)`.

El gate causal exige deterioro práctico agregado y deterioro práctico en **>=6/8 bloques**.

El cambio local de la acción en la decisión 9 se conserva como evidencia mecanística auxiliar; no sustituye el efecto externo.

## 7. Localización secundaria

Solo si una estructura supera el gate causal primario se lesionan individualmente sus aristas activas, usando la misma ventana y episodios emparejados.

Los contrastes de interacción entre dos rutas son secundarios y jerárquicos. Una no aditividad no se denomina polar por sí misma.

Controles:

- intacto;
- sham;
- lesión de una arista inactiva para soportes dispersos;
- G0 como control negativo del operador: lesionar sus seis rutas no disponibles debe ser indistinguible del intacto dentro de tolerancia numérica.

## 8. Utilidad estructural

G0 es la referencia primaria de utilidad. Solo se compara utilidad si candidato y G0 superan el gate de competencia.

`loss_advantage = L_G0 - L_candidate`

`displacement_advantage = D_candidate - D_G0`.

Existe ventaja práctica de bloque si:

- `loss_advantage > 130/30` y `displacement_advantage >= -1`; o
- `displacement_advantage > 1` y `loss_advantage >= -(130/30)`.

El gate de utilidad exige la regla agregada y **>=6/8 bloques**. GD y PPO son referencias secundarias; no sustituyen G0.

No se fuerza una estructura ganadora. Si varias cumplen los criterios, se reporta el conjunto o clase.

## 9. Análisis

- unidad primaria: bloque de entrenamiento;
- efectos por episodio se agregan primero dentro de cada bloque;
- bootstrap percentil al 95% con 10.000 remuestreos de bloques, descriptivo;
- no se usa `p<0.05` como criterio primario;
- lesiones individuales se interpretan solo después del gate causal global;
- todos los signos adversos y resultados mixtos se conservan.

Estados posibles: `NOT_COMPETENT`, `COMPETENT_RELATIONALLY_REDUNDANT`, `CAUSALLY_DEPENDENT_WITHOUT_STRUCTURAL_UTILITY`, `COMPETENT_CAUSAL_STRUCTURE`, `INCONCLUSIVE_OR_MIXED`.

## 10. Límites

Un resultado favorable no valida el catálogo histórico, transferencia, consciencia, ASI ni una ley física. Un grafo útil puede ser organización relacional genérica. El estudio no autoriza B1-E automáticamente.

`B1E_disposition=ON_HOLD_FOR_CURRENT_STRUCTURAL_CONFIRMATION`

`ready_for_b1e_protocol_design=false`

`ready_for_b1e_freeze=false`

`ready_for_b1e_confirmatory_run=false`

`B1E_executed=false`

`final_seeds_generated=false`

`H_CAT=NOT_EVALUABLE`

`H_TRANSFER=NOT_EVALUATED`
