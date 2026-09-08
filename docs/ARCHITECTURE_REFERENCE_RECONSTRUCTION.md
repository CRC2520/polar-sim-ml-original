# Documento de referencia obligatorio — Arquitectura Polar Dynamics

**Reconstrucción v0.1**  
**Fecha de corte:** 7 de septiembre de 2026  
**Estatus:** `REF_out`; este documento es una salida nueva de reconstrucción y no es fuente histórica de sí mismo.

## 1. Mapa arquitectónico conservado

La arquitectura histórica permanece como **programa propuesto de largo plazo**, no como descripción de un sistema completo ya ejecutado:

1. **Tension Engine** — dinámica nodal/relacional de la propuesta.
2. **Integración Consciente `C_t`** — disponibilidad/broadcast de contenido e intenciones según la propuesta original.
3. **Auto-Regulación Ética `E_t`** — escalado, checks y overrides propuestos.
4. **Inconsciente Polar `U_t`** — trazas latentes, gates, release/reconsolidación en la propuesta.
5. **Objetivos Motivacionales** — metas, prioridades y compromisos.
6. **Planificación / razonamiento** — interfaces a planificación.
7. **IACL** — coordinación interagente propuesta.
8. **Acción / feedback y adaptadores especializados** — interfaz con entorno y módulos externos.

Conservar el mapa no atribuye implementación o utilidad a cada capa. Las contrapartes de CODE se clasifican por correspondencia y evidencia.

## 2. Genealogía de fuentes

La regla documental es:

```text
O -> [R_src1 -> ... -> R_srcn] -> CODE
```

Solo se inserta un `R_src` cuando existe un documento fuente concreto, versionado y cronológicamente pertinente. No se inventa una revisión para completar la cadena.

### 2.1 Fuentes identificadas

| Identificador | Fuente | Rol | Estado_fuente |
|---|---|---|---|
| `O` | `Polar_Dynamics_of_Consciousness_arXiv_v2.0.pdf`, SHA-256 `e124988b78e6708d753d9c80775f8c7589253a0554e35994921caa4cfd8bf20e` | propuesta histórica | `ok` para PDF; commit Git `no_determinado` |
| `CODE` | `CRC2520/polar-sim-ml-original@05caf7abef6fdcb769604539a22cdcd1cfd6b397` | mecanismos y evidencia ejecutada | `ok` |
| `M` | `CRC2520/POLAR_MODEL_CRC@b23507cf80ca79ca0ac16e4b36ca9f5a535ff1e5` | manuscrito integrado posterior a CODE | `ok`; no se inserta retroactivamente como revisión pre-CODE |
| `REF_out` | este documento | reconstrucción actual | salida, no fuente de sí mismo |

## 3. Diferencias de formulación que no deben colapsarse

- `O` usa activación firmada `A_i`; realizaciones posteriores usan, entre otras representaciones, dos canales `(p_i+,p_i-)`. Dos canales no son por nombre la formulación original.
- En `O`, la Ec. (1) corresponde a la actualización nodal; la Ec. (2) al baseline adaptativo `B_i`; las Ecs. (3)–(4) describen trazas/release de `U`. Numeraciones posteriores no se atribuyen a `O`.
- Un intercepto posterior `b_i`, una ganancia aprendida o `B_hat` no se identifican automáticamente con el `B_i` original.
- `W`, `M(C_t)`, `B_hat`, `Pi`, factibilidad, proyección y dinámica ambiental son objetos distintos.
- `K_effective` y `W_effective` del planner integrado son sensibilidades/diagnósticos derivados del objetivo y no se renombran como `W` y `M(C_t)` originales.

## 4. Escalas de correspondencia y evidencia

### Correspondencia documental

`conserva | reformula | sustituye | introduce | no_determinado`

### Transición final hacia CODE

`realiza | aproxima | sustituye | introduce_mecanismo_nuevo | sin_contraparte_ejecutada | no_determinado`

`introduce_mecanismo_nuevo` describe ingeniería sin antecedente documental identificado. No equivale a una hipótesis científica nueva por sí solo.

### Cuatro casillas de evidencia

Toda correspondencia se evalúa por separado en:

1. **Implementación** — ¿existe y ejecuta el mecanismo?
2. **Función causal** — ¿una intervención cambia la función definida?
3. **Utilidad** — ¿mejora una métrica externa bajo comparadores adecuados?
4. **Generalización** — ¿se sostiene fuera de la tarea/familia evaluada?

Una casilla positiva no completa las otras.

## 5. Tabla de fidelidad

| Componente | Correspondencia hacia CODE | Implementación | Función causal | Utilidad | Generalización / límite |
|---|---|---|---|---|---|
| **Activación nodal** | `aproxima` | Existen representaciones duales y controles signed+intensity. | El estado entra en decisiones de versiones que lo usan. | No se atribuye ventaja a la coordenada; la recodificación invertible funciona como control de equivalencia. | No valida una ventaja polar distintiva. |
| **Baseline/adaptación** | identidad con `B_i`: `no_determinado` | S1 usa gain/capability; P0–P2 usa `B_hat` RLS físico. | Estimadores son lesionables. | Mixta: S1 muestra trade-offs; P1 no alcanza el criterio de predicción acoplada. | Misma/s familias sintéticas; no fundir estimadores de versiones distintas. |
| **`W + M(C_t)` original** | S3: `aproxima` estructura configurada; P0–P2: `sustituye` la regla de decisión por predictor+MPC. | Ambas realizaciones existen. | S3 activo altera conducta; P1 cross-effects son lesionables. | S3 activo adverso; P1 coupled favorable en media pero no pasa margen; S2 cross-effects sí mejoran vs lesión, mientras shuffled/dense superan la organización nominada. | No identificar una realización con la Ec. (1) completa de `O`. |
| **`K_effective/W_effective`** | `introduce_mecanismo_nuevo` como campos diagnósticos | Sí, como traza del planner. | No evaluados como módulos autónomos. | `no_determinado` como módulos. | No son `W/M` independientes. |
| **`C_t` / acceso de contenido** | allocator S1: identidad `no_determinado`; routing posterior: `aproxima` | Resource allocator y rutas mínimas existen. | Hay cortes/intervenciones delimitadas. | Función local; no evidencia de workspace neuronal/consciencia. | Competencia de contenidos y consumidores heterogéneos siguen abiertas. |
| **`U_t` / memoria** | `aproxima` retención/recuperación; parte de la recurrencia original es sustituida por registros discretos | Storage, cue retrieval, gates, erase y reconsolidación versionada. | Sí en ensayos delimitados. | Memoria cue-addressed respaldada en S1/P1. | No autobiografía, trauma, sueños o consciencia; retrieval, erase y goal cache son operaciones distintas. |
| **`E_t` / regulación** | `aproxima` enforcement acotado | Permisos, prohibiciones, precedencia, mínimos, razones, halt/review. | Reglas cambian acciones. | Compliance de demostración; policy de review P2 es adversa para tracking. | No moralidad general, autenticación o safety física global. |
| **Objetivos motivacionales** | `aproxima` | Persistencia de metas suministradas y prioridades. | Prioridad cambia decisiones. | Prioridad pasa P1. | No motivación endógena ni valores intrínsecos. |
| **Planificación** | `aproxima/realiza` función numérica según versión | S2 usa horizonte finito real; P0–P2 MPC restringido. | Horizonte/cross-effects son lesionables. | S2 cross-effects favorables vs lesión; P1 horizonte no supera criterio. | `horizon` de S1 no es horizonte de planificación. |
| **IACL** | `aproxima` | P2 tiene controladores locales, ofertas escalares y grants acotados. | Las ofertas afectan asignación. | Mejora vs reparto igualitario en P2. | No negociación rica, autonomía social general o consciencia colectiva. |
| **Adaptadores/percepción** | `aproxima` | Gramática operacional tank/fill/drain y canales tipados. | Efectos físicos verificables. | Útiles para la tarea local. | No lenguaje/visión general ni validación de las ocho etiquetas filosóficas. |
| **Self-model: predicción** | identidad con un único módulo original: `no_determinado` | Gain estimator/RLS/`B_hat` según versión. | Sí, lesionable. | Mixta. | No fundir mecanismos de versiones diferentes. |
| **Self-model: procedencia** | `introduce_mecanismo_nuevo` | P2 infiere entre candidatos suministrados y conserva tokens de feedback. | Capacidad acotada: 278/288; abstención ante ambigüedad idéntica. | Utilidad en loop primario `no_determinada`. | No autenticación ni discovery abierto. |
| **Self-model: fiabilidad** | `introduce_mecanismo_nuevo` | Monitor probabilístico P2. | Produce forecast pre-feedback evaluable. | Calibración aceptada en el dominio. | No implica self-awareness ni política útil. |
| **Self-model: policy ante fiabilidad** | `introduce_mecanismo_nuevo` | Review experimental P2. | Cambia acciones. | Adversa: empeora tracking. | No fallback seguro general. |

## 6. G01–G14 son brechas, no capas

| Gap | Estado actual | Permanece abierto |
|---|---|---|
| **G01** | estado interno, intención y acción separados e intervenibles | necesidad/ventaja general de continuidad interna |
| **G02** | información firmada diagnosticada en P0 | separar subcambios del factor compuesto |
| **G03** | rutas diferenciadas/lesionables | utilidad de una organización relacional específica |
| **G04** | se aprenden coeficientes acción–efecto | qué aristas deben existir y qué significado tiene `Pi` |
| **G05** | predictor acoplado implementado | margen práctico, aislamiento y transferencia |
| **G06** | prioridades implementadas y respaldadas | generalización; no prueba exclusivamente polar |
| **G07** | transmisión mínima + cortes | selección entre contenidos, consumidores independientes y uso diferencial |
| **G08** | memoria operativa con utilidad acotada | ciclo `U` original más amplio; separar erase/retrieval/cache en assays |
| **G09** | compliance de reglas de demostración | razonamiento normativo amplio y garantías formales |
| **G10** | metas persistentes + planificación multietapa | utilidad temporal y motivación endógena |
| **G11** | predicción, procedencia, fiabilidad y policy separadas parcialmente | atribución en loop primario y policy útil ante baja fiabilidad |
| **G12** | coordinación local + ofertas escalares | negociación con contenido y cooperación heterogénea |
| **G13** | semántica operacional de tarea | justificación/grounding de categorías y dominio externo |
| **G14** | ciclo integrado y replay interno | estabilidad, safety, escalabilidad, compute y transferencia |

## 7. Registro de decisiones de reconstrucción

| Decisión | Motivo | Estatus | No-claim |
|---|---|---|---|
| Separar Programa A y Programa B | impedir transferencia de evidencia de control a consciencia | editorial/metodológica | no abandona la arquitectura ni exige dos manuscritos físicos |
| Usar `Pi` como relación de auditoría | distinguir relación, pesos, aprendizaje y factibilidad | editorial | no reescribe la arquitectura histórica |
| Usar `Pi_S2_nom` / `Pi_S3_aligned` | evitar identidad retrospectiva entre organizaciones distintas | editorial | no son nombres históricos certificados |
| Interpretar `tau` como `indice_de_desajuste_y_conflicto_declarado` | conservar simultáneamente `r+h` | editorial | no psicologiza la medida ni determina qué componente domina |
| Separar content routing y resource allocator | evitar identificar `C_t` con presupuesto | editorial | no crea una capa nueva |
| Separar predicción, procedencia, fiabilidad y policy | evitar fundir funciones de self-model | metodológica | ninguna equivale por sí sola a metacognición consciente |
| Conservar HGI/INC solo como hechos históricos | preservar fidelidad a `O` | editorial | no son criterios externos actuales de éxito |

## 8. HGI/INC

El programa histórico utilizó HGI/INC en su propuesta de coordinación. Esta reconstrucción conserva ese hecho, pero **no adopta HGI/INC como criterios suficientes de control externo, ética, consciencia o integración**.

Un resultado favorable en HGI/INC no sustituye tracking, regret, restricciones, transferencia ni un protocolo teórico independiente.

## 9. Jerarquía de fuente por objeto

1. teoría/propuesta original → `O`;
2. revisión documental preexistente → `R_src` versionado, solo cuando sea cronológicamente pertinente;
3. mecanismo ejecutado → CODE + configuración/freeze/manifiesto necesarios;
4. resultado → artefacto congelado + análisis que produjo la cantidad;
5. regla prospectiva → protocolo/freeze anterior a resultados;
6. interpretación posterior → citada como posterior.

Ante fuentes del mismo nivel que discrepen: el campo científico será, por defecto, `no_determinado`; se asignará `Estado_fuente=conflicto_de_fuentes` y se conservarán las fuentes en disputa.

## 10. Alcance

La arquitectura por capas permanece como programa de largo plazo. CODE **realiza, aproxima, sustituye o introduce** mecanismos según cada componente; ninguna correspondencia equivale por sí sola a evidencia de utilidad o consciencia.

La experiencia fenoménica permanece como objetivo teórico separado y no se declara medida por los ensayos de control archivados.
