# B1-SC-D2-R1 v1.0 — Freeze binario de inicialización

**Estado: INIT_FREEZE_GENERATION_ONLY — NO SCIENTIFIC EXECUTION.**

Esta rama corrige prospectivamente el problema de reproducibilidad observado en el run técnico D2 #2 sin modificar retrospectivamente B1-SC-D2 v1.0 ni sus resultados.

## Objetivo

Eliminar la dependencia de `torch.manual_seed()` + `nn.init.orthogonal_()` para garantizar el emparejamiento ON/OFF entre runners distintos.

Para cada bloque `b=0..7` se generará una sola vez un snapshot binario canónico que contiene el estado inicial completo de:

- `RoutingActor("100110")`;
- `Value()`.

Los dos tratamientos futuros del mismo bloque (`S6-ON` y `S6-OFF-TRAIN`) deberán cargar exactamente el mismo archivo antes del primer paso de optimización.

## Reglas congeladas

1. Ocho snapshots, uno por bloque.
2. El snapshot se genera una sola vez bajo el runtime preservado de D2: Python 3.11.16, torch 2.2.2+cpu, numpy 1.26.4.
3. La semilla fuente de cada bloque es la `initial_seed` ya definida por el registro D2; la generación del snapshot no constituye evidencia científica.
4. El formato binario es propio y determinista: cabecera JSON canónica + bytes `float32`/enteros contiguos de todos los tensores, en orden lexicográfico.
5. Cada snapshot queda identificado por SHA-256 de archivo, `actor_digest` y `critic_digest`.
6. El entrenamiento futuro no puede reconstruir pesos desde la seed: debe cargar los bytes congelados y fallar si el SHA-256 no coincide.
7. ON y OFF del mismo bloque deben verificar igualdad exacta de `actor_digest`, `critic_digest` e `initial_state_sha256` antes del primer `env.step`.
8. La identidad de tratamiento no participa en la elección del snapshot.
9. Los snapshots D2-R1 no se interpretan como los pesos históricos exactos usados en D2 run 1; son un nuevo freeze prospectivo para reproducibilidad técnica.
10. No existe `START_REQUEST.json`, no hay autorización de entrenamiento científico y B1-E permanece sin cambios.

## Linaje

- B1-SC-D2 run primario: `34980477849` — `ONLINE_DEPENDENCE_ONLY_SUPPORTED`.
- B1-SC-D2 run técnico #2: `34980494295` — 16/16 fits completados pero agregación bloqueada por `Paired initialization mismatch`.
- Base de esta rama: `3b65403c309d4fe3ef01be4c1fd2693c48f02702`, estado de execution-readiness D2 previo al `START_REQUEST`.

## Frontera

`scientific_training_authorized=false`

`scientific_evaluation_authorized=false`

`B1E_disposition=ON_HOLD_FOR_CURRENT_STRUCTURAL_CONFIRMATION`

`B1E_executed=false`

`final_seeds_generated=false`
