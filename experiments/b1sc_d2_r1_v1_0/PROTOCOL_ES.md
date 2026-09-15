# B1-SC-D2-R1 v1.0 — Inicialización binaria congelada

**Estado: INIT_BYTES_FROZEN_QA_PENDING — NO SCIENTIFIC EXECUTION.**

Esta rama corrige prospectivamente el problema de reproducibilidad observado en B1-SC-D2 run técnico #2 sin modificar retrospectivamente D2 ni sus resultados.

## Problema aislado

D2 run `34980494295` completó 16/16 fits, pero la agregación detectó `Paired initialization mismatch`: usar la misma `initial_seed` no garantizó pesos bit-a-bit iguales entre runners independientes cuando `RoutingActor` y `Value` se inicializaron mediante `torch.manual_seed()` + `nn.init.orthogonal_()`.

## Corrección D2-R1

Se generaron una sola vez ocho estados iniciales canónicos —uno por bloque— bajo Python 3.11.16, torch 2.2.2+cpu y numpy 1.26.4. El run de generación QA es `35007729637`; su artefacto canónico es `10412775720`, `b1sc-d2-r1-init-candidates`, digest `sha256:0f3a2ac9fe3a29cbf2764257527735be000e487db1462dc3c555ef4bb34580a4`.

Cada estado contiene el `state_dict` completo de `RoutingActor("100110")` y `Value()` en un formato binario canónico. `INIT_FREEZE.json` fija para cada bloque:

- SHA-256 de los bytes del bloque;
- `actor_digest`;
- `critic_digest`;
- seed fuente solo como linaje.

## Contrato obligatorio para un futuro runner R1

1. Descargar exclusivamente el artefacto canónico fijado arriba.
2. Verificar el SHA-256 interno del bloque antes de cargarlo.
3. Construir actor/critic y sobrescribir **todos** sus tensores con los bytes congelados.
4. Verificar `actor_digest` y `critic_digest` después de la carga.
5. Tanto `S6-ON` como `S6-OFF-TRAIN` del mismo bloque deben consumir exactamente el mismo bloque binario.
6. Está prohibido reconstruir el estado inicial científico desde `torch.manual_seed()` o volver a ejecutar `orthogonal_()` como fuente de pesos.
7. La condición ON/OFF no participa en la selección del snapshot.
8. La QA usa dos runners independientes (ON y OFF) y exige igualdad exacta por bloque de SHA, actor, critic y forward digest.

Los snapshots D2-R1 son un nuevo freeze prospectivo; **no se afirma que sean los pesos históricos exactos de D2 run 1**.

## Frontera

`scientific_training_authorized=false`

`scientific_evaluation_authorized=false`

`B1E_disposition=ON_HOLD_FOR_CURRENT_STRUCTURAL_CONFIRMATION`

`B1E_executed=false`

`H_CAT=NOT_EVALUABLE`

`H_TRANSFER=NOT_EVALUATED`
