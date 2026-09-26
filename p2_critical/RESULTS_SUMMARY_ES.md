# POLAR P2-Critical — cierre de cuatro brechas operacionales

Fecha: 20/21-sep-2026  
Semillas confirmatorias v2: 996001–996012  
Criterio global por brecha: >=9/12 semillas.

## Resultado

| Brecha | Semillas PASS | Veredicto |
|---|---:|---|
| E1 — POLAR vs GENERIC relacional isomorfo | 12/12 | PASS |
| E2 — Descubrimiento relacional dentro del agente integrado | 11/12 | PASS |
| E3 — Lesiones funcionales discriminantes | 12/12 | PASS |
| E4 — Validación OOD mediante ecuaciones estándar | 12/12 | PASS |

Las cuatro brechas **operacionales** cumplen sus criterios congelados.

## E1 — resultado de equivalencia, no de superioridad exclusiva

Las 12 semillas terminan en la misma resolución:
`EQUIVALENT_WHEN_GENERIC_RECONSTRUCTS_RELATIONS`.

Medianas:
- MSE POLAR: 0.010322
- MSE GENERIC: 0.010322
- MSE RANDOM: 2.823890
- diferencia POLAR−GENERIC: 0
- relaciones verdaderas recuperadas por POLAR: 8/8
- relaciones verdaderas recuperadas por GENERIC: 8/8

La brecha de especificidad se resuelve de forma negativa para una ventaja exclusiva:
un learner genérico iguala a POLAR cuando reconstruye la misma organización relacional efectiva.
La evidencia respalda la organización relacional, no una notación o marca arquitectónica única.

## E2 — relaciones aprendidas dentro del agente cerrado

Resultado: 11/12.

Medianas:
- recuperación pre-cambio: 3/3
- recuperación post-cambio: 3/3
- ventaja adaptativo vs relaciones congeladas: +0.073807 reward
- daño de la lesión sin relaciones: +0.076184 reward

La semilla 996009 se conserva como FAIL:
+0.048145 frente al control congelado, por debajo del umbral +0.05.

Esto demuestra, dentro del benchmark, que el mismo agente puede aprender relaciones
desde consecuencias factuales de sus propias acciones, cambiar su organización después
de una modificación de régimen y usar esa organización en control.

## E3 — firmas funcionales discriminantes

Resultado: 12/12.

Medianas del sistema intacto:
- primary accuracy: 0.7703
- workspace recall: 0.8030
- source balanced accuracy: 0.6272

Daños medianos:
- lesión relacional → workspace: +0.1251
- lesión relacional → meta-Brier: +0.0246
- lesión relacional → source: +0.2778
- lesión broadcast/workspace → primary: +0.1335
- lesión higher-order → meta-Brier: +0.0962

Las lesiones producen perfiles diferentes en vez de una caída global indiferenciada.

**Límite:** son predicciones funcionales discriminantes. No demuestran experiencia
subjetiva ni prueban que una teoría neurocientífica de conciencia sea correcta.

## E4 — OOD con familias matemáticas estándar

Resultado: 12/12.

Ratio mediano agente/oracle:
- logistic harvesting: 1.00196
- RC thermal: 0.89392
- queue service: 0.98604

La fracción alive mínima fue 1.0 en los tres dominios.

Esto extiende la validación a dinámicas definidas por ecuaciones estándar y no por
variables latentes POLAR. No es una réplica científica realizada por un equipo externo.

## Integridad

La corrida confirmatoria v1, semillas 995001–995012, se retiró por un fallo técnico
de serialización de `numpy.int64` ocurrido después del cálculo y antes de persistir
o mostrar resultados. Ninguna métrica v1 fue inspeccionada. El motor y los umbrales
científicos no cambiaron; v2 utilizó semillas nuevas 996001–996012.

La campaña v2 se ejecutó dos veces:
- reproducción exacta: 12/12
- auditoría: 10/10 PASS

SHA-256 del código/protocolo usado:
- experiments.py: `372fde54943420d4689575d23e629bc0af5fef44edc31b359dda059eac28c87b`
- config_confirm_v2.py: `2cde0f7e240b5a5a9563e178178791c855e6873f1fbd6d9a88753e31e35a5fef`
- PREREG_P2_CRITICAL_V2.md: `200d6fad830c19245a2dbbe3f36d8ef625e47e0e6d8a331caa3c629d081d159e`
- confirm.py: `664a5af7b39c7455991519e063845f7dadb7afd7afa286fb1b0007f1513d3240`
- audit.py: `0b2aef75152f0555b27bf7c04ada10b0e3f9f2941ae3cde4568b4cdc97a60690`

## Decisión científica

Las cuatro brechas planteadas quedan cerradas **a nivel operacional interno**.

Sin embargo, el resultado E1 obliga a una formulación precisa:
POLAR no muestra una ventaja exclusiva frente a un learner genérico que reconstruye
la misma estructura. La hipótesis respaldada es una clase de equivalencia funcional de
organización relacional.

La validación externa verdaderamente independiente sigue pendiente porque requiere
otra implementación/equipo o un benchmark administrado fuera de este programa.
