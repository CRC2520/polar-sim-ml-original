# B1-SC-D4 v1.0 — Adjudicación primaria

Run científico `35173828866`, run number `2`, attempt `1`, commit `b74b37d19717470b35e827363e1a314d1fc1aa19`.

La integridad cerró en **PASS** con 36/36 fits, 12/12 bloques, 12/12 paneles causales LATE, 12/12 paneles causales ALWAYS y el checkpoint `786432` verificado como pre-switch. El primer run operativo `35173090984` falló antes del entrenamiento científico y queda excluido; no se usó GitHub Re-run.

Los gates se adjudican exactamente con los márgenes congelados (`loss > 130/30`, `displacement = 1.0`, replicación `>=9/12`) y sin exclusión de bloques.

## H_D4_LATE_TOTAL

`LOCAL-S6-LATE configured` vs `LOCAL-S6-OFF configured`:

- agregado: `loss_advantage=2.342026451610243`, `displacement_advantage=-0.4123025300602121` → **no cumple ventaja práctica**;
- bloques con ventaja práctica: **3/12**;
- LATE competente frente a no-action: **3/12**.

**H_D4_LATE_TOTAL=false.**

## H_D4_EARLY_PREPARATION

`LOCAL-S6-ALWAYS configured` vs `LOCAL-S6-LATE configured`:

- agregado: `loss_advantage=14.88967032223226`, `displacement_advantage=0.1727302012344203` → cumple ventaja práctica agregada;
- bloques con ventaja práctica: **4/12**;
- ALWAYS competente frente a no-action: **3/12**.

**H_D4_EARLY_PREPARATION=false.**

## Modificadores mecanísticos

### H_D4_LATE_SCAFFOLD

LATE con lesión permanente vs OFF configured: ventaja práctica agregada (`loss_advantage=10.064898090491912`, `displacement_advantage=-0.24981025916834687`), pero solo **3/12** bloques y **5/12** bloques competentes.

**H_D4_LATE_SCAFFOLD=false.**

### H_D4_ONLINE_LATE

LATE intact vs lesión permanente: `loss_harm=-9.558163857390145`, `displacement_harm=-0.029024793455997955`; no hay deterioro práctico agregado, solo **2/12** bloques muestran deterioro práctico. Sham e inactive-edge son exactos en **12/12**.

**H_D4_ONLINE_LATE=false.**

### H_D4_ONLINE_ALWAYS

ALWAYS intact vs lesión permanente: deterioro práctico agregado (`loss_harm=7.547670512007468`, `displacement_harm=1.150025966266791`), con controles exactos **12/12**, pero solo **5/12** bloques alcanzan deterioro práctico; el gate exige `>=9/12`.

**H_D4_ONLINE_ALWAYS=false.**

## Adjudicación

`H_D4_LATE_TOTAL=false` y `H_D4_EARLY_PREPARATION=false` producen, por la tabla congelada:

**`NO_REPRODUCIBLE_ROUTE_TIMING_EXPLANATION`**

D4 no encuentra evidencia reproducible de que una ventana S6 tardía fija explique la heterogeneidad entre bloques observada en D3, ni de que la exposición temprana produzca un beneficio de preparación reproducible. Los modificadores preespecificados tampoco alcanzan 9/12. Esto no demuestra ausencia universal de utilidad de rutas; rechaza únicamente la explicación temporal prospectivamente especificada bajo este diseño.

B1-E permanece cerrado: `B1E_executed=false`, `final_seeds_generated=false`, `H_CAT=NOT_EVALUABLE`, `H_TRANSFER=NOT_EVALUATED`.
