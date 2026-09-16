# B1-SC-D3 v1.0 — Adjudicación primaria

Run científico `35059178684`, attempt `1`, commit `89894e6259bd361810928d0765624fc11d7722e2`.

La integridad cerró en PASS con 32/32 fits y 8/8 paneles causales LOCAL-S6.

## H_INFO_UTILITY

El agregado LOCAL-S6 vs LOCAL-G0 cumple ventaja práctica (`loss_advantage=16.129236303717537`, `displacement_advantage=0.9688519164919853`), pero solo **4/8** bloques cumplen la regla práctica y LOCAL-S6 es competente frente a no-action solo en **2/8** bloques. El gate congelado exige >=6/8 en ambos criterios.

**H_INFO_UTILITY=false.**

## H_INFO_CAUSAL

La lesión permanente muestra deterioro práctico agregado (`loss_harm=8.319530286886845`, `displacement_harm=1.5831875763833523`) y los controles sham/inactive-edge pasan exactamente en 8/8 bloques. Sin embargo, solo **3/8** bloques cumplen deterioro práctico; el gate exige >=6/8.

**H_INFO_CAUSAL=false.**

## Adjudicación

`NO_REPRODUCIBLE_PARTITION_INDUCED_S6_NECESSITY`

El resultado significa que D3 no alcanzó la evidencia reproducible preespecificada para aceptar utilidad estructural S6 ni dependencia causal online bajo partición de información. No implica ausencia universal de utilidad de rutas ni reinterpreta retrospectivamente D2/D2-R1.

B1-E permanece cerrado: `B1E_executed=false`, `final_seeds_generated=false`, `H_CAT=NOT_EVALUABLE`, `H_TRANSFER=NOT_EVALUATED`.
