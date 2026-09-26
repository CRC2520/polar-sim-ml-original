# POLAR P3 — cierre confirmatorio de brechas internas restantes

Fecha: 20/21-sep-2026  
Semillas de desarrollo: 997001–997004  
Stress panel: 997100–997119  
Semillas confirmatorias: **998001–998012**  
Criterio global por brecha: **>=9/12**.

## Resultado

| Brecha P3 | PASS seeds | Veredicto |
|---|---:|---|
| A — Autonomía prolongada / continual learning | **12/12** | **PASS** |
| B — Memoria autobiográfica operacional específica de identidad | **10/12** | **PASS** |
| C — Priorización endógena de metas operacionales | **12/12** | **PASS** |
| D — Bucle individuo → población → ecología | **11/12** | **PASS** |

Las cuatro brechas **internamente testables** cumplen los criterios congelados.

## A — autonomía prolongada

El agente opera 4,800 pasos en ocho bloques y cuatro regímenes recurrentes sin recibir ID de régimen.

Medianas confirmatorias:
- beneficio en reentrada frente a borrar modelos al cambiar contexto: **+0.17731**;
- reward final (últimos dos bloques): **0.93101**;
- retención segunda visita / primera visita: **1.01814**.

Resultado: **12/12 PASS**.

Interpretación: existe reutilización de modelos factuales después de ausencias prolongadas y cambios de contexto dentro de un horizonte acotado. No equivale a autonomía indefinida o de vida completa.

## B — own-history autobiográfico

La identidad oculta del agente persiste entre adquisición y evaluación, pero cambia la experiencia exógena. La memoria se aprende sólo de acciones y recompensas propias. Se compara con control sin memoria y con trasplante de memoria de una identidad complementaria.

Medianas:
- mejora de MAE propia vs stateless: **+0.11641**;
- daño predictivo por memoria extranjera: **+0.35845**;
- daño de reward por memoria extranjera: **+0.40197**.

Resultado: **10/12 PASS**.

Fallos retenidos:
- seed **998006**: mejora predictiva propia 0.04517 < 0.05;
- seed **998011**: mejora predictiva propia 0.01685 < 0.05.

En ambas, el trasplante extranjero sí deteriora fuertemente predicción y acción. El resultado apoya una memoria de historia propia específica de identidad, no un self fenomenal o narrativo.

## C — metas/prioridades endógenas operacionales

El agente no recibe etiqueta de prioridad. Calcula pesos a partir de sus déficits y tendencias internas. Se aplican shocks grandes pero sobrevivibles a recursos diferentes.

Medianas:
- alive fraction: **1.000**;
- acierto de prioridad sobre recurso más deficitario: **0.910**;
- mejora proporcional de latencia de recuperación: **0.74755**;
- reducción de exposición bajo umbral de seguridad: **0.02833**.

Resultado: **12/12 PASS**.

Resultado adverso conservado:
- trade-off mediano de reward total frente al comparador fijo: **−0.06190**.

La prioridad endógena mejora recuperación/seguridad a costa de hacer menos acciones de tarea. Esto demuestra priorización operacional derivada del propio estado; **no** demuestra valores intrínsecos, moralidad emergente ni ética auto-generada.

## D — individuo, población y ecología

Treinta agentes actúan sobre un recurso compartido renovable. Parte de la población adapta restricción desde feedback local; la transmisión poblacional puede copiar estrategias; el recurso común retroalimenta la viabilidad.

Medianas:
- alive full: **1.000**;
- ganancia de recurso full vs sin adaptación local: **+0.05120**;
- ganancia de restraint: **+0.18034**;
- cambio por transmisión: **+0.25637**;
- efecto ecológico de la transmisión: **+0.07759**.

Resultado: **11/12 PASS**.

Fallo retenido:
- seed **998003**: restraint gain = 0.07658 < 0.08, aunque los demás criterios de esa semilla pasan.

La transmisión usa explícitamente un score que incluye el recurso compartido. Por ello el resultado demuestra un bucle multiescala especificado, no instituciones espontáneas ni moral social emergente.

## Reproducción y auditoría

La campaña completa se ejecutó dos veces:
- reproducción exacta: **12/12**;
- auditoría: **10/10 PASS**;
- reconstrucción de los cuatro conteos y del veredicto global: PASS.

Hashes SHA-256 del freeze:
- PREREG_P3.md: `c6326bc21066db7e4c985f2b30c0e2e9955c51b30bb7b4b846367491106ac0b5`
- experiments.py: `47868904f776dfd0d17548f3a8320890fdd6845e57329623051b63e4282d77a5`
- config_dev.py: `edbe7ef17c023640c10624a6bc873153feb544212c3428fe01fe4dd45b8a2e1f`
- config_confirm.py: `d324dae82947228d29cee41574583c54aa4e6640193f0d583377ab8dde3c24da`
- confirm.py: `31a32a89b81e0984d11e10d53489c9f535154ed94d979d9d01bc43ea3f563cff`
- audit.py: `371ccdfd88b1eb1408cfe8c2ca41a68cb2c6d0e009c36fd237eeeef9ef5e208e`

## Estado después de P3

Quedan dos brechas que este mismo programa no puede auto-certificar:
1. **réplica científica independiente** por otro equipo/implementación;
2. **correspondencia neurobiológica real** con mediciones externas.

También sigue sin demostrarse conciencia fenomenal. P0–P3 aportan evidencia de organización y funciones computacionales, no una prueba de experiencia subjetiva.
