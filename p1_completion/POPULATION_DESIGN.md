# P1-A/C: cobertura poblacional descriptiva

Versión previa a semillas finales. No sustituye el prerregistro histórico ni
convierte sus gates de potencia/manipulación pendientes en aprobados. No se
ajustan parámetros tras los pilotos. Las simulaciones usan el motor preservado
`collective/bridge_v2/engine.py`; las combinaciones factoriales son una subclase
local que compone interruptores ya existentes sin modificar el motor.

## Diseño y presupuesto

- Grilla de composición inicial: .35, .40, .45, .50, .55, .60, .65, .70; exacta
  con 20 agentes por grupo.
- Dos transmisiones: `success`, `conformity`.
- Configuración: 6 grupos, 20 agentes/grupo, 20 generaciones, 400 pasos por
  generación, 4 generaciones de warmup; ventana de evaluación últimas 5.
- A: `base`, `copy_score_equal`, `resource_pool`: 48 celdas.
- C: ocho variantes originales (`full`, `qv_policy_off`, `qv_learning_off`,
  `short_vitality_discount`, `short_group_discount`, `blind_exploration`,
  `generic_shared_target`, `coordinate_equivalent`) y cuatro combinaciones:
  `anchor_off_short_vitality`, `anchor_off_blind`, `short_vitality_blind`,
  `anchor_off_short_vitality_blind`: 192 celdas.
- Final: 30 semillas pareadas dentro de cada parte y entre todas sus variantes,
  transmisiones y valores de la grilla. Total 7200 corridas poblacionales,
  144000 episodios de generación, 6 912 000 000 pasos-agente potenciales. La
  cantidad de transiciones con agentes vivos es menor y está en los outcomes.
- Los flags factoriales son arquitectónicos: quitar Qv de la política,
  acortar gamma de vitalidad .97→.5 y exploración aleatoria ciega. Qv continúa
  aprendiendo al quitar su contribución a la política. Las cuatro combinaciones
  completan el factorial 2³ junto con full y los tres switches simples.
- Comparaciones factoriales: cada efecto marginal se promedia por igual sobre
  los otros dos factores; interacción triple = 111−110−101−011+100+010+001−000.

## Semillas y desviación de desarrollo

- Piloto A y C: 930001–930003, 3 semillas por celda, todos los parámetros finales.
- Final A: **935001–935030**.
- Final C: **932001–932030**.
- Un benchmark de rendimiento de desarrollo ejecutó solamente la generación 0
  de A/base/p=.50 con 931001–931030. No se inspeccionaron ni conservaron sus
  outcomes. El rango completo se excluyó y se sustituyó antes del análisis final.
  Nunca se afirma que 931001–931030 estuvieran intactas.
- Piloto A se conserva con el hash de fuente anterior al cambio exclusivo de C
  que usa `qt + 0*qv + qg` para evitar cancelación numérica. Replay de A con la
  fuente final reproduce exactamente los 36 elementos evaluados. Piloto C usa
  la fuente final. No se han lanzado semillas finales al redactar este diseño.
- RNG auxiliar de bootstrap: 937003, 937004 y 937005; no son semillas del motor.

## Endpoints y estimación

La unidad de incertidumbre es la semilla, nunca el individuo o el grupo. Se
conservan todos los outcomes y transmisiones por generación. Los endpoints por
semilla resumen las últimas cinco generaciones:

- Supervivencia medida como fracción media del tiempo vivo; recurso medio
  temporal normalizado por capacidad; supervivencia y recurso al final del
  episodio, preservados por separado.
- Frecuencia de restricción (acción≤1), fracción de agentes con frecuencia de
  restricción≥.5, composición genealógica.
- Diferencias dentro de grupo entre clases en fitness bruto, puntaje de copia,
  ingreso físico por paso elegible, ingreso físico acumulado y tiempo vivo.
- En A las clases son linaje1−linaje0. En C son restricción observada≥.5 frente
  a <.5: asociaciones endógenas, no estimaciones causales de un tipo heredado.
- Si un grupo no contiene ambas clases, su diferencia queda missing. El resumen
  conserva contadores de observaciones y semillas; no imputa cero. Los primarios
  de supervivencia, recursos y viabilidad incluyen todas las semillas y extinciones.

Una semilla se declara viable si su media de supervivencia temporal final≥.8 y
su media de recurso temporal final≥.2. Para cada variante/transmisión se estima
la probabilidad empírica de viabilidad sobre la grilla. p* es el primer cruce de
.5 de esa probabilidad: solamente se interpola entre dos puntos adyacentes
observados. Si la curva disminuye en cualquier punto, se declara no identificable
por no monotonía; si ya cruza al inicio, está en o debajo del soporte; si no
cruza al final, está arriba del soporte o no existe. No se extrapola ni se aplica
suavizado/isotonicidad para forzar un umbral. La regla estricta puede declarar
no identificación por ruido; se publica la grilla completa.

Bootstrap pareado por semilla, 2000 remuestreos, IC percentiles marginales del
95%, para diferencias respecto al control por valor de p y promediadas sobre
grilla. Se calcula Δp* y, en A, diferencia-en-diferencias entre transmisiones
únicamente cuando todos los umbrales requeridos están identificados. Cada
remuestreo utiliza los mismos índices para todas las variantes y grilla y
conserva el número/causa de draws inválidos. Los intervalos calculados solamente
con draws identificados se etiquetan condicionales. La supervivencia es un
endpoint secundario distinto, no un sustituto del p* ausente.

No hay ajustes de multiplicidad: todos los intervalos son exploratorios y no
autorizan afirmaciones familiares confirmatorias ni gates PASS/FAIL.

## Controles, interpretación y límites

- `copy_score_equal` iguala medias de puntaje condicionadas al linaje dentro de
  cada grupo; no redistribuye ingreso físico. Mantiene variación intraclase,
  cambia potencialmente rangos/orden y deja activa la copia grupal. No elimina
  todo el canal social.
- Generación 0 base/equal es físicamente idéntica; bajo conformity la igualdad
  persiste todas las generaciones, pues esa regla ignora los puntajes de copia.
- `resource_pool` conserva el ingreso total y lo distribuye por igual entre
  agentes vivos en cada instante. Diferencias de duración de vida pueden dejar
  diferencias de ingreso acumulado o de tasas medias de vida. Cambia fitness
  físico y señal posterior de copia: no identifica por sí solo una separación
  factorial perfecta entre beneficio real y social.
- C usa una presión ecológica contextual. Su p inicial es la proporción con un
  prior de restricción de Qg; no es un tipo conductual fijado durante la vida.
  La conformidad usa restricción observada, no etiqueta ancestral.
- Descuento corto cambia gamma en el mismo entorno variable; no cambia la
  duración física del episodio y no representa por sí solo un régimen constante.
- `coordinate_equivalent` cambia coordenadas de dos cabezas reteniendo todos
  sus grados de libertad: es control negativo de exclusividad representacional.
- Piloto A: equalización residual≤5.56e−16; error conservación pool≤1.17e−14;
  dispersión instantánea pool=0; control físico gen0 16/16 y conformity20gens
  8/8 exactos. p* no identificado para las seis condiciones.
- Piloto C: equivalencia de coordenadas 16/16 exacta incluso hashes finales en
  20 generaciones; 24/24 umbrales en o debajo del soporte. Algunas ablaciones
  mejoran supervivencia: ese resultado no motiva recalibrar ni descartar pruebas.

## Ejecución y preservación

Desde `/workspace/scratch/7dbf4975748c`:

```bash
PYTHONPATH=POLAR_reconciled python -m unittest p1_completion.tests_population -v
PYTHONPATH=POLAR_reconciled python -m p1_completion.population --phase final --part A --workers 6 --output POLAR_reconciled/p1_results/final_A
PYTHONPATH=POLAR_reconciled python -m p1_completion.population --phase final --part C --workers 6 --output POLAR_reconciled/p1_results/final_C
```

El padre autoriza los comandos finales únicamente después de congelar/publicar
el programa conjunto. Las partes se ejecutan secuencialmente para respetar el
presupuesto CPU. Puede seleccionarse una celda con `--cell-index N` (orden
transmisión, variante, grilla) o generar sólo el resumen con `--aggregate-only`.

Cada celda escribe un directorio nuevo con protocolo y hashes; los checkpoints
completos se validan antes de reutilizar. No se sobrescriben directorios parciales
ni resúmenes existentes. Se guardan outcomes, transmisiones, endpoints, snapshot
final completo, hashes de estado por semilla/generación y trazas diagnósticas
de la primera semilla en generaciones 0 y 19. La arquitectura compuesta del
snapshot se declara aparte y se reconstruye mediante `make_engine`.

Replay de una semilla (por defecto la primera):

```bash
PYTHONPATH=POLAR_reconciled python -m p1_completion.population --phase final --part A --output POLAR_reconciled/p1_results/final_A --replay-cell POLAR_reconciled/p1_results/final_A/success__copy_score_equal__p50
```

El replay compara todos los endpoints, transmisiones y estado final exactamente.
El hash de transición del batch es dependiente de la composición del batch y no
se compara contra una sola semilla; el hash diagnóstico sí corresponde a la
primera semilla. Para otras semillas puede usarse `--replay-seed-index N`.

Validación antes del freeze: 8 tests aprobados y dos replays piloto de 36 checks
exactos cada uno (A score_equal/p=.50 y C triple/p=.50, success/seed930001).
Pilotos preservados: A89MB y C378MB. Proyección final ~1.4GB total; entregar
archivos de evidencia divididos conservando íntegros los directorios de celdas.
