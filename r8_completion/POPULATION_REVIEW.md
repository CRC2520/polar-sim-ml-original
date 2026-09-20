# Revisión independiente del bloque poblacional R8

Fecha: 2026-09-20. Alcance: revisión previa a la congelación y a las semillas
finales. Esta revisión se realizó por un agente distinto del autor, en el mismo
entorno de trabajo; no constituye una replicación externa. No se modificó código
del experimento ni se ejecutaron las semillas finales 941001–941030 o
942001–942030.

## Código revisado

| Archivo | SHA256 de la instantánea revisada |
|---|---|
| `r8_completion/population.py` | `220eb80b98a604dc8ca67629c4564acc1cd8ba969cfb6801629a3c228c4e840b` |
| `r8_completion/population_thresholds.py` | `cf2557b43bbd0112d72aff5d2a6ccbb8e3999e64207366aaaef83825d4a01b9f` |
| `collective/bridge_v2/engine.py` | `48330c1a88a2312182bb488dae0096e2ccf674c6fe2b0ca7ddf6fa9b8a29dccc` |

El autor continúa preparando el diseño. Una modificación posterior requiere
revisar su diferencia respecto de esta instantánea; esta tabla no congela el
experimento ni autoriza ejecutar finales.

## Comprobaciones realizadas

- `python -m unittest r8_completion.tests_population
  r8_completion.tests_population_thresholds -v`: **24 pruebas aprobadas**.
  Seis comprueban adaptadores, ajuste factual, normalización, particiones de
  semillas y controles; dieciocho comprueban umbrales y selección piloto.
- Comprobación adicional independiente con semilla de desarrollo 940083:
  ambas reglas (`success`, `conformity`), fracciones iniciales 0, 0.5 y 1,
  tres generaciones, dos grupos, treinta pasos y calentamiento de una
  generación. En las **18 combinaciones** de regla, fracción y variante
  (`base`, `copy_score_equal`, `resource_pool`), todos los arrays históricos
  y de transmisión coinciden exactamente con `BridgeEngine`.
- En las mismas condiciones, al fijar donantes de baseline, activar o
  desactivar la igualación de puntajes produjo igualdad exacta de **todos**
  los arrays históricos, con y sin pooling: **12 comparaciones**.

Estas comprobaciones pequeñas validan contratos y casos de borde. No son
réplicas adicionales ni evidencia confirmatoria de utilidad.

## Hallazgos causales y de información

| Componente | Resultado de la revisión | Alcance correcto |
|---|---|---|
| Factorial puntaje × pooling | Ambos switches son independientes; las variantes de un switch reproducen la física y transmisión heredadas. | Las cuatro combinaciones con transmisión endógena estiman efectos totales dentro del simulador. |
| Donantes fijos | Se fijan destinatarios, donantes, reemplazos de grupos y mutaciones desde el baseline emparejado. Igualar puntajes deja de modificar decisiones de copia. | El efecto de pooling queda condicionado a ese régimen de copia; no es una identificación de efectos naturales directos o indirectos. Las tablas Q copiadas aún pueden diferir por el tratamiento. |
| Etiquetas de arbitraje | Los valores Q y el estado preceden a la acción. Se seleccionan exclusivamente probes uniformes efectivamente realizados. El retorno usa vida propia y recurso público observado durante los 40 pasos siguientes. | El retorno contiene consecuencias de la acción inicial y de la continuación real de la política; no es un efecto causal aislado de un solo paso. No utiliza etiquetas de acciones no realizadas ni crecimiento latente. |
| Separación de datos | Entrenamiento, validación, potencia piloto, piloto de umbral y finales tienen semillas disjuntas. Ridge se selecciona únicamente con validación de desarrollo. | No se observó filtración de evaluación final en el ajuste de hiperparámetros. Las Q sí siguen aprendiendo durante las realizaciones finales, como define el protocolo. |
| Permutación de pesos | `shuffled_common` permuta una vez las 20 filas de la misma tabla de `state_common`; conserva los 60 pesos, escalas y datos. | Es una lesión de la correspondencia estado→pesos aprendida. No representa un competidor genérico reentrenado u optimizado. |

## Precisiones de comparación

**Q2 compara capacidad y condicionamiento conjuntamente.** `state_common`
dispone de 20 × 3 pesos, frente a los 3 pesos de `global_common`. Aunque
reciben los mismos datos de calibración y presupuesto de selección, no tienen
igual capacidad paramétrica. Este punto fue comunicado al autor y al
responsable principal y ambos aceptaron delimitar la afirmación. El autor añadió
el contraste descriptivo `state_common − shuffled_common`; la inspección de
seguimiento confirmó su presencia en `summarize_q`. `POPULATION_DESIGN.md` y
`CLAIMS_AND_CONFIRMATION.md` distinguen también ambas capacidades. No se
introdujo una nueva hipótesis confirmatoria ni se convirtió la permutación en
un rival genérico entrenado.

**Normalización y temperatura.** Normalizar las ventajas Q modifica tanto
las escalas relativas entre cabezas como la magnitud total de los logits a
temperatura fija. Q1 evalúa ese paquete, sin separar el ajuste entre señales
del cambio de entropía de acción. La suma de pesos igual a tres no garantiza
igual entropía entre políticas.

**Descomposición factorial Q incorporada.** Además del contraste combinado Q1,
se verificaron cinco estimaciones descriptivas: normalización con sucesores
independientes; normalización con sucesor común; sucesor común sin normalizar;
sucesor común después de normalizar; e interacción de los dos factores. Todos
los brazos necesarios están presentes. Las cinco fórmulas del bloque
`factorial_secondary_descriptive_ci` coinciden con esos contrastes; no cambia
la familia de seis tests. El diseño también explicita la diferencia de
dispersión de logits y alinea la potencia con la familia vigente de seis.

## Umbral de composición p*

La rejilla de desarrollo cubre 0–1; la refinación conserva ambos extremos,
permanece dentro del soporte y usa múltiplos de 0.05, realizables exactamente
con veinte agentes. La selección de presión y rejilla depende solamente del
piloto. Si no aparece un cruce piloto elegible se conserva una exploración
diagnóstica completa y su etiqueta `final_diagnostic_only`.

El estimador no extrapola. Distingue cruce interior, reversión de cruce,
viabilidad desde el borde inferior y ausencia de cruce dentro del soporte.
La interpolación es descriptiva. Permitir pequeñas caídas que no cruzan de
nuevo 0.5 no demuestra monotonía; un cruce único observado en una rejilla
tampoco demuestra unicidad entre puntos no observados.

Las bandas binomiales simultáneas se ajustan dentro de cada perfil. El
bootstrap extrae filas completas y mantiene el pareamiento entre fracciones.
El intervalo condicionado a réplicas con cruce se identifica como condicional;
no se confunde con un intervalo incondicional del umbral. La regla del 80%
de cruces estables es un criterio diseñado, no una probabilidad posterior.

## Pendiente después de los experimentos

Verificar la congelación efectiva de fuentes/protocolos, integridad de los
artefactos y reconstrucción independiente de endpoints desde historiales y
transmisiones; después recalcular los treinta indicadores Q1/Q2, contrastes
factoriales, controles negativos y perfiles de p*. Se preservarán resultados
adversos y falta de identificación. Esta revisión previa no anticipa éxito
de ninguna hipótesis.
