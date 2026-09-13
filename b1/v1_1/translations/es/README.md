# POLAR DYNAMICS — B1-D v1.1

Esta iteración prospectiva del Programa A aborda los cuatro bloqueadores congelados en B1-D v1. El código histórico, las observaciones, las exclusiones de semillas, los contratos B0 y los manifiestos anteriores permanecen sin cambios. El Programa B sigue en B-design.

La evidencia ejecutable tiene propósitos separados:

- `p5/`: una nueva comparación de desarrollo congelada, con C4 de reposición mínima para la demanda siguiente y un testigo del techo físico con reposición completa, identificado por separado. P5 sigue sin ser elegible para superioridad positiva en utilidad o especificidad del emparejamiento.
- `resources/`: cotas para tareas y controladores finitos, medición del uso real y reproducción para QA. Las asignaciones comunes más amplias conservan las acciones seleccionadas de P7. El certificado se refiere a la oportunidad lógica declarada, con limitaciones explícitas para la CPU física, la asignación de memoria y la latencia.
- `exporter/`: reproducción autenticada de archivos y una canalización futura que rechaza la publicación si falla una comprobación. Sus 96 diagnósticos recuperados son observaciones históricas, nunca una muestra nueva. Se conservan el resumen original incompleto de 42 filas y el incidente de causa indeterminada.
- `statistics/`: inferencia prospectiva de medias pareadas con N fijo y validación sintética. Las simulaciones prueban el método estadístico, no Polar Dynamics.
- `retention/`: un subconjunto de auditoría predeclarado y evidencia causal completa, con conservación de las trazas completas de los fallos y las infracciones. Las proyecciones de recursos son estimaciones, no reservas.

Cada resultado científico de desarrollo establece `development_only=true`, `confirmatory=false` y `reusable_as_final=false`. Las nuevas semillas de tareas utilizan `PD-B1-D-v1.1`; el QA de ingeniería utiliza `PD-B1-D-v1.1-QA`. Las reproducciones conservan explícitamente su identidad original. Aquí no se implementan semillas finales ni ejecución B1-E.

La política seleccionada de P7 C1 sigue siendo la reactiva `cfg01`. Las diferencias históricas pareadas y de intervención aguda sobre la ruta siguen siendo cero de forma descriptiva. Las manipulaciones fuente/simulada siguen siendo conjuntas y no pueden establecer selectividad específica de Gamma. Los controles negativos y los hallazgos nulos se conservan sin reinterpretación favorable.

Ejecute las pruebas originales con `python -B -m unittest discover -s b1/tests -v` y las nuevas con `python -B -m unittest discover -s b1/v1_1 -t . -p 'test_*.py' -v`. El antiguo validador de recuperación espera exactamente el inventario Python anterior y se prueba en una copia de trabajo separada del commit v1 fijado. El nuevo validador comprueba tanto la base inmutable como el inventario añadido de v1.1.

Lea conjuntamente `B1D_V1_1_RESULTS.json`, `BLOCKER_CLOSURE_MATRIX.csv` y el `B1D_V1_1_MANIFEST.json` final. La finalización significa que el instrumento de desarrollo está técnicamente preparado dentro del alcance declarado; no implica superioridad empírica, generalización, conciencia, AGI ni ASI. La autorización confirmatoria final permanece siempre en falso en esta iteración.

## Documentación en español

- [Plan de resolución de bloqueadores](BLOCKER_RESOLUTION_PLAN.md)
- [Disposición prospectiva de P5 C4](P5_C4_DISPOSITION.md)
- [Informe de equiparación de recursos](RESOURCE_MATCHING_REPORT.md)
- [Investigación del incidente del exportador](EXPORTER_ROOT_CAUSE.md)
- [Rediseño estadístico prospectivo](B1E_STATISTICAL_REDESIGN.md)
- [Política de conservación de trazas](TRACE_RETENTION_POLICY.md)
- [Resultados de B1-D v1.1](B1D_V1_1_RESULTS.md)
- [Revisión adversarial independiente](ADVERSARIAL_REVIEW.md)
- [Uso del exportador v1.1](EXPORTER_README.md)

`TRANSLATION_INDEX.json` relaciona cada traducción con su fuente y registra ambos SHA-256. Los identificadores ejecutables, nombres de archivos, fórmulas y valores numéricos se conservan para facilitar su verificación.
