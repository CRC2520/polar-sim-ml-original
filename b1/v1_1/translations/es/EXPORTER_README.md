# Exportador v1.1 que impide publicar ante fallos

Esta versión autentica la entrada conservada y valida los identificadores completos de eventos y su esquema; después publica la auditoría, el resumen y el manifiesto completos mediante un cambio de nombre atómico de directorio. Nunca sobrescribe una exportación existente. El exportador histórico, su estado fallido, el cierre técnico y todas las trazas originales permanecen sin cambios.

Ejecute desde el repositorio de código:

```bash
python -B -m unittest b1.v1_1.exporter.test_exporter -v
python -B -m b1.v1_1.exporter.core verify-v1
python -B -m b1.v1_1.exporter.core archive --output /tmp/new-archive-export
python -B -m b1.v1_1.exporter.qa --output /tmp/new-qa-run
```

La suite de pruebas utiliza directorios temporales y conserva todos los artefactos versionados. El comando archive no inicia tareas ni semillas. Comprueba el manifiesto B1-D v1 fijado y los hashes de los 480 artefactos, los 83 fragmentos de trazas y los 82,176 identificadores completos de eventos; después reconstruye 96 filas diagnósticas. Las 42 filas originales y las 54 previamente recuperadas deben coincidir byte por byte. Un archivo SHA256 separado autentica el manifiesto de salida.

El identificador histórico es `(record_kind, pilot, bundle_index, comparator, context, route_status, cycle, replicate_id, epoch)`. El adaptador histórico resuelve los identificadores de comparadores enmascarados mediante el mapa de identidades congelado. `cycle=0` significa explícitamente que no se aplica el ciclo a los registros diagnósticos y de testigos físicos. Los diagnósticos conjuntos fuente/simulada utilizan `joint_source_not_selective`; nunca se convierten en lesiones selectivas de Gamma. Las rutas de arquitectura distinguen los registros intactos de aquellos con omisión de ruta.

La API normalizada de desarrollo es `export_development(directory, plan_path, plan_sha256, output, *, index_sha256)`. Su plan de expectativas contiene una lista completa `expected_ids`, un espacio de nombres de desarrollo declarado y `fixed_before_execution=true`. Quien la invoque debe congelar ese plan antes de recopilar eventos. El hash del índice del archivo se proporciona por separado después de la recopilación. Cada registro debe coincidir con el espacio de nombres del plan y con el esquema exacto. Los espacios de nombres admitidos son `PD-B1-D-v1.1` y `PD-B1-D-v1.1-QA`; no se proporciona ejecución de datos finales ni generador de semillas. `export_qa` exige además el espacio de nombres QA. El espacio de nombres por sí solo no establece una congelación científica; la ejecución que lo utiliza debe conservar su compromiso con el plan previo a la ejecución.

El caso de prueba QA conservado declara su plan de identificadores antes de ejecutar la tarea y comprueba el hash del plan después. Ejecuta 16 episodios QA P6/P7, incluidos ambos ciclos C2, C6, ambos contextos y rutas intactas/con omisión, produciendo 1,536 eventos. Las políticas v1 seleccionadas permanecen sin cambios. Los hashes de eventos C6 y la pérdida racional coinciden exactamente con C1. Esta comprobación de la canalización es exclusivamente de desarrollo y no aporta evidencia experimental.

La causa del truncamiento original permanece indeterminada. La investigación cubre las 12 hipótesis solicitadas y reproduce la función de anexado sin modificar sobre los registros conservados, sin reproducir el truncamiento. Una carencia de control confirmada es la ausencia de una comprobación de integridad diagnóstica en el cierre técnico histórico. Los informes futuros ahora rechazan la publicación ante fallos, mientras que el incidente original permanece registrado como parte de la procedencia.
