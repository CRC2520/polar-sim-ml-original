# Exportador v1.1: investigación del incidente original

La causa original es **undetermined** (indeterminada). El exportador histórico sigue fallido y sin cambios. La nueva canalización se verifica por separado para impedir la publicación cuando falla una comprobación.

Las 42 filas originales forman un prefijo completo terminado en salto de línea para los paquetes 32–45. Los 83 fragmentos gzip archivados contienen 82,176 eventos, incluidos 2,304 eventos diagnósticos para las 96 filas de piloto/paquete. La nueva reproducción recupera las 42 filas originales y las 54 filas del archivo auxiliar anterior byte por byte.

Se identifica una carencia de control conocida: el cierre técnico comprueba las cantidades de ensayos de arquitectura y las restricciones de seguridad, pero no exige los 96 identificadores diagnósticos antes de crear la instantánea. Esa omisión explica que una instantánea técnicamente cerrada pueda coexistir con un resumen incompleto; no establece la causa de las escrituras faltantes.

| Hipótesis | Hallazgo | Evidencia conservada |
|---|---|---|
| incomplete_grouping_key | not_reproduced | El prefijo conservado contiene identificadores únicos (pilot,bundle); la función histórica run_bundle_diagnostics añade una fila por paquete completo y no agrupa una tabla en memoria. |
| overwrite | not_established | El flujo canónico de calibración se abre en modo x y append_row únicamente escribe en ese flujo. La telemetría de ejecución conservada no cubre una sobrescritura posterior en el sistema de archivos ni una copia parcial. |
| deduplication | not_reproduced | No existe ningún paso de deduplicación entre run_bundle_diagnostics y su flujo; los 42 identificadores conservados son únicos. |
| cycle_omission | not_consistent_with_prefix | Las filas diagnósticas corresponden a paquetes fuente/simulada de las leyes de la tarea, para los cuales no se aplica el ciclo. Los ciclos 1/2 de arquitectura C2 están presentes independientemente en la cobertura completa del archivo. |
| context_omission | not_consistent_with_prefix | Cada paquete diagnóstico conservado contiene tanto kappa1 como kappa2; los 2,304 diagnósticos archivados cubren los 96 paquetes y ambos contextos. |
| route_bypass_omission | not_consistent_with_prefix | El flujo diagnóstico contiene leyes conjuntas fuente/simulada, no ensayos exclusivamente de ruta. Los programas de arquitectura intacta y con omisión de ruta están completos de forma independiente en el archivo autenticado. |
| premature_flush | not_reproduced | La función histórica append_row vacía el búfer después de cada fila; al reproducir las 96 filas conservadas mediante esa función extraída sin cambios se conservan exactamente 96 filas. No existe un registro de durabilidad fsync. |
| exception_handling | not_established | Las excepciones dentro de la calibración se propagan después de que finally archive las trazas. El cierre técnico satisfactorio requiere alcanzar el código posterior a los bloques with/finally. El cierre histórico omite una comprobación de integridad de las filas diagnósticas. |
| partial_shard_ingestion | not_reproduced | El exportador original lee directamente el JSONL de 42 filas; la nueva reproducción autenticada consume los 83 fragmentos y los 82,176 registros. La copia parcial del resumen redundante sigue siendo posible, pero no está demostrada. |
| ordering | not_reproduced | Las 42 filas forman exactamente el prefijo de paquetes 32–45 en el orden de pilotos P5,P6,P7. El nuevo exportador valida los conjuntos de identificadores y rechaza duplicados independientemente del orden de entrada. |
| filtered_records | not_reproduced | El exportador original analiza cada línea JSONL no filtrada antes de comprobar los 96 identificadores esperados. El nuevo analizador del archivo inspecciona todos los tipos de registros; ninguno se excluye antes de las comprobaciones de cobertura. |
| file_finalization | not_established | El gestor de contexto del flujo de resumen termina antes de las instantáneas del cierre técnico. Se conserva un prefijo completo terminado en salto de línea. Ningún registro de operaciones sobre archivos ni telemetría de caída/fsync establece por qué faltan las filas posteriores. |

Reproducir únicamente la función histórica de anexado sin modificar sobre los 96 registros existentes produce un archivo de 96 filas idéntico byte por byte. Esta investigación no ejecuta ningún modelo, controlador, semilla científica ni tarea.

El exportador prospectivo fija el manifiesto anterior, valida el hash de cada fragmento y todos los identificadores completos de eventos (pilot, bundle, comparator, context, route, cycle, replicate y epoch), comprueba el esquema y publica el resumen, la auditoría y el manifiesto mediante un único cambio de nombre atómico de directorio, solo cuando todas las comprobaciones han pasado. El manifiesto tiene un archivo SHA256 separado. Los registros faltantes o duplicados, las dimensiones clave ausentes y los campos causales malformados producen un fallo antes de publicar cualquier resumen.

El cierre es **RESOLVED_FOR_FUTURE_PIPELINE**. El incidente histórico y su causa indeterminada permanecen como limitaciones registradas. Todos los artefactos son exclusivamente de desarrollo, no confirmatorios e inutilizables como datos finales B1-E.
