# Equiparación de recursos: B1-D v1.1

Estado: CERTIFIED_WITH_EXPLICIT_SCOPE. El bloqueador de recursos queda resuelto para la familia sintética finita de tareas y controladores congelada. Esto certifica la oportunidad lógica máxima declarada, no la igualdad del uso físico de CPU, de las instrucciones de hardware ni un nivel de servicio de tiempo real estricto.

Los 288 registros de piloto/comparador/eje separan el presupuesto permitido, las necesidades solicitadas por la implementación y el uso real medido. C0–C4 reciben 16,384 posiciones escalares de memoria, 65,536 unidades del contador anterior, 262,144 unidades lógicas ampliadas y 131,072 unidades de la aproximación del espacio de trabajo temporal. C5 recibe el doble de asignación y sigue siendo descriptivo. No se añaden operaciones ficticias ni ensayos de entrenamiento a C4.

Las cotas proceden de tres celdas, 32 épocas, colas de operaciones de un paso, como máximo 32 registros de informes/selector/instantáneas, dos trabajos de servicio y dos instancias P7. Suman todas las ramas de forma conservadora, incluidas las mutuamente excluyentes. La peor cota persistente de P6 para C0–C4 es de 3,408 posiciones escalares y la cota del contador de decisiones anterior es de 4,119, ya inferiores a los límites v1 sin modificar de 4,096 y 8,192. Por tanto, los límites no truncan ninguna política canónica compatible, incluido el algoritmo convencional completo. La envolvente v1.1 más amplia añade margen prospectivo sin modificar los archivos anteriores.

La contabilidad lógica es reproducible y cubre las operaciones primitivas de despacho, inspección de registros y ordenación, además de los recorridos de búferes escalares y del trabajo sobre rutas y candidatos contabilizados explícitamente. Su unidad es una aproximación documentada. El espacio de trabajo temporal utiliza una aproximación conservadora basada en recorridos escalares; las asignaciones de bytes de Python se miden independientemente con tracemalloc. El certificado no convierte escalares en bytes ni afirma que la aproximación sea una demostración exacta de las asignaciones de CPython.

El C4 convencional de P5 utiliza observaciones brutas completas y realiza el cálculo de reposición mínima para la demanda siguiente sin reposición terminal. Solicita menos posiciones de ruta y almacena menos observaciones; comparte la envolvente de oportunidad. El testigo de reposición completa permanece separado. Las políticas prospectivas de P5 no reciben ajuste basado en resultados. P6/P7 conservan todas las configuraciones seleccionadas de v1. C2 evalúa ambos ciclos fijos; su programa duplicado de evaluación y la excepción analítica de C4 se cuantifican explícitamente, sin afirmar igualdad del cómputo total real de entrenamiento.

El QA ejercitó 14,406 decisiones medidas sobre todas las configuraciones congeladas, ambos ciclos C2, observaciones con el esquema máximo y tres programas deterministas de estrés por piloto. Los controladores con límites ampliados y originales produjeron acciones y memoria idénticas. Todas las infracciones de las restricciones de seguridad de la tarea fueron cero. Además, P7 reprodujo los 32 paquetes de entradas de calibración archivados: 224 episodios y 7168 decisiones con acciones, parámetros y memoria idénticos. Son comprobaciones de infraestructura mediante QA, no nuevas muestras experimentales.

El C1 seleccionado de P7 sigue siendo cfg01/reactivo. Sus 64 decisiones de QA de intervención aguda intacta/con omisión de ruta conservan acciones idénticas en ambos contextos. Este cambio de envolvente de recursos no requiere una repetición científica ni un nuevo ajuste. La manipulación conjunta fuente/simulada existente sigue limitando la atribución.

| Piloto | Cargo lógico máximo | Estado escalar presupuestado máximo | Asignación máxima trazada durante act (bytes) |
|---|---:|---:|---:|
| P5 | 3220 | 161 | 4592 |
| P6 | 60700 | 3362 | 76120 |
| P7 | 20167 | 1148 | 30944 |

Las cifras de latencia, CPU y asignación se miden en este equipo de QA y aparecen por comparador en el JSON. No constituyen cotas superiores estrictas portables. Ningún plazo trunca las decisiones. Los ejes de CPU, latencia y memoria en bytes son `unmatched_but_quantified`; el estado decisivo de invariantes se limita a la oportunidad lógica máxima con estas restricciones explícitas de las afirmaciones. C5 nunca recibe PASS de invariantes equiparados.

No se crearon semillas finales ni ejecuciones B1-E. Los hashes de las fuentes congeladas se registran antes y después; todos coinciden.
