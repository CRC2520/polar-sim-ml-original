# Borrador prospectivo AFR-S23-0.1 — pasos 2 y 3

**Estado:** propuesta no ejecutada ni lista todavía para generación final. Todos los números siguientes son parámetros de diseño propuestos, no resultados. El código de simulación, thresholds de atribución, calibrador y estimador de intervalos se congelarán tras pruebas de desarrollo. El registro incluirá qué se fijó y antes de qué datos; no será revisión por pares ni replicación independiente. El desarrollo preliminar `iteration2/` no sustituye esta especificación.

## Pregunta única de confirmación

**H_A:** usar atribución física de procedencia para admitir actualizaciones reduce al menos 0,0005 el MSE externo normalizado frente al mismo controlador que usa solo coherencia de metadatos, manteniendo idéntica la compuerta de excitación, cuando existen efectos foráneos distinguibles, sin superar los márgenes de deterioro en condiciones limpias, recuerdo y cambios reales de dinámica.

H_A es una hipótesis operacional sobre admisión de aprendizaje. No prueba la hipótesis arquitectónica H_R, el significado de las parejas ni conciencia. Su contraste primario es A1X1−A0X1; no mezcla el cambio de atribución con otro cambio simultáneo de excitación.

## Diseño y particiones

Primero: diagnóstico exploratorio 2^5 (signo, coactivación, simetría receptora, estimación, regla de decisión) con snapshots y datos comunes, según FIDELIDAD_Y_PLAN. Publicar todas las combinaciones e interacciones; resultados locales y de políticas cerradas se mantienen separados. Este diagnóstico no se presenta como confirmación de H_A.

Desarrollo: 12 seeds 2600001–2600012. Calibración separada: 12 seeds 2605001–2605012. Confirmación: N=64 seeds completas 2610001–2610064. Son reservas propuestas que deben cotejarse contra todos los registros de exposición; no se declara comprobada su condición de intocadas sin ese cotejo. N es fijo, no se aumenta tras ver resultados y no constituye una garantía de potencia.

Tres regímenes físicos (diagonal, sparse con cruces que contradicen el pairing nominado, denso no organizado por parejas), dos ruidos y tres tipos de feedback (limpio, extranjero distinguible, fuentes indistinguibles): 18 celdas por seed. Cuatro brazos factoriales A/X y dos controles de equivalencia de A1X1: 6912 trayectorias finales previstas, 64 acciones de identificación y 192 pasos de evaluación cada una. A activa únicamente uso de atribución; X activa únicamente gate de excitación. Puntajes, candidatos y cálculos se ejecutan en todos los brazos; solo cambia la decisión de incorporarlos al aprendizaje. El genérico equivalente y signed-plus-intensity verifican identidad computacional, no compiten por superioridad.

El scaffold físico, restricciones y secuencia de metas provendrán del generador de control dinámico fijado como base; el adaptador añade regímenes, ruido y contaminación sin usar tau ni etiquetas filosóficas para generar éxito. Las asignaciones estructurales y transiciones exactas deben quedar en código y manifest antes del final. No se mezclan sus pérdidas con las de Studies 1–3/P1/P2. La ampliación a familias independientes y RL modular/monolítico pertenece al paso 5, no se declara realizada aquí.

## Intervención de procedencia y controles

Durante el régimen foráneo, un calendario exógeno de reemplazo con probabilidad 0,20, común por seed, sustituye el paquete de efecto destinado al estimador por un efecto de otra fuente candidata. El token conserva actor/action_id plausibles: rechazar un token incorrecto no prueba atribución física. Las historias candidatas y la información de metadatos se suministran por igual a todos. El rótulo verdadero y la trayectoria física de evaluación no se entregan al gate. Solo el canal de evidencia para actualizar Bhat se contamina: no modificar simultáneamente el sensor usado por el planificador, metas, permisos, recompensas o proceso físico.

La atribución utiliza predicciones y covarianzas anteriores al update. Aceptar como propio requiere ajuste absoluto y margen contra la segunda fuente, fijados con calibración independiente. Si todas fallan: unexplained. Si los efectos son indistinguibles: ambiguous y abstención de actualización, sin penalizar la imposibilidad de identificar una fuente. Las fuentes candidatas son suministradas: esto no prueba identificación abierta de actores desconocidos.

X exige amplitud y excitación frente al ruido calibrado, sin ratio y/u en acciones casi nulas. Sus puntajes se calculan en todos los brazos. Pulsos de identificación acotados, calendario de metas y restricciones son iguales; la capacidad de cada política de responder puede producir trayectorias diferentes. Rechazo de feedback propio después de un cambio real es un riesgo explícito, no un dato a eliminar.

La memoria separa valor, confianza y vigencia; no atenúa el valor verificado por edad. Todos los brazos usan la misma política de memoria. Gate de retrieval, borrado del registro y revocación de copia activa se estudian en ensayos factoriales separados. La review permanece desactivada en esta confirmación. El monitor emite probabilidades vinculadas a las acciones ejecutadas antes del feedback y se evalúa en shadow mode.

## Métricas, incertidumbre y decisión

Pérdida externa l_t=sum_i w_i*((y_i−target_i)/range_i)^2 / sum_i w_i; range_i se fija con unidades del entorno. Primaria: diferencia de MSE media A1X1−A0X1 en feedback foráneo distinguible, promediada por igual sobre sus seis celdas. Cada seed produce UNA diferencia, no 192 réplicas. Datos limpios, retención y recuperación después de cambios son guardas, no sustitutos de una primaria fallida. Interacciones A×X, tasas de aceptación foránea y rechazo propio, error predictivo y ambigüedad son diagnóstico de mecanismo; el rechazo por sí solo no es éxito externo.

Usar 20000 bootstrap emparejados de las 64 seeds completas, RNG 2620001. Cinco límites superiores unilaterales 99% proporcionan corrección Bonferroni .05/5 para primaria y cuatro guardas escalares. La cobertura percentil es aproximada. Primaria requiere U99(delta_foreign)<−0,0005. Guardas: U99(delta_clean)<=+0,00025; U99(delta_recall)<=+0,00025; U99(delta_postshift)<=+0,001; U99(delta_cost)<=+0,02 de capacidad de recurso normalizada; cero violaciones duras. Los márgenes son juicios de ingeniería propuestos, no tolerancias psicológicas ni utilidad externa ya validada; fijar su justificación antes de abrir finales y no adaptarla después.

Reportar aparte Brier, calibración, discriminación y coverage sobre el evento de éxito fijado antes de calibrar. Si no existe una clase, marcar discriminación no definida. El costo computacional incluye evaluación de candidatos/gates; runtime y memoria pico se describen, sin confundir costo de acciones con eficiencia informática. Registro completo, trazas finitas, datos sin duplicados, identidad de recodificación y fórmula genérica <=1e-10 y comprobación sin leakage son condiciones de validez. Si falta un dato por fallo del controlador, no descartarlo: regla de fallo y denominador fijados antes del final. Fallos de infraestructura y violaciones del protocolo invalidan la ejecución; no se sustituyen seeds para rescatarla.

**Regla literal:** la función `decide()` de `decision_rule.py`, junto con `protocol_parameters.json`, es la autoridad. Sin registro: `not_registered`. Con registros incompletos, no finitos, fuentes distintas, leakage, intervención inejecutada o controles de identidad fallidos: `invalid_experiment`. Con primaria y guardas aprobadas: `provenance_use_supported_in_scope`. En cualquier otro caso: `provenance_use_not_supported_in_scope`. Su texto exacto se extrae automáticamente al paquete; no se mantiene otra versión redactada del algoritmo.

Si falla la utilidad, pueden permanecer pruebas L1 y efectos L2 válidos, pero no se acepta H_A en el alcance registrado. No se concluye efecto nulo, equivalencia de modelos ni refutación de la arquitectura. Incluso si pasa, H_R y la afirmación `suspend_exclusive_polar_advantage_claim` no cambian. No optional stopping, no rescate por subgrupo y no selección de nuevos márgenes después de observar finales.
