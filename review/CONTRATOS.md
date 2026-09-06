# Contratos propuestos de interfaz y auditoría

Estado: especificación pendiente de implementar/auditar; no atribuirla al código congelado. Los tipos son conceptuales de arrays finitos de float64, sin NaN/Inf ni coerciones silenciosas. Una observación ausente lleva máscara y estado de validez; cero numérico no significa ausencia.

## 1. Convenciones e invariantes globales

n polaridades operacionales; d estados ambientales; m=2n canales; H entero positivo para horizonte, distinto de eta de relajación y de eta_i de liberación en la ecuación original. p[:,0]=p+ y p[:,1]=p-. a=p+−p-, q=p++p- y p±=(q±a)/2; dominio |a|<=q<=2−|a|. Mantener inactividad, predominancia y coactivación. El equilibrio no impone a=0 ni uniformidad.

Bhat[d,m] estima efectos físicos de acciones. W[m,m] y K[m,k] representan rutas declaradas de actividad/features solo donde están implementadas; registrar origen `configured`, `identified` o `derived`. La tau escalar histórica y el nuevo vector (residuo firmado, déficit positivo, negativo, coactivación) no tienen el mismo tipo ni significado. No escribir tau para ambas formas sin versión y mapa explícitos.

Todos los snapshots, masks, priors y pesos compartidos se congelan antes del contraste. Las invariancias se comprueban con el decoder transformado, no con igualdad textual de nombres.

## 2. Estados y operaciones

| Interfaz | Tipo/dominio | Escritura y uso | Log mínimo |
|---|---|---|---|
| PolarState p_t | float64[n,2], [0,1] | Estado recurrente distinto de creencia ambiental y acción previa. Su término de continuidad es una hipótesis operacional | p_before/p_after, regla y coeficiente de actualización, versión, consumidor que lo usa |
| Intention v_t | float64[n,2], [0,1] | Propuesta con cotas de canal, previa a restricciones institucionales/recursos. Nunca ejecutada directamente | valor, predictor y objetivo que la generan; constraints omitidas declaradas |
| ExecutedAction u_t | float64[n,2], [0,1] | Solo acción aplicada; lower<=u<=allowed, c·u<=budget cuando factible | action_id, actor, tiempo, decoder_version, hash, costos reales, permisos y cumplimiento |
| Belief xhat_t | float64[d], unidad física declarada, incertidumbre separada | Estimación del estado ambiental desde observaciones y acciones admitidas; no es p_t | sensor mask, edades, modelo, actualización medida/predicha |
| MemoryRecord | id, version, value:float64[k], units, context, verification, valid_until opcional, retrieval_confidence, timestamps | value cambia solo con evidencia observada/versionada o edición autorizada. Edad no multiplica value. Confianza de coincidencia/contexto y política de liberación son distintas | old/new hashes, evento verificador, procedencia, por qué se almacenó/recuperó/liberó |
| GoalCache | copy_id, source_record_version, payload, delivered_at, validity, revoked | Copia activa con su propio ciclo. Un registro borrado o gate cerrado no la revoca automáticamente | creación, uso, expiración/revocación, id de decisión influida |
| CtMessage | id/version, goal[H,d], residual[d], deficit±[d], coactivation[n], permissions[m], cue opcional, units, source, time, recipient, masks | Selección/transmisión; no promedio de scores ni presupuesto. Permisos recibidos son informativos y no pueden elevar las autorizaciones de Et | candidato elegido/rechazado, hash por destinatario, bytes, timing, cómputo, recepción y uso |
| InstitutionalRules Et | rule_id/version, authority, priority, prohibition[m], minimum[m], exception_condition, reason | Genera constraints factibles o conflicto explícito; ausencia de solución no se disfraza de compliance | reglas activas, conflictos, obligaciones no satisfechas, autorización, acción resultante |
| EffectEstimate Bhat | float64[d,m], covariance por fila, support_mask, sample counts | Estima coeficientes de transiciones admitidas. Support/routing y significado de parejas no se aprenden automáticamente | matriz antes/después, filas usadas, regressor, predicción anterior y decisión de admisión |

Una memoria «verificada» significa valor respaldado por una observación o instrucción autorizada, no por la verdad oculta del evaluador. Esa verificación no garantiza vigencia futura: el sistema puede revocar aplicabilidad por nueva evidencia/contexto sin alterar arbitrariamente el valor almacenado.

Tres operaciones diferentes: `erase_record(id)` no toca GoalCache; `set_retrieval_gate(id,false)` no borra nada; `revoke_copy(copy_id)` no borra MemoryRecord. `edit_record` crea nueva versión y no propaga automáticamente esa versión a copias ya entregadas. Una sincronización autorizada es un cuarto evento registrado.

## 3. Las cuatro funciones del automodelo

### F1. Predicción de efectos

Entrada: snapshot Bhat_t^- y estado relevante, acción realmente emitida, covarianza de ruido identificada en desarrollo. Salida: predicción mu_self y covarianza predictiva antes de ver el efecto que se adjudicará. El estimador de Study 1 y el de P1 son versiones distintas, no piezas de una sola prueba.

### F2. Atribución de procedencia

Primero comprobar actor, action_id, secuencia y ventana temporal. Después comparar el vector de efecto recibido con predicciones de fuentes candidatas cuya historia de acciones está disponible a TODOS los brazos. Un token correcto pero asociado a otra señal debe ser un caso explícito. No usar `true_actor` ni etiqueta de contaminación en la decisión.

Para candidatos c, evaluar d_c=(z−mu_c)^T(Sigma_c+epsilon I)^(-1)(z−mu_c) con matrices anteriores a actualizar Bhat. Resultado: `own`, `foreign_candidate`, `ambiguous` o `unexplained`, con puntajes y márgenes. Admisión propia requiere compatibilidad absoluta y separación frente al segundo candidato; umbrales se fijan con desarrollo independiente. Si predicciones son indistinguibles, abstenerse de atribuir y de actualizar, no inventar confianza. Si faltan candidatos o todas las predicciones fallan, `unexplained`.

Esto sigue siendo atribución condicionada a candidatos; incluirla en el loop no convierte candidatos suministrados en descubrimiento de fuentes desconocidas. La coherencia del token no es autenticación criptográfica. Los cambios reales de dinámica pueden causar falso rechazo propio: medirlo y añadir guarda de recuperación.

### F3. Fiabilidad prospectiva

`DecisionForecast` contiene decision_id, action_id, policy_version, timestamp_pre_feedback, horizonte de evaluación, definición de éxito y probabilidad en [0,1]. El evento debe referirse a la acción/política realmente emitida. Brier, calibración, discriminación y cobertura se reportan separados del error de control; AUROC se marca no definido si falta una clase. La calibración de una política no se transfiere sin prueba a otra política que modifica las trayectorias.

La incertidumbre interna usada para planificar no se llama probabilidad calibrada salvo evaluación correspondiente. El score residual y la probabilidad son campos distintos con versiones diferentes. En el próximo ensayo operativo el monitor se ejecuta en shadow mode: no cambia acciones.

### F4. Política ante fiabilidad

Entrada: forecast, acciones candidatas, costos de consulta/abstención/demora y constraints Et. Salida: continuar, replanificar, pedir dato, abstenerse o solicitar revisión, con utilidad esperada y acción final identificable. Si se cambia la acción, emitir una predicción vinculada a esa nueva acción o registrar explícitamente que no está calibrada. El baseline sigue siendo actuar sin intervención; solicitar review puede empeorar tracking y no es fallback físico gratuito.

## 4. Gate de admisión, no censura de resultados adversos

Para cada fila observada válida: admit = token_ok AND provenance_own AND excitation_ok AND measurement_valid AND not_censored. excitation_ok combina una amplitud mínima de acción normalizada con suficiente información frente al ruido calibrado. No dividir efecto por una acción cercana a cero. No rechazar solo porque el efecto sea grande, tenga signo desfavorable o reduzca el rendimiento.

El gate no accede a B verdadera, error futuro o rótulos de fuente del evaluador. Gate cerrado deja Bhat sin incorporar ese evento; conserva conteos/razones de exclusión y tratamiento explícito de la covarianza de olvido. Pulsos de identificación acotados y preprogramados son iguales en todos los brazos, para evitar que una estimación inicial nula bloquee para siempre la excitación. Fijar el tratamiento del cambio de régimen antes del final; no explicar después todo rechazo como ruido.

## 5. Orden obligatorio del ciclo

1. Cerrar el feedback pendiente de la acción anterior; no comenzar otra decisión con evidencia silenciosamente pendiente. Capturar snapshot anterior a actualización.
2. Verificar tokens, formar candidatos y atribuir procedencia usando ese snapshot; decidir gate de actualización y registrar razones.
3. Actualizar únicamente los coeficientes admitidos y el estado observado válido. La verdad externa usada para puntuar no se inyecta en el controlador.
4. Incorporar entradas exógenas autorizadas, versionar memoria cuando corresponda y recuperar candidatos; las copias activas tienen estado independiente.
5. Resolver reglas Et y seleccionar contenido Ct; registrar qué consumidor usa qué versión. No basta una variable calculada y no leída.
6. Calcular intención v y plan con Bhat, p, prioridades y horizonte H. En la lesión de efectos cruzados cambiar solo el uso en planning, no el estimador ni sus observaciones.
7. Aplicar la política ante fiabilidad si el protocolo específico la habilita; volver a chequear Et/recursos para la acción final.
8. Emitir forecast para esa acción ANTES del feedback, ejecutar u, conservar token y snapshot y actualizar p por su regla declarada. u nunca sobrescribe p por conveniencia.
9. El siguiente feedback reabre el paso 1. Duplicados, secuencias imposibles y obligaciones no factibles producen estados explícitos, no autoarreglos silenciosos.

La implementación puede organizar los mismos puntos mediante act()/learn(), pero debe conservar la causalidad temporal. Tests de contaminación y de invariancia deben comprobar el ciclo completo, no solo funciones aisladas.

## 6. Trazabilidad causal

Distinguir `received`, `read`, `computed`, `used_in_proposal` y `changed_executed_action`. Un hash coincidente acredita identidad del contenido, no efecto causal. Asociar content_id y memory_version con proposal_id/action_id, luego comprobar el contrafactual emparejado. Si una ruta se satura y no cambia u, conservar ese resultado, no modificar el ensayo después de observarlo. `total_inputs` u otra expresión no consumida es telemetría o código muerto; se prohíbe atribuirle una función por su nombre.
