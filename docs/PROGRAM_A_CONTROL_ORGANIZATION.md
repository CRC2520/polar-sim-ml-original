# Programa A — Organización relacional para control interpretable

**Reconstrucción documental v0.1**  
**Fecha de corte:** 7 de septiembre de 2026  
**Base de código:** `05caf7abef6fdcb769604539a22cdcd1cfd6b397`  
**Alcance de este cambio:** documental. No modifica simuladores, generadores, seeds, protocolos congelados ni resultados históricos.

## 1. Pregunta del Programa A

> ¿Una organización explícita de canales conectados por una relación anclada `Pi`, con acceso dinámico a los canales o estados relacionados, mejora el control adaptativo frente a comparadores cuyos ejes causalmente relevantes estén emparejados cuando el claim lo requiera y cuyas diferencias no emparejables estén cuantificadas y limitadas explícitamente?

En esta reconstrucción, **“polos” se reserva para realizaciones binarias con referente nativo**. `Pi` es una notación de auditoría más general para relaciones entre pares o tuplas. Una relación n-aria futura sería una extensión; no reescribe retrospectivamente la arquitectura histórica de pares/polos.

Los resultados del Programa A no se interpretan como indicadores de consciencia. La infraestructura podrá reutilizarse en un futuro Programa B, pero no se transfieren automáticamente resultados, significado ni nivel de evidencia.

## 2. Contrato metodológico mínimo

| Objeto | Rol | No debe identificarse con |
|---|---|---|
| `W` | estructura regulatoria basal declarada entre canales | `B_hat`, una traza post-plan o una matriz necesariamente fija en todo experimento |
| `M(C_t)` | modulación por contenido de relaciones/aristas declaradas | `K*tau`, el allocator de recursos o `K_effective` |
| `B_hat` | efectos acción–estado aprendidos cuando aplica | evidencia de organización polar, `W` o aprendizaje del significado de `Pi` |
| `Pi` | relación declarada entre canales | partición exhaustiva, soporte de `W/K` o factibilidad de acción |
| Máscara de permisos | acciones permitidas en un paso | conjunto admisible completo |
| `U_t` | conjunto admisible de acciones | dinámica ambiental |
| `P_t` | operador concreto que aplica restricciones | una ley universal de “proyección polar” |
| `F_t(x_t,u_t,epsilon_t)` | transición ambiental | factibilidad de acción |

La recodificación invertible `(p+,p-) <-> (a,q)` es un **control de equivalencia**, no un mecanismo nuevo ni un rival que la versión polar deba superar.

### 2.1 `tau` de Study 3

Para no psicologizar la cantidad, esta reconstrucción usa el nombre funcional:

`indice_de_desajuste_y_conflicto_declarado`

conservando sus dos componentes:

```text
tau_i = r_i + h_i
r_i   = 0.5 * (|v_i+ - p_i+| + |v_i- - p_i-|)
h_i   = chi_i * p_i+ * p_i-
```

`r_i` es desajuste de esfuerzo. `h_i` es conflicto por coactivación únicamente bajo el `chi_i` declarado. Ninguno es, por nombre, una medición psicológica.

## 3. Estudios archivados: mecanismo, estatus, resultado y límite propios

Esta tabla **no fusiona S1–S3/P0–P2 en un único aparato**.

| Estudio | Estatus documental | Mecanismo núcleo | Resultado archivado | Límite de inferencia |
|---|---|---|---|---|
| **S1** | Preliminar/exploratorio. El diagnóstico archivado reconoce inspección previa parcial de datos y ausencia de preregistro externo independiente. | Estimador de gain tipo EMA, memoria por cue y asignación de recursos. El campo `horizon` escala tasa de respuesta; no es horizonte predictivo. | En el diagnóstico de recall ya observado: original `0.012783` regret y `28/30` passes; `confidence=1`, `0.003046` y `30/30`; sin capability model `0.010688` y `30/30`; freeze del estimator durante recall `0.020983` y `0/30`. En gain-shift: full `0.004625` vs `0.013432` sin capability model, `30/30` vs `29/30`. | Evidencia exploratoria. El efecto agregado cuantitativo del workspace de recursos no fue recuperado de forma independiente en esta reconstrucción: `no_determinado`. |
| **S2** | Freeze prospectivo Git+hashes anterior a resultados finales; no equivale a OSF, peer review ni replicación externa. | RLS con organización por bloques/canales y planificador de horizonte finito genuino. La lesión paired retira cross-effects de planning. Hay shuffled de misma sparsidad y dense. | Paired vs lesion: `Delta=-0.0026998025`, IC95% `[-0.0028285572,-0.0025766470]`, superando el margen práctico registrado `0.002` y guardrails. Paired vs shuffled: `+0.0001955151` (shuffled mejor). Paired vs dense en transferencia: `+0.0014982882` (dense mejor). Adjudicación posterior: `suspend_exclusive_polar_advantage_claim`. | Respalda el uso de cross-effects en esa realización; shuffled y dense impiden atribuir una ventaja exclusiva al emparejamiento nominado. |
| **S3 principal** | Freeze prospectivo antes de datos finales; comparte desarrollo y generadores con el panel activo. | Red configurada con `W,K` fijos para el estudio, escalados por ganancias `a,b`, frente a scaffold sin grafo. | La selección de desarrollo eligió `a=b=0` en los modelos de red relevantes; los contrastes de superioridad quedaron sin apoyo. Decisión: `no_confirmed_network_advantage`. | Con la vía seleccionada en cero, el final principal no identifica el beneficio causal de una vía de red activa; tampoco demuestra equivalencia universal de redes activas. |
| **S3 panel activo** | Registro separado, pero no replicación independiente: comparte desarrollo y generador con S3 principal; usa seeds finales diferentes. | Misma familia con `K` activo y `W=0` en las configuraciones seleccionadas. | Full activo MSE `0.0112738429` vs `0.0103783489` sin `K`/scaffold; diferencia `+0.0008954940`, aproximadamente `+8.6%`. No acredita la ventaja mínima frente a rewired/reverse. Decisión: `no_confirmed_network_advantage`. | Evidencia adversa para esa realización activa y condiciones. No atribuir la causa a un submecanismo no aislado. |
| **P0** | Diagnóstico exploratorio factorial sobre la familia/controlador de Study 3. | Cuatro factores binarios: información firmada, signo receptor, modelo/inversa acoplada y proyección por prioridad. `2,448` runs. | Efectos principales MSE: información `-0.0006816667`; signo receptor `+0.0005084373`; modelo/inversa `+0.0014936174`; prioridad `-0.0001618714`. | Cada efecto corresponde al **factor implementado completo**. Un factor compuesto no queda automáticamente descompuesto en sus submecanismos. |
| **P1** | Evaluación prospectiva después del freeze de desarrollo; 24 seeds y 1,056 runs. | Prototipo parcial RLS + MPC restringido; cuatro contrastes direccionales fijados. | Pasan 2/4: prioridades `Delta=-0.0014937975`; memoria retenida en retorno `Delta=-0.0462574062`. No pasan: predicción acoplada `Delta=-0.0004528381` bajo el bound registrado; horizonte temporal/post-change `Delta=+0.0000875298`. Decisión: `engineering_verified_with_partial_empirical_support`. | Fallar un umbral no demuestra efecto cero, equivalencia ni refutación de toda la arquitectura. |
| **P2** | Capacidades delimitadas sobre seeds separados. | Atribución entre candidatos suministrados, abstención por ambigüedad, reglas institucionales, coordinación limitada, monitor y policy de review. | Atribución `278/288=96.53%`; abstención ambigua `100%`; coordinación bid vs equal-share `Delta MSE=-0.0280323732`; monitor Brier `0.0440622` vs base constante `0.2273543`, ECE `0.0132069`. La política de review empeora MSE en `+0.0732958574`. | Candidatos suministrados no implican autenticación o source discovery abierto. Calibración no vuelve útil a la policy de review. La coordinación es local/de recursos, no negociación amplia. |

## 4. Discrepancia histórica de Study 2

No se modifica la evidencia numérica. Se separan cuatro objetos:

1. **Resultado numérico:** artefacto congelado y su análisis.
2. **Regla prospectiva:** protocolo/freeze anterior a datos.
3. **Implementación de la regla:** evaluador congelado.
4. **Adjudicación posterior:** interpretación posterior que conserva los números y registra `suspend_exclusive_polar_advantage_claim`.

Una adjudicación posterior no sustituye silenciosamente la regla prospectiva ni los resultados congelados.

## 5. Relaciones históricas: identificadores de auditoría

Se introducen dos nombres **editoriales**, no nombres históricos certificados:

- `Pi_S2_nom`: organización nominal por bloques/canales de Study 2.
- `Pi_S3_aligned`: relación aligned/ring de Study 3.

No prueban identidad polar común. No se usa un `Pi_A` genérico para absorber S2, S3, P1 o futuras relaciones.

## 6. Hipótesis futura del Programa A: seis campos obligatorios

Una nueva hipótesis de organización **no está lista para contraste confirmatorio**.

| Campo | Estado actual |
|---|---|
| **1. Relación** | `no_determinada` |
| **2. Elementos** | `no_determinada` |
| **3. Dominio de tareas** | `no_determinada` |
| **4. Intervención e invariantes** | `no_determinada` como campo principal |
| **5. Comparadores** | `no_determinada` |
| **6. Predicción y decisión** | `no_determinada` |

Antes de cerrar el campo 4 deben inventariarse por separado: información; restricciones/factibilidad; perfil y recursos de aprendizaje; memoria; dinámica de respuesta/planificación; cómputo de decisión.

El mismo número de parámetros no demuestra capacidad equivalente. Se reportarán por separado **recursos/perfil de aprendizaje** y **cómputo de decisión**.

Si `Pi` y `U_t`/factibilidad cambian simultáneamente, el contraste identifica el **cambio conjunto**, no el efecto exclusivo de `Pi`.

## 7. Agenda futura — no ejecutada por este cambio

- anclar una relación `Pi` en elementos nativos antes de formular una ventaja;
- usar comparadores competitivamente entrenados cuando el claim lo requiera;
- incorporar al menos una familia de tareas creada por terceros;
- descomponer factores compuestos de P0 con intervenciones de un solo mecanismo;
- medir robustez, costo, transferencia y compute como resultados externos;
- integrar atribución de procedencia al loop primario solo bajo nueva especificación.

Todo ello requiere protocolos y corridas nuevas. **No se reabren ni reescriben los estudios archivados.**

## 8. Procedencia

| Objeto | Fuente | Identificador estable | Estado_fuente |
|---|---|---|---|
| Propuesta original `O` | `Polar_Dynamics_of_Consciousness_arXiv_v2.0.pdf` | SHA-256 `e124988b78e6708d753d9c80775f8c7589253a0554e35994921caa4cfd8bf20e`; commit Git `no_determinado` | `ok` para el PDF; VCS `no_determinado` |
| CODE examinado | `CRC2520/polar-sim-ml-original` | `05caf7abef6fdcb769604539a22cdcd1cfd6b397` | `ok` |
| S1 | `docs/STUDY1_DIAGNOSIS.md` + contextual manifest | bajo CODE anterior | `ok`, salvo efecto agregado de workspace indicado como `fuente_insuficiente` |
| S2 | `docs/STUDY2_CONTROLLER_SPEC.md`, freeze y adjudication | bajo CODE anterior | `ok` con separación de objetos históricos |
| S3 | protocol, freezes y resultados | artefactos congelados del repositorio | `ok` |
| P0–P2 | `docs/P0P2_PROTOCOL.md`, `docs/P0P2_FREEZE.json`, `results_p0p2/` | freeze `34ce5268e614b49b8910a355d7de603689b7431e`; resultados `1bda1e22364ba068eb8b14289512b0776b4242bd` | `ok` |

## 9. Regla de inferencia

Los estudios existentes acotan realizaciones concretas del Programa A. Un resultado adverso restringe la realización que lo produjo; uno favorable tampoco se generaliza automáticamente.

**Ningún resultado de este programa se convierte por sí mismo en evidencia de consciencia.**
