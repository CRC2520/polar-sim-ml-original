# Dictamen de fidelidad arquitectónica y siguiente protocolo

Estado: **propuesta de revisión; ningún nuevo ensayo empírico ejecutado**. Este paquete sustituye como siguiente diseño confirmatorio al borrador preliminar de `research/iteration2-causal-regulation`, sin borrar ese desarrollo ni modificar resultados anteriores.

Base examinada: código `05caf7abef6fdcb769604539a22cdcd1cfd6b397`; manuscrito `b23507cf80ca79ca0ac16e4b36ca9f5a535ff1e5`. El objeto sigue siendo el manuscrito inédito de Carlos Rodriguez Castro, draft 6 Sep 2026. Este paquete no es otro artículo.

Archivos:
- `FIDELIDAD_Y_PLAN.md`: correspondencia con las ecuaciones organizadoras originales, estado por nivel, hipótesis arquitectónica y ensayos desagregados.
- `CONTRATOS.md`: dominios, secuencia de actualización, procedencia, memoria/caché, Ct, Et y cuatro funciones del automodelo.
- `PROTOCOLO_PROSPECTIVO.md`: borrador del siguiente experimento operativo (pasos 2–3), no una confirmación ya registrada o ejecutada.
- `protocol_parameters.json`: parámetros propuestos y condiciones que bloquean la ejecución final.
- `decision_rule.py` y `test_decision_rule.py`: adjudicación literal y pruebas de frontera con entradas artificiales. Pasarlas no valida un controlador ni una hipótesis empírica.
- `ALCANCE_MANUSCRITO.tex`: párrafos propuestos para Abstract, §1 y discusión/conclusión; no insertados automáticamente en el PDF.

El protocolo aún requiere completar y auditar el adaptador de mediciones contaminadas, el generador exacto, el estimador de intervalos, el registro de seeds ya observadas y la versión final de código. Una publicación de este borrador NO basta para afirmar que esos elementos ya están congelados. Ningún runner de datos finales forma parte de este paquete.

No se modifican Studies 1–3, panel activo, P1/P2 ni sus decisiones. El trabajo pendiente es precisar qué se realiza, aproxima, sustituye o introduce como hipótesis nueva, no declarar toda la arquitectura refutada ni validada.
