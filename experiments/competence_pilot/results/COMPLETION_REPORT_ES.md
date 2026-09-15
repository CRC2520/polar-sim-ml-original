# Piloto de competencia CP-NB-1.0.0

**PILOT_COMPLETE_FOR_REVIEW**

Desarrollo nuevo: no confirma estructura, no modifica B1-S v1.1 y no habilita B1-E.

| Rol | Entrada | Pasos | Pérdida | Desplazamiento adicional | Competencia agregada | Bloques /3 |
|---|---|---:|---:|---:|---|---:|
| G0 | raw | 524288 | -1.009997 | -1.2376233 | False | 0 |
| G0 | raw | 1048576 | 10.534768 | 6.0297826 | True | 3 |
| G0 | standardized | 524288 | 2.7013265 | -1.4715244 | False | 0 |
| G0 | standardized | 1048576 | 8.8395536 | 1.8083547 | True | 1 |
| GD | raw | 524288 | 17.800612 | 2.3954063 | True | 1 |
| GD | raw | 1048576 | 20.415457 | 7.0294077 | True | 3 |
| GD | standardized | 524288 | 0.44247915 | -1.3670051 | False | 0 |
| GD | standardized | 1048576 | 13.98369 | -0.18380461 | False | 1 |
| PPO | raw | 524288 | -2.2035037 | 0.039004703 | False | 0 |
| PPO | raw | 1048576 | -2.4569782 | 2.8355298 | True | 3 |
| PPO | standardized | 524288 | 0.33729417 | -0.29687875 | False | 0 |
| PPO | standardized | 1048576 | -4.2858858 | 1.5438456 | True | 2 |
| SAC | raw | 524288 | 1.9689796 | -1.4006209 | False | 0 |
| SAC | raw | 1048576 | 21.26812 | -1.7793518 | False | 0 |
| SAC | standardized | 524288 | -0.14580488 | -1.6123996 | False | 0 |
| SAC | standardized | 1048576 | 2.2612359 | -1.6096474 | False | 0 |

Tres bloques: precisión limitada. Los presupuestos son checkpoints anidados, no réplicas adicionales.
El normalizador se fijó con datos de calibración de entradas, no con evaluación; no se normalizaron recompensas.
Los contrastes y las medias por bloque están en SUMMARY.json. Se conservan resultados adversos e inconclusos.
B1E_disposition=ON_HOLD_FOR_CURRENT_STRUCTURAL_CONFIRMATION; ready_for_b1e_confirmatory_run=false.
Los binarios tienen retención temporal en Actions (90 días); ver inventario y hashes. No se promete permanencia en Git.
