"""Assemble the Spanish P1 execution report from final, verified summaries."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "delivery/P1_20260920"


def read(path):
    return json.loads((ROOT / path).read_text())


def interval(value, scale=100., digits=2, key="ci95"):
    lo, hi = value[key]
    return f"{value['mean']*scale:+.{digits}f} [{lo*scale:+.{digits}f}, {hi*scale:+.{digits}f}]"


def main():
    a = read("p1_results/final_A/P1_A_FINAL_SUMMARY.json")
    c = read("p1_results/final_C/P1_C_FINAL_SUMMARY.json")
    e = read("results/p1_ecology_final_20260920/SUMMARY.json")
    t = read("results/p1_tension_final_20260920/summary.json")
    index = read("delivery/P1_20260920/POLAR_P1_Indice_Evidencia_20260920.json")
    assert all(len(x['seeds']) == 30 for x in (a, c))
    assert e['seed_count'] == t['n_seeds'] == 30
    lines = [
        "**POLAR — P1: informe de ejecución, 30 semillas por bloque**",
        "20 de septiembre de 2026.",
        "Se completaron los cuatro bloques experimentales A, C, E y T. Hay respaldo para mecanismos concretos de transmisión, aprendizaje de consecuencias y propagación interna. **No se demuestra superioridad funcional general de POLAR; el umbral poblacional sigue sin identificarse y varios comparadores o ablaciones obtienen mejores resultados.** Completar esta campaña no equivale a cerrar favorablemente todas las brechas científicas.",
        "| Bloque | Trabajo ejecutado | Estado científico |",
        "|---|---|---|",
        "| A: composición y transmisión | 48 condiciones × 30 semillas; copia por éxito/conformidad, puntajes igualados y recursos redistribuidos | Intervenciones y controles válidos; p* sin identificar |",
        "| C: regulación local y población | 192 condiciones × 30 semillas; factorial 2×2×2 y cuatro controles adicionales | Ablaciones medidas; sin demostración de necesidad o superioridad general; p* sin identificar |",
        "| E: consecuencia ecológica y crédito | 122 880 ensayos de adquisición; 30 720 trayectorias diagnósticas; 600 episodios nativos | Predicción aprendida respaldada; mejora de supervivencia no demostrada |",
        "| T: tensión entre polaridades | 10 560 pares pulso/control; 660 episodios de tareas externas | Propagación causal configurada respaldada; utilidad frente a red apagada adversa |",
        "",
        "La campaña es descriptiva y prospectiva, con código congelado antes de ejecutar las semillas finales. No sustituye el diseño confirmatorio A/C histórico ni su análisis de potencia. La unidad de incertidumbre es la semilla; los pasos y agentes no se cuentan como réplicas independientes. Los IC95 de A/C/E son bootstrap pareado, marginales y sin corrección global por multiplicidad. T fija IC97,5 por comparación primaria dentro de una familia de dos tareas. No existe un PASS global por contar resultados favorables.",
        "**Composición y transmisión (A).** Supervivencia significa fracción de tiempo-agente vivo, promediada en las últimas cinco generaciones. Las diferencias siguientes promedian primero la grilla de ocho composiciones dentro de cada semilla. Los efectos están en puntos porcentuales (pp).",
        "| Intervención − base | Copia por éxito, pp [IC95] | Conformidad, pp [IC95] |",
        "|---|---:|---:|",
    ]
    for variant, label in [('copy_score_equal', 'Igualar puntaje de copia'), ('resource_pool', 'Redistribuir ingreso físico')]:
        vals = [interval(a['paired_contrasts'][f'{rule}/{variant}-minus-base']['grid_average']['survival_time_fraction']) for rule in ('success', 'conformity')]
        lines.append(f"| {label} | {vals[0]} | {vals[1]} |")
    lines += [
        "",
        "Igualar puntajes mantiene idéntica la física de la primera generación en 16/16 celdas pareadas; bajo conformidad mantiene idénticas todas las generaciones en 8/8. Esto distingue cambiar el criterio de copia de cambiar el ingreso material. La redistribución conserva el recurso cosechado e iguala el ingreso instantáneo entre agentes vivos, pero también modifica consecuencias ecológicas y transmisión posterior; no aísla por completo cada vía social y física ni garantiza ingresos de por vida iguales.",
        "**Regulación local y población (C).** La tabla conserva todos los contrastes de supervivencia frente a `full`. Un valor positivo favorece a la alternativa. La grilla y el presupuesto son los mismos en cada brazo; aprender el crítico y utilizarlo al decidir son intervenciones diferentes.",
        "| Alternativa − full | Copia por éxito, pp [IC95] | Conformidad, pp [IC95] |",
        "|---|---:|---:|",
    ]
    labels = {
        'qv_policy_off': 'Sin contribución de Qv a la decisión',
        'qv_learning_off': 'Sin aprendizaje de Qv',
        'short_vitality_discount': 'Factor de descuento de vitalidad reducido (γ=0,5)',
        'short_group_discount': 'Factor de descuento grupal reducido (γ=0,5)',
        'blind_exploration': 'Exploración ciega, misma tasa',
        'generic_shared_target': 'Comparador con acción sucesora común',
        'coordinate_equivalent': 'Coordenadas equivalentes',
        'anchor_off_short_vitality': 'Qv sin uso + factor de descuento reducido',
        'anchor_off_blind': 'Qv sin uso + exploración ciega',
        'short_vitality_blind': 'Factor de descuento reducido + exploración ciega',
        'anchor_off_short_vitality_blind': 'Las tres intervenciones',
    }
    for variant, label in labels.items():
        vals = [interval(c['paired_contrasts'][f'{rule}/{variant}-minus-full']['grid_average']['survival_time_fraction']) for rule in ('success', 'conformity')]
        lines.append(f"| {label} | {vals[0]} | {vals[1]} |")
    statuses_a = {k: v['p_star']['status'] for k, v in a['summaries'].items()}
    statuses_c = {k: v['p_star']['status'] for k, v in c['summaries'].items()}
    assert all(s != 'within_support' for s in list(statuses_a.values()) + list(statuses_c.values()))
    lines += [
        "",
        "En este banco, retirar el uso decisional de Qv o compartir la acción sucesora mejora la supervivencia respecto de `full`. No queda respaldada la necesidad conjunta de los componentes. La interacción triple de supervivencia es +0,14 pp [−0,48, +0,77] por éxito y +0,66 pp [−0,16, +1,45] por conformidad; ambos intervalos incluyen cero. Las coordenadas equivalentes mantienen exactamente las trayectorias en las 16 comparaciones de grilla y transmisión.",
        "El factorial cruza uso de Qv, descuento de vitalidad y exploración dirigida/ciega. La fracción inicial de C es un prior sobre Q, no un tipo de agente permanentemente forzado; el linaje y la restricción observada se registran por separado. Las diferencias por clase conductual son asociaciones endógenas. Cambiar γ no identifica un horizonte de planificación finito ni demuestra adaptación óptima a escalas temporales distintas.",
        f"**El umbral p* no está identificado en {len(statuses_a)}/{len(statuses_a)} perfiles de A y {len(statuses_c)}/{len(statuses_c)} de C.** Se ensayó p entre 0,35 y 0,70. Los perfiles no ofrecen un cruce interior monótono válido del criterio de viabilidad predefinido; hay curvas no monótonas y/o viabilidad ya alta en el extremo inferior. Por tanto, sus diferencias y diferencias de diferencias tampoco reciben una estimación válida. No se extrapola, suaviza ni usa un intervalo condicionado a escasos remuestreos válidos como si el umbral estuviera identificado. Tampoco se demuestra ausencia universal de transición, cuencas asintóticas o histéresis.",
        "**Aprendizaje ecológico y crédito (E).** ID es el entorno de adquisición; OOD cambia crecimiento e inventario inicial. El predictor recibe únicamente recurso observado, vitalidad propia y acción. Cada ensayo factual ejecuta una sola acción inicial aleatorizada. Las cuatro acciones contrafactuales se utilizan sólo para diagnóstico, con continuación fijada. El costo es la diferencia de recurso futuro medio a 12 pasos frente a acción cero.",
        "| Predictor/control | MSE causal ID | MSE causal OOD |",
        "|---|---:|---:|",
    ]
    for key, label in [('aligned','Consecuencia aprendida, crédito correcto'), ('zero_cost_or_action_free','Sin acción / costo marginal cero'), ('shifted_credit','Crédito desplazado entre ensayos'), ('generic','Predictor convencional, 20 parámetros'), ('immediate','Consecuencia inmediata aprendida'), ('physical_immediate_oracle_DIAGNOSTIC','Consecuencia inmediata verdadera, sólo diagnóstico')]:
        vals = [e['mechanism'][d]['model_mse_means'][key] for d in ('id', 'ood')]
        lines.append(f"| {label} | {vals[0]:.6f} | {vals[1]:.6f} |")
    lines += [
        "",
        "Asignar correctamente el crédito supera al desplazamiento en 30/30 semillas de ambos dominios. El control conserva exactamente los estados iniciales, observaciones, acciones, trayectorias y distribución de resultados por estado; cambia únicamente la asignación entre ensayo y resultado. Identifica crédito entre ensayos reiniciados, no una modificación de la demora física de regeneración. Tanto el predictor alineado como su rival son regresiones ridge convencionales de 20 coeficientes; cambia la base de características, no la clase general de aprendizaje.",
        "En la interacción nativa se congelan los mismos valores Q y se evalúa el aporte adicional de la señal aprendida. El predictor se adquirió con continuación fijada y luego se aplica a todos los agentes, por lo que hay cambio de distribución. La preferencia por conservar recursos y su peso λ=3 son impuestos por el diseñador.",
        "| Alineado − sin head | ID, pp [IC95] | OOD, pp [IC95] |",
        "|---|---:|---:|",
    ]
    for key,label in [('alive_fraction','Supervivencia, primaria'),('resource_fraction','Recurso medio, secundaria')]:
        vals = [interval(e['native'][d]['aligned_minus_no_head'][key],key='bootstrap95') for d in ('id','ood')]
        lines.append(f"| {label} | {vals[0]} | {vals[1]} |")
    lines += [
        "",
        "Ambos intervalos de supervivencia incluyen cero: no queda demostrada una mejora primaria. Frente al crédito desplazado, la supervivencia OOD cae 9,35 pp [−16,46, −2,53], pese a predecir mejor las consecuencias. La comparación nativa contra el predictor convencional tampoco identifica superioridad en supervivencia. La combinación secundaria de mitad supervivencia y mitad recursos mejora por los pesos elegidos, pero no reemplaza el resultado primario. El oráculo inmediato OOD supera al predictor aprendido; usa información verdadera excluida del aprendizaje y se conserva como diagnóstico de sus límites.",
        "**Propagación de tensión (T).** La intervención local alcanza 5 040/5 040 destinos esperados, sin efectos fuera de las rutas ni llegadas prematuras. Apagar la red elimina el efecto cruzado; fijar el mediador τ a su valor de control reduce el efecto cruzado L1 medio de 0,07786 a 0,03006, dejando la contribución de W. Lesionar, invertir y permutar conexiones da respuestas compatibles con la topología. Las matrices W/K fueron configuradas, no aprendidas.",
        "| Tarea externa | Regret full − red apagada [IC97,5] | Lectura |",
        "|---|---:|---|",
    ]
    for key,label in [('switching_memory','Memoria con cambios'),('gain_resource_shift','Cambio de ganancia y recursos')]:
        v = t['paired_regret_full_minus_comparator'][key]['no_network']
        lines.append(f"| {label} | {v['mean_difference']:+.6f} [{v['ci_low']:+.6f}, {v['ci_high']:+.6f}] | Peor con W/K |")
    lines += [
        "",
        "El perjuicio es pequeño y consistente; no alcanza el umbral práctico predefinido de 0,005, pero eso no significa ausencia de perjuicio. La red tampoco cumple el criterio de beneficio en ambas tareas. No hay violaciones de restricciones. La formulación convencional algebraicamente equivalente coincide exactamente; las coordenadas invertibles sólo difieren por redondeo (máximo 1,28×10⁻¹⁴). El comparador recurrente diferente ofrece una comparación secundaria, no una demostración de exclusividad representacional.",
        "**Integridad y reproducibilidad.** Se aprobaron 32 pruebas nuevas de contratos causales antes de ejecutar finales. A/C preservan episodios, transmisiones, estados finales y trazas diagnósticas; E/T conservan las trazas de evaluación. Hay verificaciones de hashes, reconstrucciones desde datos crudos y reproducciones exactas preseleccionadas. Los informes de auditoría detallan sus conteos y tolerancias.",
        "La auditoría detectó contenedores dañados. En T, cuatro NPZ truncados se recuperaron con exactamente los SHA256 del manifiesto original, y se verificó el conjunto lógico de 26 originales válidos más cuatro recuperados. En C, siete NPZ dañados se reconstruyeron dentro de siete celdas: sus 63 artefactos registrados reprodujeron exactamente los SHA256 originales; todos los endpoints por semilla originales ya eran válidos. En A, una traza auxiliar ya era inválida cuando se registró su checksum: se reconstruyó por separado y pasó 38 controles contra los resultados originales de las 30 semillas, hashes de transiciones y estados; no se afirma identidad con sus bytes originales. Los resultados principales de A permanecieron intactos. Se preservaron todos los archivos dañados, manifiestos y reportes del incidente; no se añadieron réplicas independientes ni se cambiaron fuentes científicas. La causa de los daños no está determinada. Para trabajar con trazas válidas deben utilizarse las recuperaciones identificadas, siguiendo sus informes.",
        "Las semillas finales son A 935001–935030, C 932001–932030, E 933001–933030 y T 934001–934030. Los pilotos usan 930xxx. Un ensayo de rendimiento previo utilizó 931001–931030 sin conservar ni inspeccionar resultados: se declaró expuesto y se excluyó por completo. No se reajustaron parámetros después de observar los resultados finales.",
        "Código científico y diseño: [commit 2c63f3df6365751af1331eb75cfd16882f88cc44](https://github.com/CRC2520/polar-sim-ml-original/commit/2c63f3df6365751af1331eb75cfd16882f88cc44), rama `research/p1-completion-20260920`. El repositorio es privado; se trata de congelación prospectiva versionada, no de preregistro público externo. El motor previo conserva SHA256 `48330c1a88a2312182bb488dae0096e2ccf674c6fe2b0ca7ddf6fa9b8a29dccc`.",
        "Desde una copia del repositorio en ese commit: `python -m p1_completion.provenance` verifica las fuentes; los diseños incluyen los comandos de ejecución. Python 3.12.14 y NumPy 2.3.5. Los datos no se mezclan con P2: sigue vigente su resultado de supervivencia inferior a su comparador convencional en las ocho condiciones de transferencia.",
        "**Brechas que permanecen.** Falta localizar una transición poblacional con un diseño y soporte adecuados; demostrar utilidad primaria robusta y transferible del costo ecológico; demostrar ventaja funcional de W/K; y evaluar topología aprendida frente a impuesta. E y T son bancos experimentales separados: no integran ambas extensiones en un mismo agente. Esta campaña no evalúa ni acredita conciencia fenomenológica. Los resultados adversos acotan las afirmaciones del modelo y deben acompañar cualquier síntesis posterior.",
        "**Archivos de evidencia.** Cada ZIP es independiente. Para reconstruir las carpetas, extraer las partes correspondientes en un mismo directorio. El índice JSON enumera cada archivo y su SHA256; las observaciones originales y las recuperaciones se distinguen expresamente. Los ZIP de pilotos no forman parte de los 30 casos finales.",
        "| Archivo | Tamaño MB | Contenido |",
        "|---|---:|---|",
    ]
    for item in index['archives']:
        path = OUT / item['filename']
        lines.append(f"| [{path.name}](sandbox:{path}) | {item['bytes']/1e6:.1f} | {item['front']} |")
    lines += ["", f"[Índice completo de integridad](sandbox:{OUT / 'POLAR_P1_Indice_Evidencia_20260920.json'}) · [Hashes de los paquetes](sandbox:{OUT / 'POLAR_P1_Evidencia_20260920.sha256'})", ""]
    (OUT / "POLAR_P1_Informe_30_semillas_20260920_ES.md").write_text("\n\n".join(lines).replace("|\n\n|", "|\n|"))


if __name__ == '__main__':
    main()
