"""Build the Spanish R9 delivery note from closed, audited evidence."""
import argparse
import json
from pathlib import Path
from r8_completion.io import atomic_text
from r9_completion.provenance import verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulation-commit', required=True)
    parser.add_argument('--manuscript-commit', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    results = root/'r9_results'
    analysis = results/'final/analysis'
    def read(path):
        return json.loads(path.read_text())
    audit = read(results/'final/AUDIT_FINAL.json')
    primary = read(analysis/'R9_CONFIRMATION.json')
    desc = read(analysis/'R9_DESCRIPTIVES.json')
    diag = read(analysis/'R9_DIAGNOSTICS.json')
    support = read(analysis/'R9_SELECTION_SUPPORT_BUDGET.json')
    physics = read(results/'final/PHYSICS_REPLAY_952001.json')
    metadata = read(analysis/'R9_COMPILATION_METADATA.json')
    if not audit['all_checks_passed'] or audit['seeds_audited'] != 80:
        raise ValueError('Complete independent audit required')
    if not physics['all_exact'] or not (analysis/'COMPLETE.json').is_file():
        raise ValueError('Missing physical replay or completed compilation')
    for left, right in zip(audit['raw_reconstruction']['rows'], primary['rows']):
        for key in ('claim', 'wins', 'n', 'p_holm', 'supported', 'seed_success'):
            if left[key] != right[key]:
                raise ValueError('Independent inference mismatch: '+key)
    verify()
    wins = {r['claim']: r['wins'] for r in primary['rows']}
    off = support['by_kind']['full']['gate_selection_counts'].get('off', 0)
    active = support['by_kind']['full']['numeric']['active_slope_coefficients']['mean']
    confirmed = sum(r['supported'] for r in primary['rows'])
    lines = [
        '# POLAR R9: ocho ajustes y resultados finales', '',
        f'Los ocho ajustes de diseño están implementados. Pasaron 62 contratos previos a finales; '
        f'la campaña incluyó 80 semillas completas, 320 controladores con adquisición propia, '
        f'10 240 episodios y 3 276 800 pasos. **{confirmed}/6 hipótesis primarias recibieron apoyo** '
        'bajo los criterios previstos. Implementación correcta y superioridad experimental son conclusiones distintas.', '',
        'R9 es una nueva realización acotada. Conserva R8 y la arquitectura teórica de 2025 como '
        'antecedentes explícitos; no es una reproducción numérica de R8 ni la implementación completa '
        'de la arquitectura cognitiva original.', '',
        '## Ocho ajustes', '',
        '| # | Corrección aplicada | Verificación y alcance |',
        '|---|---|---|',
        '| 1 | Separación de funcionamiento, causalidad computacional, utilidad, transferencia y conciencia. | Los contratos no se presentan como evidencia de conciencia. |',
        '| 2 | Distinción H_REL, H_CAT y H_TRANSFER. | Contrastes operativos fijados; correspondencia del catálogo filosófico sigue sin validación independiente. |',
        f'| 3 | Soporte predictivo disperso aprendido, frente a soporte fijo y ajuste denso. | Media de {active:.2f} pendientes activas; no identifica un grafo causal ambiental. |',
        f'| 4 | Tensión con error, coactivación, oposición predicha e incertidumbre empírica. | Contraste vectorial–escalar: {wins["TENSION"]}/80 éxitos conjuntos. |',
        f'| 5 | Compuerta contextual con apagado exacto en ambas pasadas. | Off seleccionado en {off}/80; {diag["off_gate_exact_equivalent_episodes"]}/{diag["off_gate_native_episodes"]} episodios apagados idénticos a noCross. |',
        f'| 6 | Mensajes tipados, dos consumidores, memoria H1/4/12 y metas persistentes en un agente. | CONTENT: {wins["CONTENT"]}/80; las lesiones de memoria/metas se reportan como secundarias. |',
        '| 7 | Descenso con signo negativo, regularización L1 proximal y escala de paso explícita. | Contratos y auditoría de pérdidas, coeficientes, soporte y normalización. |',
        '| 8 | Comparadores con experiencia propia, presupuesto 8+2+7, límites nativos y nuevas familias. | Cuatro dominios y quince variantes; la transferencia conjunta no se confirmó. |', '',
        '## Contrastes primarios', '',
        'Cada éxito exige superar el margen de recompensa en **todas** las condiciones de la hipótesis, '
        'mantener una fracción de vida ≥0.80 y no perder más de 0.005 frente al comparador. '
        'Se contrasta Pr(éxito por semilla)>0.5, con prueba binomial exacta y Holm sobre las seis hipótesis. '
        'Los 10 240 episodios no se cuentan como réplicas independientes.', '',
        '| Hipótesis | Semillas que cumplen todo | p ajustado por Holm | Apoyo |',
        '|---|---:|---:|---|']
    labels = {'REL':'Relaciones frente a noCross', 'GATE':'Compuerta contextual',
              'TENSION':'Tensión vectorial', 'CONTENT':'Correspondencia del contenido',
              'GENERIC':'Ventaja frente a denso', 'TRANSFER':'Transferencia a ambas familias'}
    for row in primary['rows']:
        lines.append(f'| {labels[row["claim"]]} | {row["wins"]}/{row["n"]} | {row["p_holm"]:.4f} | '+('Sí' if row['supported'] else 'No')+' |')
    lines += ['', 'No obtener apoyo no demuestra equivalencia ni refuta por sí solo la teoría general.', '',
              '## Resultados descriptivos', '',
              '| Dominio | Recompensa R9 / denso | Fracción viva R9 / denso | Supervivientes al paso 320, R9 / denso |',
              '|---|---:|---:|---:|']
    names = {'ecology_train':'Ecología original', 'ecology_delay9':'Ecología, retardo 9',
             'inventory_transfer':'Inventario', 'thermal_transfer':'Microgrid térmico'}
    for domain in metadata['domains']:
        f, d = desc['native'][domain]['full'], desc['native'][domain]['dense']
        lines.append(f'| {names[domain]} | {f["reward_mean"]["mean"]:.4f} / {d["reward_mean"]["mean"]:.4f} | '
                     f'{f["alive_fraction"]["mean"]:.4f} / {d["alive_fraction"]["mean"]:.4f} | '
                     f'{round(80*f["terminal_survival"]["mean"])}/80 / {round(80*d["terminal_survival"]["mean"])}/80 |')
    eco = desc['variant_minus_full']['ecology_train']['dense']['reward_mean']
    inventory = desc['native']['inventory_transfer']
    lines += ['',
        f'En ecología original, la diferencia media R9−denso fue **{(-eco["mean"]):+.4f}** '
        f'(IC descriptivo 95% [{(-eco["upper"]):+.4f}, {(-eco["lower"]):+.4f}]). '
        'Este efecto medio acotado no satisface el criterio de ventaja robusta en ambos dominios ecológicos. '
        'Los intervalos descriptivos son marginales, no están ajustados por multiplicidad.', '',
        f'La limitación más grave fue inventario: sólo {round(80*inventory["full"]["terminal_survival"]["mean"])}/80 agentes completos '
        'sobrevivieron hasta el final. Congelar las metas desde el inicio elevó descriptivamente la fracción viva '
        f'de {inventory["full"]["alive_fraction"]["mean"]:.4f} a {inventory["noGoalRevision"]["alive_fraction"]["mean"]:.4f}; '
        'la lesión también cambia la formación inicial de pesos y no identifica exclusivamente el efecto de revisiones posteriores.', '',
        'Permutar contenido empeoró la recompensa ecológica pero la mejoró en el dominio térmico. '
        'Por tanto, la correspondencia diseñada no puede presentarse como útil de forma general. '
        'La frecuencia de operaciones de memoria o metas tampoco prueba por sí sola su utilidad.', '',
        '## Alcance y brechas que permanecen', '',
        '- La ecología tiene poco margen de mejora: el piloto ya mostró rendimiento cercano al techo. '
        'Debe diseñarse un banco futuro más discriminante antes de congelar nuevos ensayos, sin cambiar estos endpoints retrospectivamente.',
        '- La puerta suele quedar apagada o constante; aprende un contraste de error predictivo crudo a X fijo, '
        'que no equivale al beneficio de la política completa con tensión recalculada.',
        '- La transferencia a inventario y la selección de metas necesitan revisión de diseño. '
        'El filtro predictivo no garantiza viabilidad física.',
        '- Igual presupuesto y 784 coeficientes asignados no implican igual capacidad efectiva. '
        'Los generadores son internos; no hubo replicación externa ni validación del catálogo psicológico o de conciencia.', '',
        '## Integridad, fuentes y reproducción', '',
        f'La auditoría independiente aprobó {audit["rollouts_audited"]:,} episodios y reconstruyó los seis vectores de decisión '
        'estadística desde trazas. La compilación verificó 32 400 archivos por SHA256 y 10 240 NPZ por CRC. '
        'El replay físico de la semilla 952001 coincidió exactamente en 128 episodios y 40 960 transiciones. '
        'R8 y las 76 fuentes de la clausura R9 permanecieron intactos. Se conservan los tres pilotos y su fuente original.', '',
        'Fuente publicada antes de finales: [6f3f739](https://github.com/CRC2520/polar-sim-ml-original/commit/6f3f739a029cbbab91a00b744bbf562f3bcd3991).', '',
        f'Cierre del simulador y utilidades: [{args.simulation_commit[:7]}](https://github.com/CRC2520/polar-sim-ml-original/commit/{args.simulation_commit}). '
        f'Manuscrito integrado: [{args.manuscript_commit[:7]}](https://github.com/CRC2520/POLAR_MODEL_CRC/commit/{args.manuscript_commit}).', '',
        'Los datos, checkpoints, auditorías y resúmenes están en los ZIP POLAR_R9_EVIDENCE_*.zip. '
        'R9_EVIDENCE_INDEX.json asigna cada archivo a su ZIP; R9_EVIDENCE_CHECKSUMS.sha256 verifica las partes. '
        'El código se obtiene de Git. Extraer las partes en la raíz del repositorio restaura r9_results; '
        'R9_README.md y r9_completion/REPRODUCIBILITY.md describen los comandos.', '',
        'Gráfico complementario: R9_comparacion_dominios.png. Manuscrito canónico actualizado: '
        'POLAR_Manuscrito_R9_20260920.pdf.', '']
    atomic_text(results/'INFORME_R9.md', '\n'.join(lines))
    print(str(results/'INFORME_R9.md'))


if __name__ == '__main__':
    main()
