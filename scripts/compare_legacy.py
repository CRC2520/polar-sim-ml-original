#!/usr/bin/env python3
"""Compare preserved endpoints with recorded corrected runs; never impute data.

The historical runs have incomplete configuration provenance. These are
descriptive revision deltas, not paired causal estimates or exact replications.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = {
    'p2': 'results_p2_baseline/p2_baseline_timeseries.csv',
    'p6': 'results_p6_stimulus/p6_stimulus_timeseries.csv',
    'p7': 'results_p7_learnedM/baseline/P7 Baseline_timeseries.csv',
    'p7_no_modulation': 'results_p7_no_ethics/no_ethics/P7 No Ethics_timeseries.csv',
    'p7_no_homeostasis': 'results_p7_no_homeo/no_homeostasis/P7 No Homeostasis_timeseries.csv',
    'iacl': 'results_p_mini_iacl/mini_iacl_timeseries.csv',
    'perturb_no_reduction': 'results_p_mini_trauma_compare/no_recon/Trauma (no Recon)_timeseries.csv',
    'perturb_reduced': 'results_p_mini_trauma_compare/with_recon/Trauma+Recon_timeseries.csv',
}


def endpoint(path):
    with path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f'Empty historical trace: {path}')
    last = rows[-1]
    return {key: float(last[key]) for key in ('HGI', 'INC')}


def compare(index_path, seed, output):
    index_path = Path(index_path)
    manifest = json.loads(index_path.read_text())
    records = [r for r in manifest['scenarios'] if r['seed'] == seed]
    if set(r['scenario'] for r in records) != set(HISTORICAL):
        raise ValueError('Exactly the eight specified scenarios are required for this seed')
    rows, hashes = [], {}
    for record in records:
        scenario = record['scenario']
        source = ROOT / HISTORICAL[scenario]
        old = endpoint(source)
        summary_file = index_path.parent / record['summary_json']
        new = json.loads(summary_file.read_text())
        row = {'scenario': scenario, 'corrected_seed': seed}
        for key in ('HGI', 'INC'):
            value = new.get(key + '_final')
            row.update({f'original_{key}': old[key], f'corrected_{key}': value,
                        f'delta_{key}': None if value is None else value - old[key]})
        for key in ('Continuous_modulation_pct', 'Override_pct', 'HGI_guard_pct',
                    'INC_guard_pct', 'Clipping_pct', 'Recovery', 'Recovery_status'):
            row[key] = new.get(key)
        rows.append(row)
        for path in (source, summary_file):
            hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'legacy_revision_deltas.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    note = ('Archived endpoints versus corrected version 2.0.1, seed ' + str(seed) +
            '. Configuration and stimulus changes are documented; these are not paired causal estimates. '
            'Historical missing interventions/recovery are not reconstructed or treated as zero.')
    (output / 'legacy_revision_deltas.json').write_text(json.dumps(
        {'interpretation': note, 'source_sha256': hashes, 'rows': rows}, indent=2, allow_nan=False) + '\n')
    lines = ['# Changes relative to archived endpoints', '', note, '',
             '| Scenario | Original HGI | Corrected HGI | Original INC | Corrected INC |',
             '|---|---:|---:|---:|---:|']
    tex = [r'\begin{table}[htbp]', r'\centering\small',
           r'\caption{Descriptive revision deltas at seed 2025. Historical configurations are incompletely recorded; differences are not causal effect estimates.}',
           r'\label{tab:revision-deltas}', r'\begin{tabular}{lrrrr}', r'\toprule',
           r'Condition & Old HGI & New HGI & Old INC & New INC \\', r'\midrule']
    for row in rows:
        vals = [row[k] for k in ('original_HGI', 'corrected_HGI', 'original_INC', 'corrected_INC')]
        fmt = ['--' if v is None else f'{v:.4f}' for v in vals]
        lines.append('| ' + row['scenario'] + ' | ' + ' | '.join(fmt) + ' |')
        tex.append(row['scenario'].replace('_', r'\_') + ' & ' + ' & '.join(fmt) + r' \\')
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (output / 'legacy_revision_deltas.md').write_text('\n'.join(lines) + '\n')
    (output / 'legacy_revision_deltas.tex').write_text('\n'.join(tex) + '\n')
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, default=ROOT / 'results_corrected/p0/index.json')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--out', type=Path, default=ROOT / 'results_corrected/comparison')
    args = parser.parse_args()
    compare(args.index, args.seed, args.out)
    print(f'Regenerated historical comparison in {args.out}')
