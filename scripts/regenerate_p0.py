#!/usr/bin/env python3
"""Regenerate P0 summaries, tables and a figure solely from recorded traces."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from report_utils import summarize_trace


def read_json(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def regenerate(index, out):
    index, out = Path(index), Path(out)
    manifest = read_json(index)
    rows = []
    for record in manifest['scenarios']:
        trace = read_json(index.parent / record['trace_json'])['timeseries']
        ctrlpath = record['paired_control_json']
        control = None if ctrlpath is None else read_json(index.parent / ctrlpath)['timeseries']
        summary = summarize_trace(trace, record['scenario'], control)
        rows.append({'scenario': record['scenario'], 'seed': record['seed'], **summary})
    out.mkdir(parents=True, exist_ok=True)
    (out / 'p0_summaries.json').write_text(json.dumps(rows, indent=2, allow_nan=False) + '\n')
    scalar_fields = [key for key in rows[0] if key != 'recovery']
    with (out / 'p0_summaries.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in scalar_fields} for row in rows)
    scenarios = list(dict.fromkeys(row['scenario'] for row in rows))
    def avg(group, key):
        values = [row[key] for row in group]
        return None if any(v is None for v in values) else float(np.mean(values))
    tex = [r'\begin{table}[htbp]', r'\centering\small',
           r'\caption{Corrected reference experiments: means over five recorded seeds. Modulation and override are separate percentages of transitions; return status is not converted to zero.}',
           r'\label{tab:p0-corrected}', r'\begin{tabular}{lrrrr}', r'\toprule',
           r'Condition & HGI & INC & Modulation (\%) & Override (\%) \\', r'\midrule']
    groups = [[r for r in rows if r['scenario'] == name] for name in scenarios]
    for name, group in zip(scenarios, groups):
        vals = [avg(group, key) for key in ['HGI_final', 'INC_final', 'Continuous_modulation_pct', 'Override_pct']]
        tex.append(name.replace('_', r'\_') + ' & ' + ' & '.join('--' if v is None else f'{v:.4f}' for v in vals) + r' \\')
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (out / 'p0_corrected.tex').write_text('\n'.join(tex) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7), sharey=True)
    for ax, metric, label in zip(axes, ['HGI_final', 'INC_final'],
                                  ['Graph activation homogeneity', 'Temporal / self-EMA alignment']):
        for i, group in enumerate(groups):
            values = np.asarray([r[metric] for r in group], dtype=float)
            ax.errorbar(values.mean(), i, xerr=values.std(ddof=1) if len(values) > 1 else 0,
                        fmt='o', color='#155b88', capsize=3)
        ax.set_xlabel(label); ax.grid(axis='x', alpha=.2)
        ax.set_yticks(np.arange(len(scenarios)), [s.replace('_', ' ') for s in scenarios])
    axes[0].invert_yaxis()
    fig.suptitle('Corrected reference: mean and SD across recorded seeds\nDescriptors only; not evidence of consciousness', fontsize=11)
    fig.tight_layout()
    fig.savefig(out / 'p0_summary.pdf'); fig.savefig(out / 'p0_summary.png', dpi=160)
    plt.close(fig)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, default=ROOT / 'results_corrected/p0/index.json')
    parser.add_argument('--out', type=Path, default=ROOT / 'results_corrected/p0_reports')
    args = parser.parse_args()
    rows = regenerate(args.index, args.out)
    print(f'Regenerated {len(rows)} P0 summaries from recorded traces')
