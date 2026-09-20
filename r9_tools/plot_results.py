"""Plot already compiled R9 descriptive results; no simulation or new tests."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    if data['metadata']['phase'] != 'final' or data['metadata']['n'] != 80:
        raise ValueError('Figure requires the complete final compilation')
    if args.output.exists():
        raise FileExistsError(args.output)
    domains = data['metadata']['domains']
    names = {'ecology_train': 'Ecología\noriginal', 'ecology_delay9': 'Ecología\nretardo 9',
             'inventory_transfer': 'Inventario', 'thermal_transfer': 'Microgrid\ntérmico'}
    x = np.arange(len(domains))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    for ax, metric, title in zip(axes, ('reward_mean', 'alive_fraction'),
                                 ('Recompensa media', 'Fracción del horizonte con vida')):
        for variant, label, offset, color in (
                ('full', 'POLAR R9 completo', -.18, '#175E87'),
                ('dense', 'Denso con adquisición propia', .18, '#D28B36')):
            stats = [data['native'][d][variant][metric] for d in domains]
            means = np.array([s['mean'] for s in stats])
            errors = np.array([[s['mean']-s['lower'] for s in stats],
                               [s['upper']-s['mean'] for s in stats]])
            ax.bar(x+offset, means, width=.34, color=color, label=label,
                   yerr=errors, capsize=3, error_kw={'linewidth': 1})
        ax.set_title(title, loc='left', fontweight='bold')
        ax.set_xticks(x, [names[d] for d in domains])
        ax.set_ylim(0, 1.04)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#D9DFE5', linewidth=.6)
    axes[0].set_ylabel('Escala 0–1')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .94), ncol=2, frameon=False)
    fig.suptitle('R9 · resultados descriptivos de 80 semillas', x=.065, ha='left',
                 y=.995, fontsize=15, fontweight='bold')
    fig.text(.065, .025, 'Barras: media. Intervalos: bootstrap 95% por semilla, sin ajuste múltiple.\n'
             'La confirmación se decide por seis contrastes conjuntos predefinidos con corrección de Holm.',
             fontsize=9, color='#46535E')
    fig.subplots_adjust(top=.79, bottom=.22, left=.065, right=.98, wspace=.13)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170, facecolor='white')
    plt.close(fig)
    print(str(args.output))


if __name__ == '__main__':
    main()
