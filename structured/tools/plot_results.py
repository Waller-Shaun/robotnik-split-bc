"""Reproducible static coursework figures from archived metrics, no fitting."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT/'figures'
    output.mkdir(exist_ok=True)
    rows = list(csv.DictReader((ROOT/'runs/full/history.csv').open()))
    epochs = [int(r['epoch']) for r in rows]
    finished = json.loads((ROOT/'runs/full/training_finished.json').read_text())
    fig, ax = plt.subplots(figsize=(6.2, 3.4), constrained_layout=True)
    for split in ('train', 'val'):
        ax.plot(epochs, [float(r[split+'_normalized_torque_mse']) for r in rows],
                label=('Weighted train' if split=='train' else 'Validation (all frames)'))
    ax.axhline(finished['initial_validation']['normalized_torque_mse'], ls='--',
               color='gray', label='Initial fixed gains: validation')
    ax.axvline(finished['best_epoch'], ls=':', color='black', label='Selected epoch')
    ax.set(xlabel='Epoch', ylabel='Normalized joint-torque MSE', title='Supervised gain-head learning')
    ax.legend(fontsize=8)
    fig.savefig(output/'training_curve.png', dpi=180)
    plt.close(fig)
    final = ROOT/'FINAL_RESULTS.json'
    if not final.exists():
        print('Training figure saved. No final result figure until the full test is complete.')
        return
    result = json.loads(final.read_text())
    values = [result['metrics'][v]['success_rate_percent'] for v in ('fixed', 'dynamic')]
    intervals = np.array([result['metrics'][v]['target_cluster_stratified_bootstrap_95_percent']
                          for v in ('fixed', 'dynamic')])
    fig, ax = plt.subplots(figsize=(5.4, 3.8), constrained_layout=True)
    ax.bar([0, 1], values, width=.55, color=['#667085', '#2867b2'])
    ax.errorbar([0, 1], values, yerr=np.array([values-intervals[:, 0], intervals[:, 1]-values]),
                fmt='none', ecolor='black', capsize=5)
    for i, variant in enumerate(('fixed', 'dynamic')):
        row = result['metrics'][variant]
        ax.text(i, 82, f"{row['successes']}/1500\n{row['success_rate_percent']:.2f}%",
                ha='center', fontsize=11, color='white')
    ax.set(xticks=[0, 1], xticklabels=['Fixed P/D', 'Learned dynamic P/D'],
           ylim=(0, 105), ylabel='Reach success (%)',
           title='500 new goals × 3 shared noise seeds')
    ax.text(.5, 3, '95% target-cluster stratified bootstrap intervals',
            transform=ax.get_yaxis_transform(), ha='center', fontsize=8)
    fig.savefig(output/'final_paired_success.png', dpi=180)
    plt.close(fig)
    print('Figures saved under '+str(output))


if __name__ == '__main__':
    main()
