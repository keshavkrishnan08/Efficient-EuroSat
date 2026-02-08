#!/usr/bin/env python3
"""Plot temperature evolution over training epochs (Fig D)."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_file', type=str, required=True,
                        help='Path to training results JSON with temperature_dynamics')
    parser.add_argument('--save_dir', type=str, default='./analysis_results/dynamics')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    with open(args.results_file) as f:
        results = json.load(f)

    dynamics = results.get('training', {}).get('temperature_dynamics', [])
    if not dynamics:
        print("No temperature dynamics found in results file.")
        return

    epochs = [d['epoch'] for d in dynamics]
    tau_e = [d['mean_tau_e'] for d in dynamics]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, tau_e, 'o-', color='coral', label=r'$\tau_e$ (epistemic)', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Temperature')
    ax.set_title('Epistemic Temperature During Training')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'fig_d_training_dynamics.png'), dpi=300)
    plt.close()

    print(f"Figure saved to {args.save_dir}/fig_d_training_dynamics.png")
    print(f"  tau_e start: {tau_e[0]:.4f}")
    print(f"  tau_e end:   {tau_e[-1]:.4f}")


if __name__ == '__main__':
    main()
