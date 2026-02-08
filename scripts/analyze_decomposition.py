#!/usr/bin/env python3
"""Decomposition validation analysis for Enhanced UCAT.

Produces:
- tau_a vs tau_e scatter (Fig A)
- Blur response curves (Fig B)
- OOD decomposition box plots (Fig C)
- Temperature distribution histograms (Fig E)
- Decomposition validation metrics JSON
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.efficient_vit import EfficientEuroSATViT
from src.models.baseline import BaselineViT
from src.data.datasets import get_dataloaders, get_dataset_info
from src.data.blur import apply_all_blur_levels
from src.training.losses import UCATLoss
from src.utils.helpers import set_seed, get_device


def load_model(checkpoint_path, device, dataset_name='eurosat'):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get('model_config', {})
    default_num_classes = get_dataset_info(dataset_name)['num_classes']

    model = EfficientEuroSATViT(
        num_classes=config.get('num_classes', default_num_classes),
        use_learned_temp=config.get('use_learned_temp', True),
        use_early_exit=config.get('use_early_exit', True),
        use_learned_dropout=config.get('use_learned_dropout', True),
        use_learned_residual=config.get('use_learned_residual', True),
        use_temp_annealing=config.get('use_temp_annealing', True),
        use_decomposition=config.get('use_decomposition', False),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    if hasattr(model, 'early_exit_enabled'):
        model.early_exit_enabled = False
    return model, config


def collect_temperatures(model, dataloader, device):
    """Collect tau_a and tau_e for all samples."""
    all_tau_a = []
    all_tau_e = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            result = model(images, training_progress=1.0, return_temperatures=True)

            if len(result) == 4:  # decomposed
                logits, _, tau_a, tau_e = result
                all_tau_a.append(tau_a.cpu())
                tau_e_expanded = tau_e.expand(logits.shape[0]).cpu()
                all_tau_e.append(tau_e_expanded)
            else:
                logits, temps = result
                all_tau_a.append(temps.cpu())
                all_tau_e.append(torch.zeros(logits.shape[0]))

            _, predicted = logits.max(1)
            all_labels.append(labels)
            all_preds.append(predicted.cpu())

    return {
        'tau_a': torch.cat(all_tau_a).numpy(),
        'tau_e': torch.cat(all_tau_e).numpy(),
        'labels': torch.cat(all_labels).numpy(),
        'preds': torch.cat(all_preds).numpy(),
    }


def blur_response_analysis(model, dataloader, device, num_samples=500):
    """Measure temperature response to blur at each level."""
    images_batch = []
    count = 0
    for images, _ in dataloader:
        images_batch.append(images)
        count += images.shape[0]
        if count >= num_samples:
            break
    images_all = torch.cat(images_batch)[:num_samples].to(device)

    results = {'levels': [], 'mean_tau_a': [], 'mean_tau_e': [], 'mean_tau_total': []}

    with torch.no_grad():
        for level in range(5):
            from src.data.blur import apply_gaussian_blur
            blurred = apply_gaussian_blur(images_all, level)
            result = model(blurred, training_progress=1.0, return_temperatures=True)

            if len(result) == 4:
                _, _, tau_a, tau_e = result
                results['levels'].append(level)
                results['mean_tau_a'].append(float(tau_a.mean()))
                results['mean_tau_e'].append(float(tau_e.mean()))
                results['mean_tau_total'].append(float(tau_a.mean() + tau_e.mean()))
            else:
                _, temps = result
                results['levels'].append(level)
                results['mean_tau_a'].append(float(temps.mean()))
                results['mean_tau_e'].append(0.0)
                results['mean_tau_total'].append(float(temps.mean()))

    # Compute correlations
    tau_a_arr = np.array(results['mean_tau_a'])
    levels_arr = np.array(results['levels'], dtype=float)
    if tau_a_arr.std() > 0:
        results['tau_a_blur_corr'] = float(np.corrcoef(tau_a_arr, levels_arr)[0, 1])
    else:
        results['tau_a_blur_corr'] = 0.0

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='eurosat',
                        choices=['eurosat', 'cifar100', 'resisc45'],
                        help='Dataset to evaluate on')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./analysis_results/decomposition')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(42)
    device = get_device()

    print("Loading model...")
    model, config = load_model(args.checkpoint, device, dataset_name=args.dataset)

    print("Loading data...")
    _, _, test_loader, _ = get_dataloaders(
        dataset_name=args.dataset,
        root=args.data_root, batch_size=args.batch_size, num_workers=4
    )

    # Collect temperatures
    print("Collecting temperatures...")
    data = collect_temperatures(model, test_loader, device)

    # Correlation between tau_a and tau_e
    if data['tau_a'].std() > 0 and data['tau_e'].std() > 0:
        corr = float(np.corrcoef(data['tau_a'], data['tau_e'])[0, 1])
    else:
        corr = 0.0

    # Blur response
    print("Analysing blur response...")
    blur_data = blur_response_analysis(model, test_loader, device)

    # Save results
    results = {
        'tau_a_tau_e_correlation': corr,
        'mean_tau_a': float(data['tau_a'].mean()),
        'mean_tau_e': float(data['tau_e'].mean()),
        'std_tau_a': float(data['tau_a'].std()),
        'std_tau_e': float(data['tau_e'].std()),
        'blur_response': blur_data,
        'accuracy': float((data['preds'] == data['labels']).mean()),
    }

    with open(os.path.join(args.save_dir, 'decomposition_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # --- Figures ---
    sns.set_style('whitegrid')

    # Fig A: tau_a vs tau_e scatter
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(data['tau_a'], data['tau_e'], c=data['labels'],
                         cmap='tab10', alpha=0.5, s=10)
    ax.set_xlabel(r'$\tau_a$ (Aleatoric)')
    ax.set_ylabel(r'$\tau_e$ (Epistemic)')
    ax.set_title(f'Temperature Decomposition (r={corr:.3f})')
    plt.colorbar(scatter, label='Class')
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'fig_a_decomposition_scatter.png'), dpi=300)
    plt.close()

    # Fig B: Blur response
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(blur_data['levels'], blur_data['mean_tau_a'], 'o-', label=r'$\tau_a$ (aleatoric)')
    ax.plot(blur_data['levels'], blur_data['mean_tau_e'], 's-', label=r'$\tau_e$ (epistemic)')
    ax.plot(blur_data['levels'], blur_data['mean_tau_total'], '^-', label=r'$\tau_{total}$')
    ax.set_xlabel('Blur Level')
    ax.set_ylabel('Mean Temperature')
    ax.set_title('Temperature Response to Image Blur')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'fig_b_blur_response.png'), dpi=300)
    plt.close()

    # Fig E: Temperature distributions
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(data['tau_a'], bins=50, alpha=0.7, color='steelblue')
    axes[0].set_xlabel(r'$\tau_a$')
    axes[0].set_title('Aleatoric Temperature Distribution')
    axes[1].hist(data['tau_e'], bins=50, alpha=0.7, color='coral')
    axes[1].set_xlabel(r'$\tau_e$')
    axes[1].set_title('Epistemic Temperature Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'fig_e_temp_distributions.png'), dpi=300)
    plt.close()

    print(f"Results saved to {args.save_dir}")
    print(f"  tau_a-tau_e correlation: {corr:.4f}")
    print(f"  tau_a-blur correlation:  {blur_data.get('tau_a_blur_corr', 'N/A')}")


if __name__ == '__main__':
    main()
