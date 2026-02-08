#!/usr/bin/env python3
"""
A5: Calibration Analysis -- Baseline vs UCAT.

Evaluates prediction calibration for both a baseline ViT and a UCAT
(EfficientEuroSAT) model on the EuroSAT test set.  Computes Expected
Calibration Error (ECE), Maximum Calibration Error (MCE), generates
reliability diagrams for each model, and produces a side-by-side
calibration comparison plot.

Usage:
    python analyze_calibration.py \
        --checkpoint_ucat ./checkpoints/ucat_best.pth \
        --checkpoint_baseline ./checkpoints/baseline_best.pth

    python analyze_calibration.py \
        --checkpoint_ucat ./checkpoints/ucat_best.pth \
        --checkpoint_baseline ./checkpoints/baseline_best.pth \
        --save_dir ./analysis_results/calibration \
        --n_bins 20
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json

import numpy as np
import torch

from src.data.eurosat import get_eurosat_dataloaders
from src.models.efficient_vit import EfficientEuroSATViT
from src.models.baseline import BaselineViT
from src.evaluation.calibration import (
    evaluate_calibration,
    compare_calibration,
    plot_reliability_diagram,
    plot_calibration_comparison,
)
from src.utils.helpers import set_seed, get_device


def parse_args():
    parser = argparse.ArgumentParser(
        description='A5: Calibration analysis -- compare baseline vs UCAT',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--checkpoint_ucat', type=str, required=True,
                        help='Path to trained UCAT model checkpoint')
    parser.add_argument('--checkpoint_baseline', type=str, required=True,
                        help='Path to trained baseline model checkpoint')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory for EuroSAT data')
    parser.add_argument('--save_dir', type=str,
                        default='./analysis_results/calibration',
                        help='Directory to save results and plots')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (auto-detected if not specified)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Evaluation batch size')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--n_bins', type=int, default=15,
                        help='Number of calibration bins for ECE computation')
    return parser.parse_args()


def load_model(checkpoint_path, device):
    """Load a model from a saved checkpoint.

    Reads ``model_config`` from the checkpoint and reconstructs either an
    ``EfficientEuroSATViT`` or a ``BaselineViT`` accordingly.

    Returns
    -------
    tuple
        ``(model, model_config)``
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint.get('model_config', {})
    model_type = model_config.get('model_type', 'efficient_eurosat')

    if model_type == 'efficient_eurosat':
        model = EfficientEuroSATViT(
            img_size=model_config.get('img_size', 224),
            num_classes=model_config.get('num_classes', 10),
            use_learned_temp=model_config.get('use_learned_temp', True),
            use_early_exit=model_config.get('use_early_exit', True),
            use_learned_dropout=model_config.get('use_learned_dropout', True),
            use_learned_residual=model_config.get('use_learned_residual', True),
            use_temp_annealing=model_config.get('use_temp_annealing', True),
            use_decomposition=model_config.get('use_decomposition', False),
            tau_min=model_config.get('tau_min', 0.1),
            dropout_max=model_config.get('dropout_max', 0.3),
            exit_threshold=model_config.get('exit_threshold', 0.9),
            exit_min_layer=model_config.get('exit_min_layer', 4),
        )
    elif model_type == 'baseline':
        model = BaselineViT(
            img_size=model_config.get('img_size', 224),
            num_classes=model_config.get('num_classes', 10),
        )
    else:
        raise ValueError(f"Unknown model type in checkpoint: {model_type}")

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, model_config


def main():
    args = parse_args()
    set_seed(args.seed)

    # --- Device -----------------------------------------------------------
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = get_device()

    print("=" * 70)
    print("A5: Calibration Analysis -- Baseline vs UCAT")
    print("=" * 70)
    print(f"UCAT checkpoint:     {args.checkpoint_ucat}")
    print(f"Baseline checkpoint: {args.checkpoint_baseline}")
    print(f"Device:              {device}")
    print(f"Bins:                {args.n_bins}")

    os.makedirs(args.save_dir, exist_ok=True)

    # --- Load models ------------------------------------------------------
    print("\nLoading UCAT model...")
    ucat_model, ucat_config = load_model(args.checkpoint_ucat, device)
    print(f"  Model type: {ucat_config.get('model_type', 'efficient_eurosat')}")

    print("Loading baseline model...")
    baseline_model, baseline_config = load_model(args.checkpoint_baseline, device)
    print(f"  Model type: {baseline_config.get('model_type', 'baseline')}")

    # --- Load test data ---------------------------------------------------
    print("\nLoading EuroSAT test data...")
    _, _, test_loader, _ = get_eurosat_dataloaders(
        root=args.data_root,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(f"  Test batches: {len(test_loader)}")

    # --- Evaluate calibration ---------------------------------------------
    print("\nEvaluating baseline calibration...")
    baseline_results = evaluate_calibration(
        baseline_model, test_loader, device=str(device), n_bins=args.n_bins,
    )

    print("Evaluating UCAT calibration...")
    ucat_results = evaluate_calibration(
        ucat_model, test_loader, device=str(device), n_bins=args.n_bins,
    )

    # --- Compare calibration ----------------------------------------------
    comparison = compare_calibration(baseline_results, ucat_results)

    # --- Generate plots ---------------------------------------------------
    print("\nGenerating plots...")

    baseline_reliability_path = os.path.join(
        args.save_dir, 'reliability_diagram_baseline.png',
    )
    plot_reliability_diagram(
        baseline_results,
        save_path=baseline_reliability_path,
        label='Baseline',
    )

    ucat_reliability_path = os.path.join(
        args.save_dir, 'reliability_diagram_ucat.png',
    )
    plot_reliability_diagram(
        ucat_results,
        save_path=ucat_reliability_path,
        label='UCAT',
    )

    comparison_path = os.path.join(
        args.save_dir, 'calibration_comparison.png',
    )
    plot_calibration_comparison(
        baseline_results,
        ucat_results,
        save_path=comparison_path,
    )

    # --- Save results JSON ------------------------------------------------
    results = {
        'baseline': {
            'checkpoint': args.checkpoint_baseline,
            'ece': baseline_results['ece'],
            'mce': baseline_results['mce'],
            'accuracy': baseline_results['accuracy'],
            'mean_confidence': baseline_results['mean_confidence'],
            'overconfidence_gap': baseline_results['overconfidence_gap'],
            'bin_confidences': baseline_results['bin_confidences'].tolist(),
            'bin_accuracies': baseline_results['bin_accuracies'].tolist(),
            'bin_counts': baseline_results['bin_counts'].tolist(),
        },
        'ucat': {
            'checkpoint': args.checkpoint_ucat,
            'ece': ucat_results['ece'],
            'mce': ucat_results['mce'],
            'accuracy': ucat_results['accuracy'],
            'mean_confidence': ucat_results['mean_confidence'],
            'overconfidence_gap': ucat_results['overconfidence_gap'],
            'bin_confidences': ucat_results['bin_confidences'].tolist(),
            'bin_accuracies': ucat_results['bin_accuracies'].tolist(),
            'bin_counts': ucat_results['bin_counts'].tolist(),
        },
        'comparison': comparison,
        'config': {
            'n_bins': args.n_bins,
            'seed': args.seed,
            'batch_size': args.batch_size,
        },
    }

    results_path = os.path.join(args.save_dir, 'calibration_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # --- Print summary ----------------------------------------------------
    print("\n" + "=" * 70)
    print("CALIBRATION SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<25} {'Baseline':>12} {'UCAT':>12} {'Reduction':>12}")
    print("-" * 61)
    print(
        f"{'ECE':<25} {baseline_results['ece']:>12.4f} "
        f"{ucat_results['ece']:>12.4f} "
        f"{comparison['ece_reduction_pct']:>11.1f}%"
    )
    print(
        f"{'MCE':<25} {baseline_results['mce']:>12.4f} "
        f"{ucat_results['mce']:>12.4f} "
        f"{comparison['mce_reduction_pct']:>11.1f}%"
    )
    print(
        f"{'Accuracy':<25} {baseline_results['accuracy']:>12.4f} "
        f"{ucat_results['accuracy']:>12.4f}"
    )
    print(
        f"{'Mean Confidence':<25} {baseline_results['mean_confidence']:>12.4f} "
        f"{ucat_results['mean_confidence']:>12.4f}"
    )
    print(
        f"{'Overconfidence Gap':<25} {baseline_results['overconfidence_gap']:>12.4f} "
        f"{ucat_results['overconfidence_gap']:>12.4f}"
    )
    print("=" * 70)
    print(f"\nPlots saved to: {args.save_dir}")
    print(f"  - reliability_diagram_baseline.png")
    print(f"  - reliability_diagram_ucat.png")
    print(f"  - calibration_comparison.png")


if __name__ == '__main__':
    main()
