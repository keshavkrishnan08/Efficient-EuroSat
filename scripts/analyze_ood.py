#!/usr/bin/env python3
"""
OOD detection analysis script for EfficientEuroSAT (A4).

Uses UCAT attention temperatures as OOD scores:
  - In-distribution: EuroSAT satellite imagery
  - Out-of-distribution: DTD textures

Higher learned temperatures indicate greater model uncertainty and
serve as a zero-cost OOD signal without any auxiliary training.

Usage:
    python analyze_ood.py --checkpoint ./checkpoints/best_model.pth
    python analyze_ood.py --checkpoint ./checkpoints/best_model.pth --save_dir ./ood_results
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import numpy as np
import torch

from src.data.datasets import get_dataloaders, get_dataset_info
from src.models.efficient_vit import EfficientEuroSATViT
from src.models.baseline import BaselineViT
from src.evaluation.ood_detection import (
    collect_temperatures,
    compute_ood_metrics,
    run_ood_analysis,
    get_ood_dataloader,
)
from src.utils.helpers import set_seed, get_device


def parse_args():
    parser = argparse.ArgumentParser(
        description='A4: OOD Detection via UCAT temperatures',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--dataset', type=str, default='eurosat',
                        choices=['eurosat', 'cifar100', 'resisc45'],
                        help='Dataset to evaluate on')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory for datasets')
    parser.add_argument('--save_dir', type=str,
                        default='./analysis_results/ood',
                        help='Directory to save OOD analysis results')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (auto-detected if not specified)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    return parser.parse_args()


def load_model(checkpoint_path, device, dataset_name='eurosat'):
    """Load model from checkpoint, supporting both efficient and baseline."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint.get('model_config', {})
    model_type = model_config.get('model_type', 'efficient_eurosat')
    default_num_classes = get_dataset_info(dataset_name)['num_classes']

    if model_type == 'efficient_eurosat':
        model = EfficientEuroSATViT(
            img_size=model_config.get('img_size', 224),
            num_classes=model_config.get('num_classes', default_num_classes),
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
            num_classes=model_config.get('num_classes', default_num_classes),
        )
    else:
        raise ValueError(f"Unknown model type in checkpoint: {model_type}")

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    if hasattr(model, 'early_exit_enabled'):
        model.early_exit_enabled = False

    return model, model_config


def main():
    args = parse_args()
    set_seed(args.seed)

    # Device setup
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = get_device()

    print("=" * 70)
    print("A4: OOD Detection via UCAT Temperatures")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device:     {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # Load model
    print("\nLoading model...")
    model, model_config = load_model(args.checkpoint, device, dataset_name=args.dataset)
    print(f"  Model type: {model_config.get('model_type', 'efficient_eurosat')}")

    # Load in-distribution test data
    print("\nLoading in-distribution data...")
    _, _, test_loader, _ = get_dataloaders(
        dataset_name=args.dataset,
        root=args.data_root,
        img_size=model_config.get('img_size', 224),
        batch_size=args.batch_size,
    )

    # Load OOD data (DTD textures)
    print("Loading out-of-distribution data (DTD)...")
    ood_loader = get_ood_dataloader(
        ood_dataset="dtd",
        root=args.data_root,
        batch_size=args.batch_size,
    )

    # Run OOD analysis (using total temperature)
    print("\nRunning OOD analysis...")
    results = run_ood_analysis(
        model,
        in_loader=test_loader,
        ood_loader=ood_loader,
        device=device,
        save_path=os.path.join(args.save_dir, "ood_histogram.png"),
    )

    # Extract metrics (run_ood_analysis returns metrics + raw temps)
    metrics = {
        "auroc": float(results["auroc"]),
        "aupr": float(results["aupr"]),
        "fpr_at_95_tpr": float(results["fpr_at_95_tpr"]),
        "mean_in_temp": float(results["mean_in_temp"]),
        "mean_ood_temp": float(results["mean_ood_temp"]),
        "temp_gap": float(results["temp_gap"]),
    }

    # If decomposition model, also compute OOD metrics using tau_e only
    is_decomposed = model_config.get('use_decomposition', False)
    if is_decomposed:
        print("\nRunning decomposed OOD analysis (epistemic only)...")
        in_temps_e = collect_temperatures(model, test_loader, device=device, component="epistemic")
        ood_temps_e = collect_temperatures(model, ood_loader, device=device, component="epistemic")
        metrics_e = compute_ood_metrics(in_temps_e, ood_temps_e)

        metrics["epistemic_auroc"] = float(metrics_e["auroc"])
        metrics["epistemic_aupr"] = float(metrics_e["aupr"])
        metrics["epistemic_fpr_at_95_tpr"] = float(metrics_e["fpr_at_95_tpr"])
        metrics["epistemic_mean_in_temp"] = float(metrics_e["mean_in_temp"])
        metrics["epistemic_mean_ood_temp"] = float(metrics_e["mean_ood_temp"])
        metrics["epistemic_temp_gap"] = float(metrics_e["temp_gap"])

    # Save results JSON
    results_path = os.path.join(args.save_dir, "ood_results.json")
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Print formatted summary
    print("\n" + "=" * 70)
    print("OOD DETECTION SUMMARY")
    print("=" * 70)
    print(f"  AUROC (total):      {metrics['auroc']:.4f}")
    print(f"  AUPR:               {metrics['aupr']:.4f}")
    print(f"  FPR@95TPR:          {metrics['fpr_at_95_tpr']:.4f}")
    print(f"  Mean temp (in):     {metrics['mean_in_temp']:.4f}")
    print(f"  Mean temp (OOD):    {metrics['mean_ood_temp']:.4f}")
    print(f"  Temperature gap:    {metrics['temp_gap']:.4f}")
    if is_decomposed:
        print(f"  --- Epistemic Only ---")
        print(f"  AUROC (epistemic):  {metrics['epistemic_auroc']:.4f}")
        print(f"  AUPR (epistemic):   {metrics['epistemic_aupr']:.4f}")
        print(f"  FPR@95 (epistemic): {metrics['epistemic_fpr_at_95_tpr']:.4f}")
    print(f"  Histogram saved to: {os.path.join(args.save_dir, 'ood_histogram.png')}")
    print("=" * 70)


if __name__ == '__main__':
    main()
