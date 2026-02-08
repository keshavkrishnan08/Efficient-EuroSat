#!/usr/bin/env python3
"""
A6: Robustness analysis for EfficientEuroSAT UCAT temperatures.

Applies corruptions (blur, noise, jitter, brightness, contrast) to test
images and measures whether UCAT temperatures increase for corrupted
(harder) inputs.  If temperatures serve as reliable difficulty indicators,
they should rise when images are degraded and accuracy drops.

Usage:
    python analyze_robustness.py --checkpoint ./checkpoints/best.pth
    python analyze_robustness.py --checkpoint ./checkpoints/best.pth --save_dir ./robustness
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
from src.evaluation.robustness import (
    run_robustness_analysis,
    summarize_robustness,
    plot_robustness,
)
from src.utils.helpers import set_seed, get_device


def parse_args():
    parser = argparse.ArgumentParser(
        description='A6: Robustness analysis of UCAT temperatures under corruptions',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory for dataset')
    parser.add_argument('--save_dir', type=str,
                        default='./analysis_results/robustness',
                        help='Directory to save analysis results')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (auto-detected if not specified)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--noise_std', type=float, default=0.15,
                        help='Standard deviation for Gaussian noise corruption')
    return parser.parse_args()


def load_model(checkpoint_path, device):
    """Load a model from a saved checkpoint."""
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

    # Setup device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = get_device()

    print("=" * 70)
    print("A6: Robustness Analysis")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device:     {device}")
    print(f"Noise std:  {args.noise_std}")

    os.makedirs(args.save_dir, exist_ok=True)

    # Load model
    print("\nLoading model...")
    model, model_config = load_model(args.checkpoint, device)

    # Load test dataset
    print("Loading EuroSAT test data...")
    _, _, test_loader, _ = get_eurosat_dataloaders(
        root=args.data_root,
        img_size=model_config.get('img_size', 224),
        batch_size=args.batch_size,
    )
    test_dataset = test_loader.dataset

    # Run robustness analysis
    print("\nRunning robustness analysis across corruption types...")
    results = run_robustness_analysis(
        model,
        test_dataset,
        batch_size=args.batch_size,
        device=device,
        noise_std=args.noise_std,
    )

    # Summarize results
    summary = summarize_robustness(results)

    # Plot
    plot_path = os.path.join(args.save_dir, "robustness_analysis.png")
    plot_robustness(results, save_path=plot_path)

    # Save raw results
    results_path = os.path.join(args.save_dir, "robustness_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Save summary
    summary_path = os.path.join(args.save_dir, "robustness_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    # Print formatted summary
    print("\n" + "=" * 70)
    print("ROBUSTNESS SUMMARY")
    print("=" * 70)
    print(f"  Clean accuracy:          {summary['clean_acc'] * 100:.2f}%")
    print(f"  Clean temperature:       {summary['clean_temp']:.4f}")
    print(f"  Mean corrupted accuracy: {summary['mean_corrupted_acc'] * 100:.2f}%")
    print(f"  Mean corrupted temp:     {summary['mean_corrupted_temp']:.4f}")
    print(f"  Accuracy drop:           {summary['acc_drop'] * 100:.2f}%")
    print(f"  Temperature increase:    {summary['temp_increase']:+.4f}")
    print(f"  Temp-acc correlation:    {summary['temp_acc_correlation']:.4f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
