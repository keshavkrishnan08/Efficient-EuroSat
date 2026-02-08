#!/usr/bin/env python3
"""
Ablation study script for EfficientEuroSAT.

Runs a comprehensive ablation study across all modification combinations
with multiple random seeds for statistical significance.

Experiment configurations:
1. Baseline (all modifications off)
2. Each modification individually (5 configs)
3. All modifications on (full model)
4. All modifications minus one (5 configs)
Total: 12 configurations x 3 seeds = 36 runs

Usage:
    python ablation_study.py --epochs 100 --data_root ./data
    python ablation_study.py --epochs 50 --seeds 42 123 456 789
    python ablation_study.py --no_wandb --epochs 30
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import csv
import time
import datetime
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

from src.data.eurosat import get_eurosat_dataloaders
from src.models.efficient_vit import EfficientEuroSATViT, create_efficient_eurosat_tiny
from src.models.baseline import BaselineViT
from src.training.trainer import EuroSATTrainer
from src.utils.helpers import set_seed, get_device, count_parameters


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run ablation study for EfficientEuroSAT modifications',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456],
                        help='Random seeds for repeated experiments')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs per run')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory for dataset')
    parser.add_argument('--save_dir', type=str, default='./ablation_results',
                        help='Directory to save ablation results')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='efficient_eurosat_ablation',
                        help='W&B project name')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from previous ablation run (skip completed configs)')
    return parser.parse_args()


# ============================================================================
# Define all ablation experiment configurations
# ============================================================================

MODIFICATION_KEYS = [
    'use_learned_temp',
    'use_early_exit',
    'use_learned_dropout',
    'use_learned_residual',
    'use_temp_annealing',
]

SHORT_NAMES = {
    'use_learned_temp': 'LTemp',
    'use_early_exit': 'EExit',
    'use_learned_dropout': 'LDrop',
    'use_learned_residual': 'LResid',
    'use_temp_annealing': 'TAnneal',
}


def get_ablation_configs():
    """Define all 12 ablation experiment configurations."""
    configs = []

    # 1. Baseline: all modifications OFF
    configs.append({
        'name': 'baseline_none',
        'description': 'Baseline (no modifications)',
        'use_learned_temp': False,
        'use_early_exit': False,
        'use_learned_dropout': False,
        'use_learned_residual': False,
        'use_temp_annealing': False,
    })

    # 2. Each modification individually (5 configs)
    for key in MODIFICATION_KEYS:
        config = {
            'name': f'only_{SHORT_NAMES[key]}',
            'description': f'Only {SHORT_NAMES[key]} enabled',
            'use_learned_temp': False,
            'use_early_exit': False,
            'use_learned_dropout': False,
            'use_learned_residual': False,
            'use_temp_annealing': False,
        }
        config[key] = True
        configs.append(config)

    # 3. All modifications ON (full model)
    configs.append({
        'name': 'full_model',
        'description': 'Full model (all modifications)',
        'use_learned_temp': True,
        'use_early_exit': True,
        'use_learned_dropout': True,
        'use_learned_residual': True,
        'use_temp_annealing': True,
    })

    # 4. All modifications minus one (5 configs)
    for key in MODIFICATION_KEYS:
        config = {
            'name': f'no_{SHORT_NAMES[key]}',
            'description': f'All except {SHORT_NAMES[key]}',
            'use_learned_temp': True,
            'use_early_exit': True,
            'use_learned_dropout': True,
            'use_learned_residual': True,
            'use_temp_annealing': True,
        }
        config[key] = False
        configs.append(config)

    return configs


def run_single_experiment(config, seed, args, device, train_loader, val_loader,
                          test_loader, save_dir):
    """Run a single training + evaluation experiment."""
    set_seed(seed)
    exp_name = f"{config['name']}_seed{seed}"

    print(f"\n{'=' * 60}")
    print(f"Running: {exp_name}")
    print(f"Config:  {config['description']}")
    print(f"Seed:    {seed}")
    print(f"{'=' * 60}")

    # Build model
    model = EfficientEuroSATViT(
        img_size=args.img_size,
        num_classes=10,
        use_learned_temp=config['use_learned_temp'],
        use_early_exit=config['use_early_exit'],
        use_learned_dropout=config['use_learned_dropout'],
        use_learned_residual=config['use_learned_residual'],
        use_temp_annealing=config['use_temp_annealing'],
    )
    model = model.to(device)
    total_params, trainable_params = count_parameters(model)

    # Create optimizer and trainer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    exp_save_dir = os.path.join(save_dir, 'checkpoints', exp_name)
    trainer = EuroSATTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        device=device,
        save_dir=exp_save_dir,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=exp_name,
    )

    # Train
    train_start = time.time()
    train_results = trainer.train(num_epochs=args.epochs)
    train_duration = time.time() - train_start

    # Evaluate on test set
    test_results = trainer.test()

    # Collect results
    result = {
        'config_name': config['name'],
        'description': config['description'],
        'seed': seed,
        'modifications': {k: config[k] for k in MODIFICATION_KEYS},
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'train_duration_seconds': train_duration,
        'test_accuracy': test_results.get('accuracy', 0),
        'test_loss': test_results.get('loss', 0),
        'best_val_accuracy': train_results.get('best_val_accuracy', 0),
        'best_epoch': train_results.get('best_epoch', 0),
        'avg_exit_layer': test_results.get('avg_exit_layer', None),
        'early_exit_ratio': test_results.get('early_exit_ratio', None),
    }

    # Save individual result
    result_path = os.path.join(save_dir, 'individual', f'{exp_name}.json')
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"  Test Accuracy: {result['test_accuracy'] * 100:.2f}%")
    print(f"  Train Time:    {train_duration / 60:.1f} min")

    return result


def aggregate_results(all_results):
    """Aggregate results across seeds for each configuration."""
    grouped = defaultdict(list)
    for result in all_results:
        grouped[result['config_name']].append(result)

    aggregated = []
    for config_name, results in grouped.items():
        accuracies = [r['test_accuracy'] for r in results]
        losses = [r['test_loss'] for r in results]
        durations = [r['train_duration_seconds'] for r in results]
        val_accs = [r['best_val_accuracy'] for r in results]

        exit_layers = [r['avg_exit_layer'] for r in results
                       if r['avg_exit_layer'] is not None]
        exit_ratios = [r['early_exit_ratio'] for r in results
                       if r['early_exit_ratio'] is not None]

        agg = {
            'config_name': config_name,
            'description': results[0]['description'],
            'modifications': results[0]['modifications'],
            'num_seeds': len(results),
            'seeds': [r['seed'] for r in results],
            'total_parameters': results[0]['total_parameters'],
            'test_accuracy_mean': float(np.mean(accuracies)),
            'test_accuracy_std': float(np.std(accuracies)),
            'test_accuracy_min': float(np.min(accuracies)),
            'test_accuracy_max': float(np.max(accuracies)),
            'test_loss_mean': float(np.mean(losses)),
            'test_loss_std': float(np.std(losses)),
            'val_accuracy_mean': float(np.mean(val_accs)),
            'val_accuracy_std': float(np.std(val_accs)),
            'train_duration_mean': float(np.mean(durations)),
            'train_duration_std': float(np.std(durations)),
        }

        if exit_layers:
            agg['avg_exit_layer_mean'] = float(np.mean(exit_layers))
            agg['avg_exit_layer_std'] = float(np.std(exit_layers))

        if exit_ratios:
            agg['early_exit_ratio_mean'] = float(np.mean(exit_ratios))
            agg['early_exit_ratio_std'] = float(np.std(exit_ratios))

        aggregated.append(agg)

    # Sort by mean accuracy (descending)
    aggregated.sort(key=lambda x: x['test_accuracy_mean'], reverse=True)
    return aggregated


def save_summary_csv(aggregated, save_path):
    """Save aggregated results as a CSV table."""
    fieldnames = [
        'Rank', 'Config', 'Description',
        'LTemp', 'EExit', 'LDrop', 'LResid', 'TAnneal',
        'Accuracy (mean)', 'Accuracy (std)',
        'Val Acc (mean)', 'Loss (mean)',
        'Parameters', 'Train Time (min)',
        'Avg Exit Layer', 'Early Exit Ratio',
    ]

    with open(save_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for rank, agg in enumerate(aggregated, 1):
            mods = agg['modifications']
            row = {
                'Rank': rank,
                'Config': agg['config_name'],
                'Description': agg['description'],
                'LTemp': 'Y' if mods['use_learned_temp'] else 'N',
                'EExit': 'Y' if mods['use_early_exit'] else 'N',
                'LDrop': 'Y' if mods['use_learned_dropout'] else 'N',
                'LResid': 'Y' if mods['use_learned_residual'] else 'N',
                'TAnneal': 'Y' if mods['use_temp_annealing'] else 'N',
                'Accuracy (mean)': f"{agg['test_accuracy_mean'] * 100:.2f}%",
                'Accuracy (std)': f"{agg['test_accuracy_std'] * 100:.2f}%",
                'Val Acc (mean)': f"{agg['val_accuracy_mean'] * 100:.2f}%",
                'Loss (mean)': f"{agg['test_loss_mean']:.4f}",
                'Parameters': f"{agg['total_parameters']:,}",
                'Train Time (min)': f"{agg['train_duration_mean'] / 60:.1f}",
                'Avg Exit Layer': (
                    f"{agg.get('avg_exit_layer_mean', 0):.2f}"
                    if 'avg_exit_layer_mean' in agg else 'N/A'
                ),
                'Early Exit Ratio': (
                    f"{agg.get('early_exit_ratio_mean', 0) * 100:.1f}%"
                    if 'early_exit_ratio_mean' in agg else 'N/A'
                ),
            }
            writer.writerow(row)


def print_results_table(aggregated):
    """Print a formatted results table to the console."""
    print("\n" + "=" * 100)
    print("ABLATION STUDY RESULTS (sorted by test accuracy)")
    print("=" * 100)

    header = (
        f"{'Rank':>4} | {'Config':<20} | {'LT':>2} {'EE':>2} {'LD':>2} "
        f"{'LR':>2} {'TA':>2} | {'Accuracy':>14} | {'Val Acc':>14} | "
        f"{'Params':>10} | {'Time':>8}"
    )
    print(header)
    print("-" * 100)

    for rank, agg in enumerate(aggregated, 1):
        mods = agg['modifications']
        lt = 'Y' if mods['use_learned_temp'] else '.'
        ee = 'Y' if mods['use_early_exit'] else '.'
        ld = 'Y' if mods['use_learned_dropout'] else '.'
        lr = 'Y' if mods['use_learned_residual'] else '.'
        ta = 'Y' if mods['use_temp_annealing'] else '.'

        acc_str = (
            f"{agg['test_accuracy_mean'] * 100:.2f} +/- "
            f"{agg['test_accuracy_std'] * 100:.2f}"
        )
        val_str = (
            f"{agg['val_accuracy_mean'] * 100:.2f} +/- "
            f"{agg['val_accuracy_std'] * 100:.2f}"
        )
        time_str = f"{agg['train_duration_mean'] / 60:.1f}m"

        row = (
            f"{rank:>4} | {agg['config_name']:<20} | {lt:>2} {ee:>2} {ld:>2} "
            f"{lr:>2} {ta:>2} | {acc_str:>14} | {val_str:>14} | "
            f"{agg['total_parameters']:>10,} | {time_str:>8}"
        )
        print(row)

    print("=" * 100)
    print("Legend: LT=LearnedTemp, EE=EarlyExit, LD=LearnedDropout, "
          "LR=LearnedResidual, TA=TempAnnealing")
    print("Y=enabled, .=disabled")


def main():
    args = parse_args()
    device = get_device()

    print("=" * 70)
    print("EfficientEuroSAT Ablation Study")
    print("=" * 70)
    print(f"Seeds:       {args.seeds}")
    print(f"Epochs:      {args.epochs}")
    print(f"Batch Size:  {args.batch_size}")
    print(f"Device:      {device}")
    print()

    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'individual'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'checkpoints'), exist_ok=True)

    # Get all configs
    configs = get_ablation_configs()
    total_runs = len(configs) * len(args.seeds)
    print(f"Configurations: {len(configs)}")
    print(f"Seeds per config: {len(args.seeds)}")
    print(f"Total runs: {total_runs}")

    # Load completed results if resuming
    completed = set()
    all_results = []
    if args.resume:
        individual_dir = os.path.join(args.save_dir, 'individual')
        if os.path.exists(individual_dir):
            for fname in os.listdir(individual_dir):
                if fname.endswith('.json'):
                    fpath = os.path.join(individual_dir, fname)
                    with open(fpath, 'r') as f:
                        result = json.load(f)
                    all_results.append(result)
                    key = f"{result['config_name']}_seed{result['seed']}"
                    completed.add(key)
            print(f"Resuming: {len(completed)} runs already completed")

    # Load data once (shared across all runs)
    print("\nLoading EuroSAT dataset...")
    train_loader, val_loader, test_loader, _ = get_eurosat_dataloaders(
        root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Run all experiments
    study_start = time.time()
    run_count = len(completed)

    for config_idx, config in enumerate(configs):
        for seed in args.seeds:
            exp_key = f"{config['name']}_seed{seed}"
            run_count += 1

            if exp_key in completed:
                print(f"\n[{run_count}/{total_runs}] Skipping {exp_key} (already completed)")
                continue

            print(f"\n[{run_count}/{total_runs}] Starting {exp_key}")
            elapsed = time.time() - study_start
            if run_count > len(completed) + 1:
                runs_done = run_count - len(completed) - 1
                avg_time = elapsed / runs_done
                remaining = avg_time * (total_runs - run_count)
                print(f"  Estimated time remaining: {remaining / 60:.0f} minutes")

            try:
                result = run_single_experiment(
                    config=config,
                    seed=seed,
                    args=args,
                    device=device,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    save_dir=args.save_dir,
                )
                all_results.append(result)
            except Exception as e:
                print(f"  ERROR: {e}")
                error_result = {
                    'config_name': config['name'],
                    'description': config['description'],
                    'seed': seed,
                    'error': str(e),
                    'modifications': {k: config[k] for k in MODIFICATION_KEYS},
                    'test_accuracy': 0,
                    'test_loss': float('inf'),
                    'best_val_accuracy': 0,
                    'best_epoch': 0,
                    'total_parameters': 0,
                    'trainable_parameters': 0,
                    'train_duration_seconds': 0,
                    'avg_exit_layer': None,
                    'early_exit_ratio': None,
                }
                all_results.append(error_result)
                # Save error result
                result_path = os.path.join(
                    args.save_dir, 'individual', f'{exp_key}.json'
                )
                with open(result_path, 'w') as f:
                    json.dump(error_result, f, indent=2)

    total_duration = time.time() - study_start

    # Filter out error results for aggregation
    valid_results = [r for r in all_results if 'error' not in r]

    # Aggregate results
    print("\n\nAggregating results across seeds...")
    aggregated = aggregate_results(valid_results)

    # Save all results
    all_results_path = os.path.join(args.save_dir, 'all_results.json')
    with open(all_results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"All results saved to: {all_results_path}")

    # Save aggregated results
    agg_path = os.path.join(args.save_dir, 'aggregated_results.json')
    with open(agg_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    print(f"Aggregated results saved to: {agg_path}")

    # Save summary CSV
    csv_path = os.path.join(args.save_dir, 'ablation_summary.csv')
    save_summary_csv(aggregated, csv_path)
    print(f"Summary CSV saved to: {csv_path}")

    # Print results table
    print_results_table(aggregated)

    # Print study summary
    print(f"\nAblation study completed in {total_duration / 60:.1f} minutes")
    print(f"Total runs: {len(valid_results)} successful, "
          f"{len(all_results) - len(valid_results)} failed")

    if aggregated:
        best = aggregated[0]
        print(f"\nBest configuration: {best['config_name']}")
        print(f"  Accuracy: {best['test_accuracy_mean'] * 100:.2f}% "
              f"+/- {best['test_accuracy_std'] * 100:.2f}%")

        # Find baseline for comparison
        baseline = next(
            (a for a in aggregated if a['config_name'] == 'baseline_none'), None
        )
        if baseline:
            improvement = (
                best['test_accuracy_mean'] - baseline['test_accuracy_mean']
            ) * 100
            print(f"  Improvement over baseline: +{improvement:.2f}%")


if __name__ == '__main__':
    main()
