#!/usr/bin/env python3
"""
Latency benchmarking script for EfficientEuroSAT.

Compares inference latency between baseline and EfficientEuroSAT models
across different configurations (GPU/CPU, with/without early exit).

Usage:
    python benchmark_latency.py --data_root ./data
    python benchmark_latency.py --checkpoint ./checkpoints/best.pth --device cuda
    python benchmark_latency.py --num_runs 500 --save_dir ./latency_results
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import time
import numpy as np
import torch

from src.data.datasets import get_dataloaders, get_dataset_info
from src.models.efficient_vit import EfficientEuroSATViT, create_efficient_eurosat_tiny
from src.models.baseline import BaselineViT
from src.utils.helpers import set_seed, get_device, count_parameters

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def parse_args():
    parser = argparse.ArgumentParser(
        description='Benchmark inference latency for EfficientEuroSAT',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--dataset', type=str, default='eurosat',
                        choices=['eurosat', 'cifar100', 'resisc45'],
                        help='Dataset to evaluate on')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory for dataset')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to trained model checkpoint (optional)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to benchmark on (auto-detect if not set)')
    parser.add_argument('--num_runs', type=int, default=200,
                        help='Number of inference runs for timing')
    parser.add_argument('--warmup_runs', type=int, default=50,
                        help='Number of warmup runs before timing')
    parser.add_argument('--batch_sizes', type=int, nargs='+',
                        default=[1, 8, 16, 32, 64],
                        help='Batch sizes to benchmark')
    parser.add_argument('--save_dir', type=str, default='./latency_results',
                        help='Directory to save results and plots')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    return parser.parse_args()


def benchmark_model(model, input_tensor, device, num_runs, warmup_runs):
    """Benchmark a model's inference latency."""
    model.eval()
    input_tensor = input_tensor.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)
            if device.type == 'cuda':
                torch.cuda.synchronize()

    # Timed runs
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            output = model(input_tensor)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

    latencies = np.array(latencies)

    stats = {
        'mean_ms': float(np.mean(latencies)),
        'std_ms': float(np.std(latencies)),
        'median_ms': float(np.median(latencies)),
        'min_ms': float(np.min(latencies)),
        'max_ms': float(np.max(latencies)),
        'p50_ms': float(np.percentile(latencies, 50)),
        'p90_ms': float(np.percentile(latencies, 90)),
        'p95_ms': float(np.percentile(latencies, 95)),
        'p99_ms': float(np.percentile(latencies, 99)),
        'throughput_per_sec': float(
            input_tensor.size(0) / (np.mean(latencies) / 1000)
        ),
        'raw_latencies': latencies.tolist(),
    }
    return stats


def create_model_configs():
    """Create model configurations to benchmark."""
    configs = [
        {
            'name': 'Baseline ViT',
            'model_class': 'baseline',
            'kwargs': {},
        },
        {
            'name': 'EfficientEuroSAT (no early exit)',
            'model_class': 'efficient_eurosat',
            'kwargs': {
                'use_learned_temp': True,
                'use_early_exit': False,
                'use_learned_dropout': True,
                'use_learned_residual': True,
                'use_temp_annealing': True,
            },
        },
        {
            'name': 'EfficientEuroSAT (with early exit)',
            'model_class': 'efficient_eurosat',
            'kwargs': {
                'use_learned_temp': True,
                'use_early_exit': True,
                'use_learned_dropout': True,
                'use_learned_residual': True,
                'use_temp_annealing': True,
                'exit_threshold': 0.9,
                'exit_min_layer': 4,
            },
        },
        {
            'name': 'EfficientEuroSAT (aggressive exit)',
            'model_class': 'efficient_eurosat',
            'kwargs': {
                'use_learned_temp': True,
                'use_early_exit': True,
                'use_learned_dropout': True,
                'use_learned_residual': True,
                'use_temp_annealing': True,
                'exit_threshold': 0.8,
                'exit_min_layer': 2,
            },
        },
    ]
    return configs


def build_model(config, img_size, device, num_classes=10):
    """Build a model from a config dictionary."""
    if config['model_class'] == 'baseline':
        model = BaselineViT(img_size=img_size, num_classes=num_classes)
    else:
        model = EfficientEuroSATViT(
            img_size=img_size, num_classes=num_classes, **config['kwargs']
        )
    model = model.to(device)
    model.eval()
    return model


def plot_latency_comparison(all_results, save_dir):
    """Generate latency comparison plots."""
    if not HAS_MATPLOTLIB:
        print("  matplotlib not available, skipping plots")
        return

    # Plot 1: Latency distribution for batch_size=1
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Find batch_size=1 results
    bs1_results = {}
    for config_name, bs_results in all_results.items():
        if 1 in bs_results:
            bs1_results[config_name] = bs_results[1]

    if bs1_results:
        # Histogram of latencies
        ax = axes[0]
        for name, stats in bs1_results.items():
            latencies = stats['raw_latencies']
            ax.hist(latencies, bins=50, alpha=0.5, label=name, density=True)
        ax.set_xlabel('Latency (ms)')
        ax.set_ylabel('Density')
        ax.set_title('Single Image Latency Distribution')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Bar chart of mean latency
        ax = axes[1]
        names = list(bs1_results.keys())
        means = [bs1_results[n]['mean_ms'] for n in names]
        stds = [bs1_results[n]['std_ms'] for n in names]
        short_names = [n.replace('EfficientEuroSAT', 'ETSR') for n in names]

        bars = ax.bar(range(len(names)), means, yerr=stds, capsize=5,
                      color=['#2196F3', '#4CAF50', '#FF9800', '#F44336'][:len(names)])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(short_names, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Mean Latency Comparison (batch=1)')
        ax.grid(True, alpha=0.3, axis='y')

        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{mean:.1f}ms', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'latency_distribution.png'), dpi=150)
    plt.close()

    # Plot 2: Throughput vs batch size
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for name, bs_results in all_results.items():
        batch_sizes = sorted(bs_results.keys())
        throughputs = [bs_results[bs]['throughput_per_sec'] for bs in batch_sizes]
        short_name = name.replace('EfficientEuroSAT', 'ETSR')
        ax.plot(batch_sizes, throughputs, 'o-', label=short_name)
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Throughput (images/sec)')
    ax.set_title('Throughput vs Batch Size')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Latency vs batch size
    ax = axes[1]
    for name, bs_results in all_results.items():
        batch_sizes = sorted(bs_results.keys())
        means = [bs_results[bs]['mean_ms'] for bs in batch_sizes]
        short_name = name.replace('EfficientEuroSAT', 'ETSR')
        ax.plot(batch_sizes, means, 'o-', label=short_name)
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Batch Latency vs Batch Size')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'throughput_vs_batchsize.png'), dpi=150)
    plt.close()

    # Plot 3: Percentile comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    percentiles = ['p50_ms', 'p90_ms', 'p95_ms', 'p99_ms']
    percentile_labels = ['P50', 'P90', 'P95', 'P99']

    if bs1_results:
        x = np.arange(len(percentiles))
        width = 0.8 / len(bs1_results)

        for i, (name, stats) in enumerate(bs1_results.items()):
            values = [stats[p] for p in percentiles]
            short_name = name.replace('EfficientEuroSAT', 'ETSR')
            ax.bar(x + i * width, values, width, label=short_name)

        ax.set_xticks(x + width * (len(bs1_results) - 1) / 2)
        ax.set_xticklabels(percentile_labels)
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Latency Percentiles (batch=1)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'latency_percentiles.png'), dpi=150)
    plt.close()

    print("  Plots saved to:")
    print(f"    {os.path.join(save_dir, 'latency_distribution.png')}")
    print(f"    {os.path.join(save_dir, 'throughput_vs_batchsize.png')}")
    print(f"    {os.path.join(save_dir, 'latency_percentiles.png')}")


def main():
    args = parse_args()
    set_seed(args.seed)

    # Setup device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = get_device()

    print("=" * 70)
    print("EfficientEuroSAT Latency Benchmark")
    print("=" * 70)
    print(f"Device:       {device}")
    print(f"Num runs:     {args.num_runs}")
    print(f"Warmup runs:  {args.warmup_runs}")
    print(f"Batch sizes:  {args.batch_sizes}")
    print()

    os.makedirs(args.save_dir, exist_ok=True)

    num_classes = get_dataset_info(args.dataset)['num_classes']

    # Create model configurations
    model_configs = create_model_configs()

    # If a checkpoint is provided, load its weights for the EfficientEuroSAT models
    checkpoint_state = None
    if args.checkpoint is not None:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        checkpoint_state = checkpoint.get('model_state_dict', None)

    # Run benchmarks
    all_results = {}

    for config in model_configs:
        print(f"\n--- Benchmarking: {config['name']} ---")

        model = build_model(config, args.img_size, device, num_classes=num_classes)
        total_params, _ = count_parameters(model)
        print(f"  Parameters: {total_params:,}")

        # Load weights if available and applicable
        if checkpoint_state is not None and config['model_class'] == 'efficient_eurosat':
            try:
                model.load_state_dict(checkpoint_state, strict=False)
                print("  Loaded checkpoint weights")
            except Exception as e:
                print(f"  Could not load weights: {e}")

        config_results = {}

        for batch_size in args.batch_sizes:
            # Create synthetic input
            input_tensor = torch.randn(
                batch_size, 3, args.img_size, args.img_size
            )

            print(f"  Batch size {batch_size}...", end=' ', flush=True)
            stats = benchmark_model(
                model, input_tensor, device,
                args.num_runs, args.warmup_runs
            )

            # Remove raw latencies for batch sizes > 1 to save space
            stats_for_save = {
                k: v for k, v in stats.items() if k != 'raw_latencies'
            }
            if batch_size == 1:
                stats_for_save['raw_latencies'] = stats['raw_latencies']

            config_results[batch_size] = stats
            print(
                f"mean={stats['mean_ms']:.2f}ms, "
                f"p99={stats['p99_ms']:.2f}ms, "
                f"throughput={stats['throughput_per_sec']:.0f} img/s"
            )

        all_results[config['name']] = config_results

        # Clean up model
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Cross-device comparison (if CUDA available and not forced to specific device)
    if args.device is None and torch.cuda.is_available():
        print("\n\n--- CPU vs GPU Comparison (batch=1) ---")
        cpu_device = torch.device('cpu')
        gpu_device = torch.device('cuda')

        comparison_configs = [
            model_configs[0],  # Baseline
            model_configs[2],  # EfficientEuroSAT with early exit
        ]

        for config in comparison_configs:
            input_tensor = torch.randn(1, 3, args.img_size, args.img_size)

            # CPU benchmark
            model_cpu = build_model(config, args.img_size, cpu_device, num_classes=num_classes)
            cpu_stats = benchmark_model(
                model_cpu, input_tensor, cpu_device,
                args.num_runs, args.warmup_runs
            )
            del model_cpu

            # GPU benchmark
            model_gpu = build_model(config, args.img_size, gpu_device, num_classes=num_classes)
            gpu_stats = benchmark_model(
                model_gpu, input_tensor, gpu_device,
                args.num_runs, args.warmup_runs
            )
            del model_gpu
            torch.cuda.empty_cache()

            speedup = cpu_stats['mean_ms'] / gpu_stats['mean_ms']
            print(f"  {config['name']}:")
            print(f"    CPU: {cpu_stats['mean_ms']:.2f}ms, "
                  f"GPU: {gpu_stats['mean_ms']:.2f}ms, "
                  f"Speedup: {speedup:.1f}x")

            all_results[f"{config['name']} (CPU)"] = {1: cpu_stats}

    # Generate plots
    print("\n--- Generating Plots ---")
    # Prepare results without raw latencies for JSON (except batch=1)
    results_for_json = {}
    for name, bs_results in all_results.items():
        results_for_json[name] = {}
        for bs, stats in bs_results.items():
            results_for_json[name][str(bs)] = {
                k: v for k, v in stats.items() if k != 'raw_latencies'
            }

    plot_latency_comparison(all_results, args.save_dir)

    # Save results
    results_path = os.path.join(args.save_dir, 'latency_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_for_json, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Print summary table
    print("\n" + "=" * 90)
    print("LATENCY BENCHMARK SUMMARY (batch_size=1)")
    print("=" * 90)
    header = (
        f"{'Model':<35} | {'Mean':>8} | {'Std':>8} | {'P50':>8} | "
        f"{'P95':>8} | {'P99':>8} | {'Thput':>10}"
    )
    print(header)
    print("-" * 90)

    for name, bs_results in all_results.items():
        if 1 in bs_results:
            stats = bs_results[1]
            row = (
                f"{name:<35} | {stats['mean_ms']:>7.2f}ms | "
                f"{stats['std_ms']:>7.2f}ms | {stats['p50_ms']:>7.2f}ms | "
                f"{stats['p95_ms']:>7.2f}ms | {stats['p99_ms']:>7.2f}ms | "
                f"{stats['throughput_per_sec']:>8.0f}/s"
            )
            print(row)

    print("=" * 90)

    # Print speedup analysis
    baseline_key = 'Baseline ViT'
    if baseline_key in all_results and 1 in all_results[baseline_key]:
        baseline_mean = all_results[baseline_key][1]['mean_ms']
        print(f"\nSpeedup relative to {baseline_key}:")
        for name, bs_results in all_results.items():
            if name != baseline_key and 1 in bs_results:
                model_mean = bs_results[1]['mean_ms']
                speedup = baseline_mean / model_mean
                diff_ms = baseline_mean - model_mean
                direction = "faster" if diff_ms > 0 else "slower"
                print(f"  {name}: {speedup:.2f}x ({abs(diff_ms):.2f}ms {direction})")

    print(f"\nAll results saved to: {args.save_dir}")


if __name__ == '__main__':
    main()
