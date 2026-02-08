#!/usr/bin/env python3
"""
Early exit analysis script for EfficientEuroSAT.

Performs comprehensive analysis of early exit behavior including:
- Exit layer distribution across the full test set
- Per-class exit layer statistics
- Confidence calibration at exit points
- Accuracy vs. computational cost tradeoff analysis
- Exit layer correlation with difficulty/class type

Usage:
    python analyze_early_exit.py --checkpoint ./checkpoints/best.pth
    python analyze_early_exit.py --checkpoint ./checkpoints/best.pth --save_dir ./exit_analysis
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

from src.data.eurosat import get_eurosat_dataloaders, EUROSAT_CLASS_NAMES
from src.models.efficient_vit import EfficientEuroSATViT, create_efficient_eurosat_tiny
from src.utils.helpers import set_seed, get_device

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib not available. No plots will be generated.")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze early exit behavior of EfficientEuroSAT',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory for dataset')
    parser.add_argument('--save_dir', type=str, default='./exit_analysis',
                        help='Directory to save analysis results')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    return parser.parse_args()


def load_model(checkpoint_path, device):
    """Load the EfficientEuroSAT model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint.get('model_config', {})

    if model_config.get('model_type', 'efficient_eurosat') != 'efficient_eurosat':
        raise ValueError(
            "This script requires an EfficientEuroSAT model with early exit. "
            "The loaded checkpoint is a baseline model."
        )

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

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    if not model.use_early_exit:
        raise ValueError(
            "The loaded model does not have early exit enabled. "
            "Re-train with --use_early_exit or use a different checkpoint."
        )

    return model, model_config


def collect_exit_statistics(model, test_loader, device):
    """Collect detailed exit statistics on the full test set."""
    records = []

    model.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            if isinstance(outputs, dict):
                logits = outputs['logits']
                exit_layers = outputs.get('exit_layer', None)
                exit_confidences = outputs.get('exit_confidence', None)
                # Per-layer logits if available
                per_layer_logits = outputs.get('per_layer_logits', None)
            else:
                logits = outputs
                exit_layers = None
                exit_confidences = None
                per_layer_logits = None

            preds = logits.argmax(dim=-1).cpu().numpy()
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            labels_np = labels.cpu().numpy()

            batch_size = len(labels_np)

            # Handle exit layer format
            if exit_layers is not None:
                if isinstance(exit_layers, (int, float)):
                    exit_layers = [int(exit_layers)] * batch_size
                elif isinstance(exit_layers, torch.Tensor):
                    exit_layers = exit_layers.cpu().numpy().tolist()
                elif isinstance(exit_layers, list):
                    exit_layers = [int(x) for x in exit_layers]
            else:
                exit_layers = [-1] * batch_size

            if exit_confidences is not None:
                if isinstance(exit_confidences, (int, float)):
                    exit_confidences = [float(exit_confidences)] * batch_size
                elif isinstance(exit_confidences, torch.Tensor):
                    exit_confidences = exit_confidences.cpu().numpy().tolist()
                elif isinstance(exit_confidences, list):
                    exit_confidences = [float(x) for x in exit_confidences]
            else:
                exit_confidences = [float(np.max(p)) for p in probs]

            for i in range(batch_size):
                record = {
                    'label': int(labels_np[i]),
                    'prediction': int(preds[i]),
                    'correct': bool(preds[i] == labels_np[i]),
                    'exit_layer': int(exit_layers[i]),
                    'exit_confidence': float(exit_confidences[i]),
                    'max_prob': float(np.max(probs[i])),
                    'entropy': float(-np.sum(probs[i] * np.log(probs[i] + 1e-10))),
                }
                records.append(record)

    return records


def analyze_exit_distribution(records):
    """Analyze the distribution of exit layers."""
    exit_layers = [r['exit_layer'] for r in records]
    exit_layers = np.array(exit_layers)

    # Overall distribution
    unique, counts = np.unique(exit_layers, return_counts=True)
    distribution = {int(k): int(v) for k, v in zip(unique, counts)}
    total = len(exit_layers)

    analysis = {
        'total_samples': total,
        'exit_distribution': distribution,
        'exit_percentages': {
            int(k): float(v / total * 100) for k, v in zip(unique, counts)
        },
        'mean_exit_layer': float(np.mean(exit_layers)),
        'std_exit_layer': float(np.std(exit_layers)),
        'median_exit_layer': float(np.median(exit_layers)),
        'min_exit_layer': int(np.min(exit_layers)),
        'max_exit_layer': int(np.max(exit_layers)),
    }

    # Compute theoretical compute savings
    max_layer = int(np.max(exit_layers))
    if max_layer > 0:
        compute_ratio = float(np.mean(exit_layers) / max_layer)
        analysis['compute_ratio'] = compute_ratio
        analysis['compute_savings_pct'] = float((1 - compute_ratio) * 100)

    return analysis


def analyze_per_class_exit(records, num_classes=10):
    """Analyze exit behavior per class."""
    per_class = defaultdict(lambda: {
        'exit_layers': [],
        'confidences': [],
        'correct': [],
        'entropies': [],
    })

    for r in records:
        cls = r['label']
        per_class[cls]['exit_layers'].append(r['exit_layer'])
        per_class[cls]['confidences'].append(r['exit_confidence'])
        per_class[cls]['correct'].append(r['correct'])
        per_class[cls]['entropies'].append(r['entropy'])

    class_stats = {}
    for cls in range(num_classes):
        if cls not in per_class:
            continue
        data = per_class[cls]
        exit_layers = np.array(data['exit_layers'])
        confidences = np.array(data['confidences'])
        correct = np.array(data['correct'])
        entropies = np.array(data['entropies'])

        class_stats[cls] = {
            'num_samples': len(exit_layers),
            'mean_exit_layer': float(np.mean(exit_layers)),
            'std_exit_layer': float(np.std(exit_layers)),
            'median_exit_layer': float(np.median(exit_layers)),
            'accuracy': float(np.mean(correct)),
            'mean_confidence': float(np.mean(confidences)),
            'mean_entropy': float(np.mean(entropies)),
            'early_exit_ratio': float(np.mean(exit_layers < np.max(exit_layers))),
        }

    return class_stats


def analyze_accuracy_vs_exit(records):
    """Analyze accuracy at each exit layer."""
    by_layer = defaultdict(lambda: {'correct': 0, 'total': 0, 'confidences': []})

    for r in records:
        layer = r['exit_layer']
        by_layer[layer]['total'] += 1
        if r['correct']:
            by_layer[layer]['correct'] += 1
        by_layer[layer]['confidences'].append(r['exit_confidence'])

    layer_accuracy = {}
    for layer in sorted(by_layer.keys()):
        data = by_layer[layer]
        layer_accuracy[int(layer)] = {
            'accuracy': float(data['correct'] / max(data['total'], 1)),
            'num_samples': data['total'],
            'mean_confidence': float(np.mean(data['confidences'])),
            'std_confidence': float(np.std(data['confidences'])),
        }

    return layer_accuracy


def analyze_confidence_calibration(records):
    """Analyze confidence calibration at exit points."""
    # Bin samples by confidence level
    bins = np.linspace(0, 1, 11)
    bin_data = defaultdict(lambda: {'correct': 0, 'total': 0})

    for r in records:
        conf = r['exit_confidence']
        bin_idx = min(int(conf * 10), 9)
        bin_label = f"{bins[bin_idx]:.1f}-{bins[bin_idx + 1]:.1f}"
        bin_data[bin_label]['total'] += 1
        if r['correct']:
            bin_data[bin_label]['correct'] += 1

    calibration = {}
    for bin_label in sorted(bin_data.keys()):
        data = bin_data[bin_label]
        calibration[bin_label] = {
            'expected_accuracy': float(
                (float(bin_label.split('-')[0]) + float(bin_label.split('-')[1])) / 2
            ),
            'actual_accuracy': float(
                data['correct'] / max(data['total'], 1)
            ),
            'num_samples': data['total'],
        }

    return calibration


# ============================================================================
# Plotting functions
# ============================================================================

def plot_exit_distribution(analysis, save_path):
    """Plot the exit layer distribution."""
    if not HAS_MATPLOTLIB:
        return

    dist = analysis['exit_distribution']
    layers = sorted(dist.keys())
    counts = [dist[l] for l in layers]
    total = sum(counts)
    percentages = [c / total * 100 for c in counts]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(layers)))
    bars = ax1.bar(layers, counts, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Exit Layer')
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Early Exit Layer Distribution')
    ax1.set_xticks(layers)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add count labels on bars
    for bar, count, pct in zip(bars, counts, percentages):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.005,
                 f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=7)

    # Cumulative distribution
    cumulative = np.cumsum(counts) / total * 100
    ax2.plot(layers, cumulative, 'o-', color='steelblue', linewidth=2, markersize=8)
    ax2.fill_between(layers, cumulative, alpha=0.2, color='steelblue')
    ax2.set_xlabel('Exit Layer')
    ax2.set_ylabel('Cumulative % of Samples')
    ax2.set_title('Cumulative Exit Distribution')
    ax2.set_xticks(layers)
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50%')
    ax2.axhline(y=90, color='orange', linestyle='--', alpha=0.5, label='90%')
    ax2.legend()

    # Add annotation for mean exit layer
    mean_exit = analysis['mean_exit_layer']
    ax2.axvline(x=mean_exit, color='green', linestyle='--', alpha=0.7)
    ax2.text(mean_exit + 0.1, 5, f'Mean: {mean_exit:.1f}',
             color='green', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_per_class_exit_layers(class_stats, save_path, class_names=None):
    """Plot per-class exit layer statistics."""
    if not HAS_MATPLOTLIB:
        return

    classes = sorted(class_stats.keys())
    mean_exits = [class_stats[c]['mean_exit_layer'] for c in classes]
    std_exits = [class_stats[c]['std_exit_layer'] for c in classes]
    accuracies = [class_stats[c]['accuracy'] for c in classes]

    # Sort by mean exit layer
    sorted_indices = np.argsort(mean_exits)
    classes_sorted = [classes[i] for i in sorted_indices]
    mean_sorted = [mean_exits[i] for i in sorted_indices]
    std_sorted = [std_exits[i] for i in sorted_indices]
    acc_sorted = [accuracies[i] for i in sorted_indices]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    # Mean exit layer per class
    if class_names:
        labels = [(class_names[c] if c < len(class_names) else f'C{c}')[:15] for c in classes_sorted]
    else:
        labels = [f'C{c}' for c in classes_sorted]

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(classes_sorted)))
    ax1.barh(range(len(classes_sorted)), mean_sorted, xerr=std_sorted,
             capsize=2, color=colors, edgecolor='black', linewidth=0.3)
    ax1.set_yticks(range(len(classes_sorted)))
    ax1.set_yticklabels(labels, fontsize=6)
    ax1.set_xlabel('Mean Exit Layer')
    ax1.set_title('Per-Class Mean Exit Layer (sorted)')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.invert_yaxis()

    # Scatter: exit layer vs accuracy
    scatter = ax2.scatter(
        [class_stats[c]['mean_exit_layer'] for c in classes],
        [class_stats[c]['accuracy'] * 100 for c in classes],
        c=[class_stats[c]['mean_entropy'] for c in classes],
        cmap='coolwarm', s=60, edgecolors='black', linewidth=0.5, alpha=0.8
    )
    ax2.set_xlabel('Mean Exit Layer')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Class Accuracy vs Mean Exit Layer\n(color = mean entropy)')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='Mean Entropy')

    # Annotate a few interesting points
    accs = [class_stats[c]['accuracy'] * 100 for c in classes]
    exits = [class_stats[c]['mean_exit_layer'] for c in classes]
    for i, c in enumerate(classes):
        if accs[i] < 90 or exits[i] > np.percentile(exits, 90):
            name = (class_names[c] if c < len(class_names) else f'C{c}')[:12] if class_names else f'C{c}'
            ax2.annotate(name, (exits[i], accs[i]),
                         fontsize=6, alpha=0.7,
                         xytext=(5, 5), textcoords='offset points')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_accuracy_vs_exit(layer_accuracy, save_path):
    """Plot accuracy at each exit layer."""
    if not HAS_MATPLOTLIB:
        return

    layers = sorted(layer_accuracy.keys())
    accs = [layer_accuracy[l]['accuracy'] * 100 for l in layers]
    counts = [layer_accuracy[l]['num_samples'] for l in layers]
    confs = [layer_accuracy[l]['mean_confidence'] * 100 for l in layers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy and confidence per layer
    ax1.plot(layers, accs, 'o-', color='steelblue', label='Accuracy',
             linewidth=2, markersize=8)
    ax1.plot(layers, confs, 's--', color='coral', label='Mean Confidence',
             linewidth=2, markersize=6)
    ax1.set_xlabel('Exit Layer')
    ax1.set_ylabel('Percentage (%)')
    ax1.set_title('Accuracy and Confidence by Exit Layer')
    ax1.set_xticks(layers)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)

    # Sample count per layer with accuracy as color
    max_count = max(counts)
    norm_accs = [a / 100 for a in accs]
    colors = plt.cm.RdYlGn(norm_accs)
    bars = ax2.bar(layers, counts, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Exit Layer')
    ax2.set_ylabel('Number of Samples')
    ax2.set_title('Sample Count by Exit Layer\n(color = accuracy: red=low, green=high)')
    ax2.set_xticks(layers)
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, acc in zip(bars, accs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max_count * 0.01,
                 f'{acc:.1f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confidence_calibration(calibration, save_path):
    """Plot confidence calibration diagram."""
    if not HAS_MATPLOTLIB:
        return

    bins = sorted(calibration.keys())
    expected = [calibration[b]['expected_accuracy'] * 100 for b in bins]
    actual = [calibration[b]['actual_accuracy'] * 100 for b in bins]
    counts = [calibration[b]['num_samples'] for b in bins]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Calibration curve
    ax1.plot([0, 100], [0, 100], 'k--', alpha=0.5, label='Perfect calibration')
    ax1.bar(expected, actual, width=8, alpha=0.6, color='steelblue',
            edgecolor='black', linewidth=0.5, label='Model calibration')
    ax1.set_xlabel('Expected Confidence (%)')
    ax1.set_ylabel('Actual Accuracy (%)')
    ax1.set_title('Confidence Calibration at Exit Points')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-5, 105)
    ax1.set_ylim(-5, 105)

    # Sample distribution across confidence bins
    ax2.bar(range(len(bins)), counts, color='coral', edgecolor='black',
            linewidth=0.5)
    ax2.set_xticks(range(len(bins)))
    ax2.set_xticklabels(bins, rotation=45, fontsize=8)
    ax2.set_xlabel('Confidence Bin')
    ax2.set_ylabel('Number of Samples')
    ax2.set_title('Sample Distribution Across Confidence Bins')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = get_device()

    print("=" * 70)
    print("EfficientEuroSAT Early Exit Analysis")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device:     {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # Load model
    model, model_config = load_model(args.checkpoint, device)
    print(f"Exit threshold: {model_config.get('exit_threshold', 0.9)}")
    print(f"Min exit layer:  {model_config.get('exit_min_layer', 4)}")

    # Load test data
    print("\nLoading test data...")
    _, _, test_loader, _ = get_eurosat_dataloaders(
        root=args.data_root,
        img_size=model_config.get('img_size', 224),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Collect exit statistics
    print("\nCollecting exit statistics on full test set...")
    records = collect_exit_statistics(model, test_loader, device)
    print(f"  Collected {len(records)} sample records")

    # 1. Exit distribution analysis
    print("\n--- Exit Distribution Analysis ---")
    dist_analysis = analyze_exit_distribution(records)
    print(f"  Mean exit layer:      {dist_analysis['mean_exit_layer']:.2f}")
    print(f"  Std exit layer:       {dist_analysis['std_exit_layer']:.2f}")
    print(f"  Median exit layer:    {dist_analysis['median_exit_layer']:.1f}")
    if 'compute_savings_pct' in dist_analysis:
        print(f"  Compute savings:      {dist_analysis['compute_savings_pct']:.1f}%")
    print(f"  Exit distribution:")
    for layer, pct in sorted(dist_analysis['exit_percentages'].items()):
        count = dist_analysis['exit_distribution'][layer]
        print(f"    Layer {layer:2d}: {count:6d} samples ({pct:5.1f}%)")

    plot_exit_distribution(
        dist_analysis,
        os.path.join(args.save_dir, 'exit_distribution.png')
    )

    # 2. Per-class analysis
    print("\n--- Per-Class Exit Analysis ---")
    class_stats = analyze_per_class_exit(records)

    # Find classes with extreme exit behavior
    sorted_by_exit = sorted(class_stats.items(), key=lambda x: x[1]['mean_exit_layer'])
    print("  Earliest exiting classes:")
    for cls, stats in sorted_by_exit[:5]:
        name = EUROSAT_CLASS_NAMES[cls] if cls < len(EUROSAT_CLASS_NAMES) else f'Class {cls}'
        print(f"    {cls:2d} ({name}): layer {stats['mean_exit_layer']:.2f}, "
              f"acc {stats['accuracy'] * 100:.1f}%")
    print("  Latest exiting classes:")
    for cls, stats in sorted_by_exit[-5:]:
        name = EUROSAT_CLASS_NAMES[cls] if cls < len(EUROSAT_CLASS_NAMES) else f'Class {cls}'
        print(f"    {cls:2d} ({name}): layer {stats['mean_exit_layer']:.2f}, "
              f"acc {stats['accuracy'] * 100:.1f}%")

    plot_per_class_exit_layers(
        class_stats,
        os.path.join(args.save_dir, 'per_class_exit_layers.png'),
        class_names=EUROSAT_CLASS_NAMES
    )

    # 3. Accuracy vs exit layer
    print("\n--- Accuracy vs Exit Layer ---")
    layer_accuracy = analyze_accuracy_vs_exit(records)
    for layer in sorted(layer_accuracy.keys()):
        stats = layer_accuracy[layer]
        print(f"  Layer {layer:2d}: accuracy={stats['accuracy'] * 100:.1f}%, "
              f"confidence={stats['mean_confidence'] * 100:.1f}%, "
              f"n={stats['num_samples']}")

    plot_accuracy_vs_exit(
        layer_accuracy,
        os.path.join(args.save_dir, 'accuracy_vs_exit_layer.png')
    )

    # 4. Confidence calibration
    print("\n--- Confidence Calibration ---")
    calibration = analyze_confidence_calibration(records)
    print(f"  {'Confidence Bin':<15} {'Expected':>10} {'Actual':>10} {'N':>8}")
    print(f"  {'-' * 45}")
    for bin_label in sorted(calibration.keys()):
        data = calibration[bin_label]
        print(f"  {bin_label:<15} {data['expected_accuracy'] * 100:>9.1f}% "
              f"{data['actual_accuracy'] * 100:>9.1f}% {data['num_samples']:>8}")

    plot_confidence_calibration(
        calibration,
        os.path.join(args.save_dir, 'confidence_calibration.png')
    )

    # 5. Compute overall statistics
    overall_correct = sum(1 for r in records if r['correct'])
    overall_accuracy = overall_correct / len(records)

    # Accuracy with and without early exit comparison
    correct_early = sum(
        1 for r in records
        if r['correct'] and r['exit_layer'] < dist_analysis['max_exit_layer']
    )
    total_early = sum(
        1 for r in records
        if r['exit_layer'] < dist_analysis['max_exit_layer']
    )
    correct_final = sum(
        1 for r in records
        if r['correct'] and r['exit_layer'] == dist_analysis['max_exit_layer']
    )
    total_final = sum(
        1 for r in records
        if r['exit_layer'] == dist_analysis['max_exit_layer']
    )

    print("\n--- Overall Statistics ---")
    print(f"  Overall accuracy:    {overall_accuracy * 100:.2f}%")
    if total_early > 0:
        print(f"  Early-exit accuracy: {correct_early / total_early * 100:.2f}% "
              f"({total_early} samples)")
    if total_final > 0:
        print(f"  Final-layer accuracy: {correct_final / total_final * 100:.2f}% "
              f"({total_final} samples)")

    # 6. Save all results
    all_results = {
        'checkpoint': args.checkpoint,
        'model_config': {k: v for k, v in model_config.items() if not callable(v)},
        'overall_accuracy': overall_accuracy,
        'exit_distribution': dist_analysis,
        'per_class_stats': {
            int(k): v for k, v in class_stats.items()
        },
        'accuracy_by_exit_layer': {
            int(k): v for k, v in layer_accuracy.items()
        },
        'confidence_calibration': calibration,
        'summary': {
            'total_samples': len(records),
            'overall_accuracy': overall_accuracy,
            'mean_exit_layer': dist_analysis['mean_exit_layer'],
            'compute_savings_pct': dist_analysis.get('compute_savings_pct', 0),
            'early_exit_samples': total_early,
            'final_layer_samples': total_final,
            'early_exit_accuracy': correct_early / max(total_early, 1),
            'final_layer_accuracy': correct_final / max(total_final, 1),
        },
    }

    results_path = os.path.join(args.save_dir, 'early_exit_analysis.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to: {results_path}")

    print(f"\nPlots saved to: {args.save_dir}")
    print("  - exit_distribution.png")
    print("  - per_class_exit_layers.png")
    print("  - accuracy_vs_exit_layer.png")
    print("  - confidence_calibration.png")


if __name__ == '__main__':
    main()
