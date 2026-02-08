#!/usr/bin/env python3
"""
Comprehensive evaluation script for trained EfficientEuroSAT models.

Loads a saved checkpoint and runs full evaluation including:
- Overall and per-class accuracy
- Latency measurements
- Confusion matrix generation
- Attention map visualization
- Learned parameter analysis

Usage:
    python evaluate.py --checkpoint ./checkpoints/best_model.pth
    python evaluate.py --checkpoint ./checkpoints/best_model.pth --save_dir ./eval_results
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

from src.data.eurosat import get_eurosat_dataloaders, EUROSAT_CLASS_NAMES
from src.models.efficient_vit import EfficientEuroSATViT, create_efficient_eurosat_tiny
from src.models.baseline import BaselineViT
from src.utils.helpers import set_seed, get_device, count_parameters
from src.evaluation.confusion import (
    plot_confusion_matrix,
    plot_per_class_accuracy,
)
from src.utils.visualization import (
    visualize_attention,
    visualize_all_learned_params,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate a trained EfficientEuroSAT model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory for dataset')
    parser.add_argument('--save_dir', type=str, default='./eval_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (auto-detected if not specified)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Evaluation batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--num_attention_samples', type=int, default=10,
                        help='Number of samples for attention visualization')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path, device):
    """Load a model from a saved checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model config from checkpoint
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

    print(f"  Model type: {model_type}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Best val accuracy: {checkpoint.get('best_val_accuracy', 'unknown')}")

    return model, model_type, model_config


def evaluate_accuracy(model, test_loader, device, num_classes=10):
    """Run full accuracy evaluation on the test set."""
    all_preds = []
    all_labels = []
    all_logits = []
    total_loss = 0.0
    num_batches = 0

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # Handle EfficientEuroSAT output format (may return dict)
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs

            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()
            num_batches += 1

            preds = logits.argmax(dim=-1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_logits.append(logits.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_logits = np.concatenate(all_logits)
    avg_loss = total_loss / max(num_batches, 1)

    # Overall accuracy
    overall_acc = float(np.mean(all_preds == all_labels))

    # Per-class accuracy
    per_class_acc = np.zeros(num_classes)
    for c in range(num_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            per_class_acc[c] = float(np.mean(all_preds[mask] == c))

    # Confusion matrix
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(all_labels, all_preds):
        conf_matrix[int(t), int(p)] += 1

    # Top-5 accuracy
    top5_correct = 0
    for i in range(len(all_logits)):
        top5_preds = np.argsort(all_logits[i])[-5:]
        if all_labels[i] in top5_preds:
            top5_correct += 1
    top5_acc = top5_correct / len(all_labels)

    results = {
        'overall_accuracy': float(overall_acc),
        'top5_accuracy': float(top5_acc),
        'average_loss': float(avg_loss),
        'per_class_accuracy': {int(k): float(v) for k, v in enumerate(per_class_acc)},
        'confusion_matrix': conf_matrix.tolist(),
        'num_samples': len(all_labels),
        'num_correct': int(np.sum(all_preds == all_labels)),
    }

    # Find worst and best classes
    sorted_classes = np.argsort(per_class_acc)
    results['worst_classes'] = [
        {'class_id': int(c), 'accuracy': float(per_class_acc[c])}
        for c in sorted_classes[:5]
    ]
    results['best_classes'] = [
        {'class_id': int(c), 'accuracy': float(per_class_acc[c])}
        for c in sorted_classes[-5:][::-1]
    ]

    return results, all_preds, all_labels, conf_matrix, per_class_acc


def measure_latency(model, test_loader, device, num_runs=100):
    """Measure inference latency."""
    model.eval()

    # Get a single batch for latency measurement
    images, _ = next(iter(test_loader))
    single_image = images[:1].to(device)
    batch_images = images.to(device)

    # Warm up
    with torch.no_grad():
        for _ in range(10):
            _ = model(single_image)

    # Measure single-image latency
    single_latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(single_image)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000  # ms
            single_latencies.append(elapsed)

    # Measure batch latency
    batch_latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(batch_images)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000  # ms
            batch_latencies.append(elapsed)

    single_arr = np.array(single_latencies)
    batch_arr = np.array(batch_latencies)
    batch_size = batch_images.size(0)

    latency_stats = {
        'single_mean_ms': float(np.mean(single_arr)),
        'single_std_ms': float(np.std(single_arr)),
        'single_p50_ms': float(np.percentile(single_arr, 50)),
        'single_p95_ms': float(np.percentile(single_arr, 95)),
        'single_p99_ms': float(np.percentile(single_arr, 99)),
        'batch_mean_ms': float(np.mean(batch_arr)),
        'batch_std_ms': float(np.std(batch_arr)),
        'throughput_imgs_per_sec': float(
            batch_size / (np.mean(batch_arr) / 1000)
        ),
    }
    return latency_stats


def collect_early_exit_stats(model, test_loader, device):
    """Collect early exit statistics if model supports it."""
    if not hasattr(model, 'use_early_exit') or not model.use_early_exit:
        return None

    exit_layers = []
    exit_confidences = []
    exit_per_class = defaultdict(list)

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            if isinstance(outputs, dict) and 'exit_layer' in outputs:
                batch_exit_layers = outputs['exit_layer']
                batch_confidences = outputs.get('exit_confidence', [None] * len(labels))

                if isinstance(batch_exit_layers, (int, float)):
                    batch_exit_layers = [batch_exit_layers] * len(labels)
                if isinstance(batch_confidences, (int, float)):
                    batch_confidences = [batch_confidences] * len(labels)

                for i, (el, ec, lbl) in enumerate(
                    zip(batch_exit_layers, batch_confidences, labels.cpu().numpy())
                ):
                    exit_layers.append(int(el) if el is not None else -1)
                    exit_confidences.append(
                        float(ec) if ec is not None else -1.0
                    )
                    exit_per_class[int(lbl)].append(
                        int(el) if el is not None else -1
                    )

    if not exit_layers:
        return None

    exit_layers = np.array(exit_layers)
    exit_confidences = np.array(exit_confidences)

    stats = {
        'avg_exit_layer': float(np.mean(exit_layers)),
        'std_exit_layer': float(np.std(exit_layers)),
        'median_exit_layer': float(np.median(exit_layers)),
        'exit_layer_distribution': {
            int(k): int(v) for k, v in
            zip(*np.unique(exit_layers, return_counts=True))
        },
        'avg_exit_confidence': float(np.mean(exit_confidences[exit_confidences >= 0])),
        'early_exit_ratio': float(np.mean(exit_layers < exit_layers.max())),
        'per_class_avg_exit': {
            int(c): float(np.mean(layers))
            for c, layers in exit_per_class.items()
        },
    }
    return stats


def main():
    args = parse_args()
    set_seed(args.seed)

    # Setup device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = get_device()

    print("=" * 70)
    print("EfficientEuroSAT Comprehensive Evaluation")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device:     {device}")

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Load model
    model, model_type, model_config = load_model_from_checkpoint(
        args.checkpoint, device
    )
    total_params, trainable_params = count_parameters(model)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Load data
    print("\nLoading EuroSAT test data...")
    _, _, test_loader, _ = get_eurosat_dataloaders(
        root=args.data_root,
        img_size=model_config.get('img_size', 224),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"  Test batches: {len(test_loader)}")

    # 1. Accuracy evaluation
    print("\n--- Accuracy Evaluation ---")
    acc_results, all_preds, all_labels, conf_matrix, per_class_acc = \
        evaluate_accuracy(model, test_loader, device)
    print(f"  Overall Accuracy: {acc_results['overall_accuracy'] * 100:.2f}%")
    print(f"  Top-5 Accuracy:   {acc_results['top5_accuracy'] * 100:.2f}%")
    print(f"  Average Loss:     {acc_results['average_loss']:.4f}")
    print(f"  Worst classes:")
    for cls_info in acc_results['worst_classes']:
        cid = cls_info['class_id']
        class_name = EUROSAT_CLASS_NAMES[cid] if cid < len(EUROSAT_CLASS_NAMES) else f"Class {cid}"
        print(f"    {cls_info['class_id']:2d} ({class_name}): {cls_info['accuracy'] * 100:.1f}%")

    # 2. Latency measurement
    print("\n--- Latency Measurement ---")
    latency_results = measure_latency(model, test_loader, device)
    print(f"  Single image (mean):  {latency_results['single_mean_ms']:.2f} ms")
    print(f"  Single image (p99):   {latency_results['single_p99_ms']:.2f} ms")
    print(f"  Batch (mean):         {latency_results['batch_mean_ms']:.2f} ms")
    print(f"  Throughput:           {latency_results['throughput_imgs_per_sec']:.0f} img/s")

    # 3. Early exit statistics
    print("\n--- Early Exit Analysis ---")
    exit_stats = collect_early_exit_stats(model, test_loader, device)
    if exit_stats is not None:
        print(f"  Average exit layer:   {exit_stats['avg_exit_layer']:.2f}")
        print(f"  Median exit layer:    {exit_stats['median_exit_layer']:.1f}")
        print(f"  Early exit ratio:     {exit_stats['early_exit_ratio'] * 100:.1f}%")
        print(f"  Avg exit confidence:  {exit_stats['avg_exit_confidence']:.4f}")
        print(f"  Exit distribution:")
        for layer, count in sorted(exit_stats['exit_layer_distribution'].items()):
            print(f"    Layer {layer}: {count} samples")
    else:
        print("  Early exit not enabled for this model.")

    # 4. Generate visualizations
    print("\n--- Generating Visualizations ---")

    # Confusion matrix
    cm_path = os.path.join(args.save_dir, 'confusion_matrix.png')
    plot_confusion_matrix(conf_matrix, save_path=cm_path, class_names=EUROSAT_CLASS_NAMES)
    print(f"  Confusion matrix saved to: {cm_path}")

    # Per-class accuracy bar chart
    pca_path = os.path.join(args.save_dir, 'per_class_accuracy.png')
    plot_per_class_accuracy(per_class_acc, save_path=pca_path, class_names=EUROSAT_CLASS_NAMES)
    print(f"  Per-class accuracy plot saved to: {pca_path}")

    # Attention maps
    print("  Generating attention maps...")
    attn_dir = os.path.join(args.save_dir, 'attention_maps')
    os.makedirs(attn_dir, exist_ok=True)

    test_images, test_labels = [], []
    for images, labels in test_loader:
        test_images.append(images)
        test_labels.append(labels)
        if sum(img.size(0) for img in test_images) >= args.num_attention_samples:
            break

    test_images = torch.cat(test_images)[:args.num_attention_samples]

    for i in range(len(test_images)):
        attn_path = os.path.join(attn_dir, f'attention_sample_{i:03d}.png')
        visualize_attention(model, test_images[i:i+1].to(device), save_path=attn_path)
    print(f"  Attention maps saved to: {attn_dir}")

    # Learned parameters visualization
    if model_type == 'efficient_eurosat':
        print("  Generating learned parameter plots...")
        params_dir = os.path.join(args.save_dir, 'learned_params')
        visualize_all_learned_params(model, save_dir=params_dir)
        print(f"  Learned parameters plots saved to: {params_dir}")

    # 5. Compile and save all results
    all_results = {
        'checkpoint': args.checkpoint,
        'model_type': model_type,
        'model_config': model_config,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'accuracy': acc_results,
        'latency': latency_results,
        'early_exit': exit_stats,
    }

    results_path = os.path.join(args.save_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
    print(f"\nAll results saved to: {results_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  Model:              {model_type}")
    print(f"  Test Accuracy:      {acc_results['overall_accuracy'] * 100:.2f}%")
    print(f"  Top-5 Accuracy:     {acc_results['top5_accuracy'] * 100:.2f}%")
    print(f"  Latency (single):   {latency_results['single_mean_ms']:.2f} ms")
    print(f"  Throughput:         {latency_results['throughput_imgs_per_sec']:.0f} img/s")
    print(f"  Parameters:         {total_params:,}")
    if exit_stats:
        print(f"  Avg Exit Layer:     {exit_stats['avg_exit_layer']:.2f}")
    print(f"  Results saved to:   {args.save_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
