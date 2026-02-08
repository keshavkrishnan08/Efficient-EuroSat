#!/usr/bin/env python3
"""Compare uncertainty estimation methods: UCAT vs baselines.

Runs four uncertainty / calibration approaches on the same test set and
produces a unified JSON report:

    1. **UCAT** -- learned per-head temperature as the uncertainty signal
       (mean temperature returned via ``return_temperatures=True``).
    2. **MC Dropout** -- stochastic forward passes with dropout active.
    3. **Deep Ensemble** -- average softmax over independently trained
       checkpoints (requires >= 2 baseline checkpoints).
    4. **Post-hoc Temperature Scaling** (Guo et al., 2017) -- single
       scalar temperature learned on the validation set.

For every method, top-1 accuracy and ECE (15 bins) are reported.

Usage:
    python compare_uncertainty_methods.py \\
        --ucat_checkpoint ./checkpoints/ucat_best.pth \\
        --baseline_checkpoints ./checkpoints/baseline_seed1.pth \\
                               ./checkpoints/baseline_seed2.pth \\
                               ./checkpoints/baseline_seed3.pth \\
        --dataset eurosat \\
        --save_dir ./results/uncertainty_comparison
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

from src.data.datasets import get_dataloaders, get_dataset_info
from src.models.efficient_vit import EfficientEuroSATViT
from src.models.baseline import BaselineViT
from src.evaluation.calibration import evaluate_calibration, compute_ece
from src.evaluation.uncertainty_baselines import (
    mc_dropout_inference,
    ensemble_inference,
    posthoc_temperature_scaling,
)
from src.utils.helpers import set_seed, get_device


# ======================================================================
# Argument parsing
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare uncertainty estimation methods',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--dataset', type=str, default='eurosat',
        choices=['eurosat', 'cifar100', 'resisc45'],
        help='Dataset to evaluate on',
    )
    parser.add_argument(
        '--data_root', type=str, default='./data',
        help='Root directory for datasets',
    )
    parser.add_argument(
        '--save_dir', type=str,
        default='./results/uncertainty_comparison',
        help='Directory to save comparison results',
    )
    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='Evaluation batch size',
    )
    parser.add_argument(
        '--ucat_checkpoint', type=str, required=True,
        help='Path to trained UCAT model checkpoint',
    )
    parser.add_argument(
        '--baseline_checkpoints', type=str, nargs='+', default=[],
        help='Path(s) to trained baseline model checkpoint(s). '
             'Provide multiple for ensemble evaluation.',
    )
    parser.add_argument(
        '--mc_forward_passes', type=int, default=30,
        help='Number of stochastic forward passes for MC Dropout',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility',
    )
    parser.add_argument(
        '--device', type=str, default=None,
        help='Device to use (auto-detected if not specified)',
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='Number of data loading workers',
    )
    return parser.parse_args()


# ======================================================================
# Model loading
# ======================================================================

def load_model(checkpoint_path: str, device: str):
    """Load a model from a saved checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to the ``.pth`` / ``.pt`` checkpoint file.
    device : str
        Target device.

    Returns
    -------
    tuple
        ``(model, model_config)`` where ``model`` is in eval mode on
        ``device``.
    """
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False,
    )
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


# ======================================================================
# UCAT inference
# ======================================================================

def run_ucat_inference(model, dataloader, device):
    """Run UCAT model inference and extract temperature-based uncertainty.

    Calls the model with ``return_temperatures=True`` and uses the mean
    learned temperature as the uncertainty signal.

    Parameters
    ----------
    model : nn.Module
        A trained ``EfficientEuroSATViT`` model.
    dataloader : DataLoader
        Evaluation dataloader.
    device : str
        Device identifier.

    Returns
    -------
    dict
        ``accuracy``, ``mean_probs``, ``temperatures``,
        ``predictive_entropy``.
    """
    model.eval()
    if hasattr(model, 'early_exit_enabled'):
        model.early_exit_enabled = False

    all_probs = []
    all_temps = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            output = model(images, return_temperatures=True)

            # return_temperatures=True yields (logits, temperatures, ...)
            if isinstance(output, tuple):
                logits = output[0]
                temperatures = output[1]
            else:
                logits = output
                temperatures = torch.zeros(
                    logits.shape[0], device=logits.device,
                )

            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())
            all_temps.append(temperatures.cpu().numpy())
            all_labels.append(labels.numpy())

    probs_all = np.concatenate(all_probs, axis=0)
    temps_all = np.concatenate(all_temps, axis=0)
    labels_all = np.concatenate(all_labels, axis=0)

    # Predictive entropy.
    log_probs = np.log(probs_all + 1e-12)
    predictive_entropy = -np.sum(probs_all * log_probs, axis=1)

    predictions = probs_all.argmax(axis=1)
    accuracy = float((predictions == labels_all).mean())

    return {
        "accuracy": accuracy,
        "mean_probs": probs_all,
        "temperatures": temps_all,
        "predictive_entropy": predictive_entropy,
    }


# ======================================================================
# ECE from pre-computed probabilities
# ======================================================================

def compute_ece_from_results(mean_probs, labels, n_bins=15):
    """Compute ECE given mean softmax probabilities and labels.

    Parameters
    ----------
    mean_probs : numpy.ndarray
        Softmax probabilities of shape ``(N, C)``.
    labels : numpy.ndarray
        Ground-truth labels of shape ``(N,)``.
    n_bins : int
        Number of calibration bins.

    Returns
    -------
    float
        The ECE value.
    """
    confidences = mean_probs.max(axis=1)
    predictions = mean_probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float64)
    ece, _mce, _bc, _ba, _bn = compute_ece(confidences, accuracies, n_bins=n_bins)
    return ece


# ======================================================================
# Main
# ======================================================================

def main():
    args = parse_args()
    set_seed(args.seed)

    # Device.
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = get_device()
    device_str = str(device)

    print("=" * 70)
    print("Uncertainty Method Comparison")
    print("=" * 70)
    print(f"Dataset:              {args.dataset}")
    print(f"UCAT checkpoint:      {args.ucat_checkpoint}")
    print(f"Baseline checkpoints: {args.baseline_checkpoints}")
    print(f"MC forward passes:    {args.mc_forward_passes}")
    print(f"Device:               {device}")
    print()

    os.makedirs(args.save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    dataset_info = get_dataset_info(args.dataset)
    num_classes = dataset_info["num_classes"]

    print(f"Loading {args.dataset} dataset ({num_classes} classes)...")
    train_loader, val_loader, test_loader, _class_weights = get_dataloaders(
        dataset_name=args.dataset,
        root=args.data_root,
        img_size=224,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    print(f"  Val batches:  {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # Collect ground-truth labels from the test set once.
    test_labels_list = []
    for _images, labels in test_loader:
        test_labels_list.append(labels.numpy())
    test_labels = np.concatenate(test_labels_list, axis=0)

    results: dict = {
        "dataset": args.dataset,
        "num_classes": num_classes,
        "methods": {},
    }

    # ------------------------------------------------------------------
    # 1. UCAT Inference
    # ------------------------------------------------------------------
    print("\n--- 1. UCAT Inference ---")
    t0 = time.time()
    ucat_model, ucat_config = load_model(args.ucat_checkpoint, device_str)
    ucat_results = run_ucat_inference(ucat_model, test_loader, device_str)
    ucat_ece = compute_ece_from_results(
        ucat_results["mean_probs"], test_labels,
    )
    ucat_time = time.time() - t0

    results["methods"]["ucat"] = {
        "accuracy": ucat_results["accuracy"],
        "ece": ucat_ece,
        "mean_temperature": float(ucat_results["temperatures"].mean()),
        "mean_predictive_entropy": float(
            ucat_results["predictive_entropy"].mean()
        ),
        "wall_time_s": ucat_time,
    }
    print(f"  Accuracy:           {ucat_results['accuracy'] * 100:.2f}%")
    print(f"  ECE:                {ucat_ece:.4f}")
    print(f"  Mean temperature:   {ucat_results['temperatures'].mean():.4f}")
    print(f"  Time:               {ucat_time:.1f}s")

    del ucat_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 2. MC Dropout (on first baseline checkpoint)
    # ------------------------------------------------------------------
    baseline_config = None
    if args.baseline_checkpoints:
        print("\n--- 2. MC Dropout ---")
        t0 = time.time()
        baseline_model, baseline_config = load_model(
            args.baseline_checkpoints[0], device_str,
        )
        mc_results = mc_dropout_inference(
            baseline_model, test_loader, device_str,
            n_forward=args.mc_forward_passes,
        )
        mc_ece = compute_ece_from_results(mc_results["mean_probs"], test_labels)
        mc_time = time.time() - t0

        results["methods"]["mc_dropout"] = {
            "accuracy": mc_results["accuracy"],
            "ece": mc_ece,
            "n_forward_passes": args.mc_forward_passes,
            "mean_predictive_entropy": float(
                mc_results["predictive_entropy"].mean()
            ),
            "mean_expected_entropy": float(
                mc_results["expected_entropy"].mean()
            ),
            "mean_mutual_information": float(
                mc_results["mutual_information"].mean()
            ),
            "mean_predictive_variance": float(
                mc_results["predictive_variance"].mean()
            ),
            "wall_time_s": mc_time,
        }
        print(f"  Accuracy:           {mc_results['accuracy'] * 100:.2f}%")
        print(f"  ECE:                {mc_ece:.4f}")
        print(f"  Forward passes:     {args.mc_forward_passes}")
        print(f"  Pred. entropy:      {mc_results['predictive_entropy'].mean():.4f}")
        print(f"  Expected entropy:   {mc_results['expected_entropy'].mean():.4f}")
        print(f"  Mutual info (epist):{mc_results['mutual_information'].mean():.4f}")
        print(f"  Pred. variance:     {mc_results['predictive_variance'].mean():.6f}")
        print(f"  Time:               {mc_time:.1f}s")

        del baseline_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print("\n--- 2. MC Dropout ---")
        print("  Skipped (no baseline checkpoints provided).")

    # ------------------------------------------------------------------
    # 3. Deep Ensemble (if multiple baseline checkpoints provided)
    # ------------------------------------------------------------------
    if len(args.baseline_checkpoints) > 1:
        print("\n--- 3. Deep Ensemble ---")
        t0 = time.time()
        ens_results = ensemble_inference(
            checkpoint_paths=args.baseline_checkpoints,
            model_config=baseline_config,
            dataloader=test_loader,
            device=device_str,
        )
        ens_ece = compute_ece_from_results(
            ens_results["mean_probs"], test_labels,
        )
        ens_time = time.time() - t0

        results["methods"]["deep_ensemble"] = {
            "accuracy": ens_results["accuracy"],
            "ece": ens_ece,
            "n_models": ens_results["n_models"],
            "mean_predictive_entropy": float(
                ens_results["predictive_entropy"].mean()
            ),
            "mean_mutual_information": float(
                ens_results["mutual_information"].mean()
            ),
            "mean_predictive_variance": float(
                ens_results["predictive_variance"].mean()
            ),
            "wall_time_s": ens_time,
        }
        print(f"  Accuracy:           {ens_results['accuracy'] * 100:.2f}%")
        print(f"  ECE:                {ens_ece:.4f}")
        print(f"  Ensemble members:   {ens_results['n_models']}")
        print(f"  Mutual info (epist):{ens_results['mutual_information'].mean():.4f}")
        print(f"  Pred. variance:     {ens_results['predictive_variance'].mean():.6f}")
        print(f"  Time:               {ens_time:.1f}s")
    else:
        print("\n--- 3. Deep Ensemble ---")
        print("  Skipped (only 1 baseline checkpoint provided).")

    # ------------------------------------------------------------------
    # 4. Post-hoc Temperature Scaling
    # ------------------------------------------------------------------
    if args.baseline_checkpoints:
        print("\n--- 4. Post-hoc Temperature Scaling ---")
        t0 = time.time()
        baseline_model_ts, _ = load_model(
            args.baseline_checkpoints[0], device_str,
        )
        ts_results = posthoc_temperature_scaling(
            baseline_model_ts, val_loader, test_loader, device_str,
        )
        ts_time = time.time() - t0

        results["methods"]["temperature_scaling"] = {
            "accuracy": ts_results["accuracy"],
            "learned_temperature": ts_results["learned_temperature"],
            "ece_before": ts_results["ece_before"],
            "ece_after": ts_results["ece_after"],
            "wall_time_s": ts_time,
        }
        print(f"  Accuracy:           {ts_results['accuracy'] * 100:.2f}%")
        print(f"  Learned T:          {ts_results['learned_temperature']:.4f}")
        print(f"  ECE before:         {ts_results['ece_before']:.4f}")
        print(f"  ECE after:          {ts_results['ece_after']:.4f}")
        print(f"  Time:               {ts_time:.1f}s")

        del baseline_model_ts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print("\n--- 4. Post-hoc Temperature Scaling ---")
        print("  Skipped (no baseline checkpoints provided).")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results_path = os.path.join(args.save_dir, 'uncertainty_comparison.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("UNCERTAINTY METHOD COMPARISON")
    print("=" * 70)
    print(f"{'Method':<25} {'Accuracy':>10} {'ECE':>10}")
    print("-" * 45)

    methods = results["methods"]

    print(
        f"{'UCAT':<25} "
        f"{methods['ucat']['accuracy'] * 100:>9.2f}% "
        f"{methods['ucat']['ece']:>10.4f}"
    )
    if 'mc_dropout' in methods:
        print(
            f"{'MC Dropout':<25} "
            f"{methods['mc_dropout']['accuracy'] * 100:>9.2f}% "
            f"{methods['mc_dropout']['ece']:>10.4f}"
        )
    if 'deep_ensemble' in methods:
        print(
            f"{'Deep Ensemble':<25} "
            f"{methods['deep_ensemble']['accuracy'] * 100:>9.2f}% "
            f"{methods['deep_ensemble']['ece']:>10.4f}"
        )
    if 'temperature_scaling' in methods:
        print(
            f"{'Temp Scaling (before)':<25} "
            f"{methods['temperature_scaling']['accuracy'] * 100:>9.2f}% "
            f"{methods['temperature_scaling']['ece_before']:>10.4f}"
        )
        print(
            f"{'Temp Scaling (after)':<25} "
            f"{methods['temperature_scaling']['accuracy'] * 100:>9.2f}% "
            f"{methods['temperature_scaling']['ece_after']:>10.4f}"
        )
    print("=" * 70)


if __name__ == '__main__':
    main()
