#!/usr/bin/env python3
"""Verify theoretical properties of the UCAT decomposition.

This script performs three analyses on a trained EfficientEuroSAT model
with UCAT decomposition:

    A. Gradient Independence -- cross-gradient norms between aleatoric
       and epistemic parameters should be near zero.
    B. Temperature Bounds -- aleatoric and epistemic temperatures
       should satisfy tau >= tau_min and be finite.
    C. Convergence -- temperature dynamics and training loss trends
       from logged results.

Usage:
    python theoretical_analysis.py --checkpoint checkpoints/decomp_best.pt
    python theoretical_analysis.py --checkpoint checkpoints/decomp_best.pt \\
        --results_file results/decomp_with_losses.json
    python theoretical_analysis.py --checkpoint checkpoints/decomp_best.pt \\
        --dataset eurosat --batch_size 32
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import glob
import json

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.models.efficient_vit import EfficientEuroSATViT
from src.data.datasets import get_dataloaders, get_dataset_info
from src.utils.helpers import set_seed, get_device


# ======================================================================
# Argument parsing
# ======================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Verify theoretical properties of the UCAT decomposition',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Path to model checkpoint (.pt file)',
    )
    parser.add_argument(
        '--data_root', type=str, default='./data',
        help='Root directory for dataset files',
    )
    parser.add_argument(
        '--dataset', type=str, default='eurosat',
        help='Dataset name (eurosat, cifar100, resisc45)',
    )
    parser.add_argument(
        '--save_dir', type=str, default='./analysis_results/theoretical',
        help='Directory to save analysis outputs',
    )
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='Batch size for data loading',
    )
    parser.add_argument(
        '--results_file', type=str, default=None,
        help='Path to training results JSON (auto-detected from checkpoint name if omitted)',
    )
    return parser.parse_args()


# ======================================================================
# Model loading
# ======================================================================

def load_model(checkpoint_path, device):
    """Load model from checkpoint with use_decomposition=True.

    Parameters
    ----------
    checkpoint_path : str
        Path to the saved checkpoint.
    device : torch.device
        Device to place the model on.

    Returns
    -------
    model : EfficientEuroSATViT
        Loaded model in eval mode.
    config : dict
        Model configuration from the checkpoint.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get('model_config', {})

    model = EfficientEuroSATViT(
        num_classes=config.get('num_classes', 10),
        use_learned_temp=config.get('use_learned_temp', True),
        use_early_exit=config.get('use_early_exit', True),
        use_learned_dropout=config.get('use_learned_dropout', True),
        use_learned_residual=config.get('use_learned_residual', True),
        use_temp_annealing=config.get('use_temp_annealing', True),
        use_decomposition=True,
    )
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()

    # Disable early exit for deterministic analysis
    if hasattr(model, 'early_exit_enabled'):
        model.early_exit_enabled = False

    return model, config


# ======================================================================
# A. Gradient Independence Analysis
# ======================================================================

def classify_parameters(model):
    """Separate model parameters into aleatoric and epistemic groups.

    Aleatoric parameters: those belonging to the temp_predictor module.
    Epistemic parameters: all other learned temperature parameters
    (i.e., LearnableTemperature raw_tau in attention blocks).

    Parameters
    ----------
    model : EfficientEuroSATViT
        The model to inspect.

    Returns
    -------
    aleatoric_params : list of (name, Parameter)
        Parameters from temp_predictor.
    epistemic_params : list of (name, Parameter)
        Learned temperature parameters outside temp_predictor.
    """
    aleatoric_params = []
    epistemic_params = []

    for name, param in model.named_parameters():
        if 'temp_predictor' in name:
            aleatoric_params.append((name, param))
        elif 'learned_temp' in name or 'raw_tau' in name:
            epistemic_params.append((name, param))

    return aleatoric_params, epistemic_params


def analyze_gradient_independence(model, test_loader, device, max_batches=5):
    """Compute cross-gradient norms between aleatoric and epistemic losses.

    For a batch of test data:
      - Compute the aleatoric loss (task loss contribution through tau_a)
      - Compute gradients of that loss w.r.t. epistemic params
      - Compute gradients of epistemic-related loss w.r.t. aleatoric params
      - Report L2 norms (should be near 0 for independent decomposition)

    Parameters
    ----------
    model : EfficientEuroSATViT
        Model with decomposition enabled.
    test_loader : DataLoader
        Test data loader.
    device : torch.device
        Computation device.
    max_batches : int
        Maximum number of batches to process.

    Returns
    -------
    dict
        Analysis results with gradient norm statistics.
    """
    aleatoric_params, epistemic_params = classify_parameters(model)

    if not aleatoric_params:
        return {
            'status': 'skipped',
            'reason': 'No aleatoric parameters (temp_predictor) found in model',
        }
    if not epistemic_params:
        return {
            'status': 'skipped',
            'reason': 'No epistemic (learned temperature) parameters found in model',
        }

    # Enable gradients temporarily
    model.train()
    criterion = torch.nn.CrossEntropyLoss()

    aleatoric_cross_norms = []
    epistemic_cross_norms = []

    batch_count = 0
    for images, labels in test_loader:
        if batch_count >= max_batches:
            break
        batch_count += 1

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass with temperatures
        model.zero_grad()
        result = model(images, training_progress=1.0, return_temperatures=True)

        if len(result) == 4:
            logits, _temps, tau_a_mean, tau_e_mean = result
        elif len(result) == 2:
            logits, _temps = result
            tau_a_mean = None
            tau_e_mean = None
        else:
            logits = result
            tau_a_mean = None
            tau_e_mean = None

        # --- Aleatoric loss: standard task loss ---
        task_loss = criterion(logits, labels)

        # Grad of task loss w.r.t. epistemic params (cross-gradient)
        epistemic_param_list = [p for _, p in epistemic_params]
        try:
            grads = torch.autograd.grad(
                task_loss, epistemic_param_list,
                retain_graph=True, allow_unused=True,
            )
            cross_norm = 0.0
            for g in grads:
                if g is not None:
                    cross_norm += g.norm().item() ** 2
            aleatoric_cross_norms.append(float(np.sqrt(cross_norm)))
        except RuntimeError:
            aleatoric_cross_norms.append(float('nan'))

        # --- Epistemic loss: temperature regularization ---
        # Use epistemic temperature magnitude as a proxy loss
        if tau_e_mean is not None and tau_e_mean.requires_grad:
            epistemic_loss = tau_e_mean.mean()
        else:
            # Fallback: collect raw_tau from blocks
            tau_vals = []
            for block in model.blocks:
                if hasattr(block.attn, 'learned_temp') and block.attn.learned_temp is not None:
                    tau_vals.append(block.attn.learned_temp().mean())
            if tau_vals:
                epistemic_loss = torch.stack(tau_vals).mean()
            else:
                epistemic_loss = None

        if epistemic_loss is not None:
            aleatoric_param_list = [p for _, p in aleatoric_params]
            try:
                grads = torch.autograd.grad(
                    epistemic_loss, aleatoric_param_list,
                    retain_graph=True, allow_unused=True,
                )
                cross_norm = 0.0
                for g in grads:
                    if g is not None:
                        cross_norm += g.norm().item() ** 2
                epistemic_cross_norms.append(float(np.sqrt(cross_norm)))
            except RuntimeError:
                epistemic_cross_norms.append(float('nan'))
        else:
            epistemic_cross_norms.append(float('nan'))

        model.zero_grad()

    model.eval()

    result = {
        'status': 'completed',
        'num_batches': batch_count,
        'num_aleatoric_params': sum(p.numel() for _, p in aleatoric_params),
        'num_epistemic_params': sum(p.numel() for _, p in epistemic_params),
        'aleatoric_param_names': [n for n, _ in aleatoric_params],
        'epistemic_param_names': [n for n, _ in epistemic_params],
    }

    if aleatoric_cross_norms:
        valid = [v for v in aleatoric_cross_norms if not np.isnan(v)]
        result['grad_aleatoric_loss_wrt_epistemic_params'] = {
            'values': aleatoric_cross_norms,
            'mean': float(np.mean(valid)) if valid else float('nan'),
            'max': float(np.max(valid)) if valid else float('nan'),
        }

    if epistemic_cross_norms:
        valid = [v for v in epistemic_cross_norms if not np.isnan(v)]
        result['grad_epistemic_loss_wrt_aleatoric_params'] = {
            'values': epistemic_cross_norms,
            'mean': float(np.mean(valid)) if valid else float('nan'),
            'max': float(np.max(valid)) if valid else float('nan'),
        }

    return result


# ======================================================================
# B. Temperature Bounds Analysis
# ======================================================================

def analyze_temperature_bounds(model, test_loader, device):
    """Collect tau_a and tau_e across the test set and verify bounds.

    Parameters
    ----------
    model : EfficientEuroSATViT
        Model with decomposition enabled.
    test_loader : DataLoader
        Test data loader.
    device : torch.device
        Computation device.

    Returns
    -------
    dict
        Temperature statistics and bound verification results.
    """
    all_tau_a = []
    all_tau_e = []

    with torch.no_grad():
        for images, _labels in test_loader:
            images = images.to(device)
            result = model(images, training_progress=1.0, return_temperatures=True)

            if len(result) == 4:
                _logits, _total, tau_a, tau_e = result
                all_tau_a.append(tau_a.cpu())
                # tau_e may be scalar; expand to match batch
                if tau_e.dim() == 0:
                    all_tau_e.append(tau_e.expand(_logits.shape[0]).cpu())
                else:
                    all_tau_e.append(tau_e.cpu())
            elif len(result) == 2:
                _logits, temps = result
                all_tau_a.append(temps.cpu())
                all_tau_e.append(torch.zeros(temps.shape[0]))
            else:
                break

    if not all_tau_a:
        return {
            'status': 'skipped',
            'reason': 'Could not collect temperatures from model forward pass',
        }

    tau_a_arr = torch.cat(all_tau_a).numpy()
    tau_e_arr = torch.cat(all_tau_e).numpy()

    # Get tau_min from model if available
    tau_min = None
    if hasattr(model, 'temp_predictor') and model.temp_predictor is not None:
        tau_min = getattr(model.temp_predictor, 'tau_min', None)
    if tau_min is None:
        # Try to get from attention blocks
        for block in model.blocks:
            if hasattr(block.attn, 'learned_temp') and block.attn.learned_temp is not None:
                tau_min = getattr(block.attn.learned_temp, 'tau_min', None)
                break

    def _stats(arr, name):
        """Compute statistics for a temperature array."""
        total = len(arr)
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        finite = arr[np.isfinite(arr)]

        stats = {
            'count': total,
            'num_nan': n_nan,
            'pct_nan': float(n_nan / total * 100) if total > 0 else 0.0,
            'num_inf': n_inf,
            'pct_inf': float(n_inf / total * 100) if total > 0 else 0.0,
        }

        if len(finite) > 0:
            stats['min'] = float(np.min(finite))
            stats['max'] = float(np.max(finite))
            stats['mean'] = float(np.mean(finite))
            stats['std'] = float(np.std(finite))
        else:
            stats['min'] = float('nan')
            stats['max'] = float('nan')
            stats['mean'] = float('nan')
            stats['std'] = float('nan')

        # Verify lower bound
        if tau_min is not None and len(finite) > 0:
            violations = int((finite < tau_min - 1e-6).sum())
            stats['tau_min'] = float(tau_min)
            stats['bound_violations'] = violations
            stats['bound_satisfied'] = violations == 0
        else:
            stats['tau_min'] = tau_min
            stats['bound_violations'] = None
            stats['bound_satisfied'] = None

        return stats

    results = {
        'status': 'completed',
        'total_samples': len(tau_a_arr),
        'tau_a': _stats(tau_a_arr, 'tau_a'),
        'tau_e': _stats(tau_e_arr, 'tau_e'),
    }

    # Store raw arrays for plotting (convert to lists for JSON serialization)
    results['_tau_a_values'] = tau_a_arr.tolist()
    results['_tau_e_values'] = tau_e_arr.tolist()

    return results


# ======================================================================
# C. Convergence Analysis
# ======================================================================

def auto_detect_results_file(checkpoint_path):
    """Try to find a matching results JSON from the checkpoint name.

    Parameters
    ----------
    checkpoint_path : str
        Path to the checkpoint file.

    Returns
    -------
    str or None
        Path to the detected results file, or None.
    """
    if not checkpoint_path:
        return None

    basename = os.path.splitext(os.path.basename(checkpoint_path))[0]
    # Remove common suffixes like '_best', '_final', '_epoch100'
    for suffix in ('_best', '_final', '_last', '_checkpoint'):
        basename = basename.replace(suffix, '')

    # Search in common result directories
    search_dirs = [
        os.path.dirname(checkpoint_path),
        os.path.join(os.path.dirname(checkpoint_path), '..', 'results'),
        './results',
        './ablation_results',
        './ablation_results/individual',
    ]

    for search_dir in search_dirs:
        search_dir = os.path.abspath(search_dir)
        if not os.path.isdir(search_dir):
            continue
        pattern = os.path.join(search_dir, f'*{basename}*.json')
        matches = glob.glob(pattern)
        if matches:
            return matches[0]

    return None


def analyze_convergence(results_file):
    """Analyze convergence from training results.

    Parameters
    ----------
    results_file : str or None
        Path to training results JSON.

    Returns
    -------
    dict
        Convergence analysis results.
    """
    if results_file is None or not os.path.isfile(results_file):
        return {
            'status': 'skipped',
            'reason': f'Results file not found: {results_file}',
        }

    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as exc:
        return {
            'status': 'skipped',
            'reason': f'Could not load results file: {exc}',
        }

    result = {
        'status': 'completed',
        'results_file': results_file,
    }

    # Check for temperature dynamics
    temp_dynamics = None
    if 'training' in data and isinstance(data['training'], dict):
        temp_dynamics = data['training'].get('temperature_dynamics')
    elif 'temperature_dynamics' in data:
        temp_dynamics = data['temperature_dynamics']

    if temp_dynamics is not None:
        result['temperature_dynamics'] = temp_dynamics
        # Extract tau_e over epochs if available
        tau_e_over_epochs = None
        if isinstance(temp_dynamics, dict):
            tau_e_over_epochs = temp_dynamics.get('tau_e_per_epoch') or \
                                temp_dynamics.get('epistemic_temp_per_epoch') or \
                                temp_dynamics.get('tau_e')
        elif isinstance(temp_dynamics, list):
            tau_e_over_epochs = temp_dynamics

        if tau_e_over_epochs is not None:
            result['tau_e_over_epochs'] = tau_e_over_epochs
    else:
        result['temperature_dynamics'] = None

    # Check for training loss history
    loss_history = None
    for key_path in [
        ('training', 'loss_history'),
        ('training', 'train_losses'),
        ('loss_history',),
        ('train_losses',),
        ('training', 'epoch_losses'),
    ]:
        obj = data
        found = True
        for k in key_path:
            if isinstance(obj, dict) and k in obj:
                obj = obj[k]
            else:
                found = False
                break
        if found and isinstance(obj, list) and len(obj) > 0:
            loss_history = obj
            break

    if loss_history is not None:
        arr = np.array(loss_history, dtype=np.float64)
        # Check for monotonic decrease trend (using linear regression slope)
        epochs = np.arange(len(arr), dtype=np.float64)
        if len(arr) > 1:
            slope, _intercept = np.polyfit(epochs, arr, 1)
            # Count consecutive decreases
            diffs = np.diff(arr)
            n_decreasing = int((diffs < 0).sum())
            n_total = len(diffs)

            result['loss_history'] = {
                'num_epochs': len(arr),
                'initial_loss': float(arr[0]),
                'final_loss': float(arr[-1]),
                'min_loss': float(np.min(arr)),
                'slope': float(slope),
                'is_decreasing_trend': slope < 0,
                'pct_decreasing_steps': float(n_decreasing / n_total * 100) if n_total > 0 else 0.0,
            }
            result['_loss_values'] = arr.tolist()
        else:
            result['loss_history'] = {'num_epochs': len(arr)}
    else:
        result['loss_history'] = None

    return result


# ======================================================================
# Figure generation
# ======================================================================

def generate_figure(gradient_results, bounds_results, convergence_results, save_path):
    """Generate the theoretical analysis figure with 2 subplots.

    Subplot 1: Gradient norms bar chart (Analysis A).
    Subplot 2: Temperature bounds histogram (Analysis B).

    Parameters
    ----------
    gradient_results : dict
        Results from gradient independence analysis.
    bounds_results : dict
        Results from temperature bounds analysis.
    convergence_results : dict
        Results from convergence analysis.
    save_path : str
        Path to save the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Subplot 1: Gradient Norms Bar Chart ---
    ax1 = axes[0]
    bar_labels = []
    bar_values = []
    bar_colors = []

    if gradient_results.get('status') == 'completed':
        aleatoric_grad = gradient_results.get(
            'grad_aleatoric_loss_wrt_epistemic_params', {}
        )
        epistemic_grad = gradient_results.get(
            'grad_epistemic_loss_wrt_aleatoric_params', {}
        )

        mean_a = aleatoric_grad.get('mean', 0)
        mean_e = epistemic_grad.get('mean', 0)

        if not np.isnan(mean_a):
            bar_labels.append(r'$\nabla_{\theta_e} \mathcal{L}_a$')
            bar_values.append(mean_a)
            bar_colors.append('steelblue')

        if not np.isnan(mean_e):
            bar_labels.append(r'$\nabla_{\theta_a} \mathcal{L}_e$')
            bar_values.append(mean_e)
            bar_colors.append('coral')

    if bar_values:
        x_pos = np.arange(len(bar_labels))
        ax1.bar(x_pos, bar_values, color=bar_colors, alpha=0.8, width=0.5)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(bar_labels, fontsize=12)
        ax1.set_ylabel('L2 Gradient Norm', fontsize=11)
        ax1.set_title('Cross-Gradient Independence', fontsize=12)
        ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

        # Add value annotations
        for i, v in enumerate(bar_values):
            ax1.text(i, v + max(bar_values) * 0.02, f'{v:.4f}',
                     ha='center', va='bottom', fontsize=10)
    else:
        ax1.text(0.5, 0.5, 'Gradient analysis\nnot available',
                 ha='center', va='center', transform=ax1.transAxes,
                 fontsize=12, color='gray')
        ax1.set_title('Cross-Gradient Independence', fontsize=12)

    # --- Subplot 2: Temperature Bounds Histogram ---
    ax2 = axes[1]

    if bounds_results.get('status') == 'completed':
        tau_a_vals = bounds_results.get('_tau_a_values', [])
        tau_e_vals = bounds_results.get('_tau_e_values', [])

        tau_a_arr = np.array(tau_a_vals)
        tau_e_arr = np.array(tau_e_vals)

        # Filter finite values
        tau_a_finite = tau_a_arr[np.isfinite(tau_a_arr)]
        tau_e_finite = tau_e_arr[np.isfinite(tau_e_arr)]

        if len(tau_a_finite) > 0:
            ax2.hist(tau_a_finite, bins=50, alpha=0.6, color='steelblue',
                     label=r'$\tau_a$ (aleatoric)', density=True)
        if len(tau_e_finite) > 0 and np.std(tau_e_finite) > 1e-8:
            ax2.hist(tau_e_finite, bins=50, alpha=0.6, color='coral',
                     label=r'$\tau_e$ (epistemic)', density=True)
        elif len(tau_e_finite) > 0:
            # tau_e is nearly constant, show as vertical line
            ax2.axvline(x=np.mean(tau_e_finite), color='coral',
                        linestyle='--', linewidth=2,
                        label=r'$\tau_e$ = {:.3f}'.format(np.mean(tau_e_finite)))

        # Show tau_min bound
        tau_min = bounds_results.get('tau_a', {}).get('tau_min')
        if tau_min is not None:
            ax2.axvline(x=tau_min, color='red', linestyle=':',
                        linewidth=1.5, label=r'$\tau_{min}$ = ' + f'{tau_min:.2f}')

        ax2.set_xlabel('Temperature', fontsize=11)
        ax2.set_ylabel('Density', fontsize=11)
        ax2.set_title('Temperature Distributions & Bounds', fontsize=12)
        ax2.legend(fontsize=9)
    else:
        ax2.text(0.5, 0.5, 'Temperature analysis\nnot available',
                 ha='center', va='center', transform=ax2.transAxes,
                 fontsize=12, color='gray')
        ax2.set_title('Temperature Distributions & Bounds', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Figure saved to: {save_path}')


# ======================================================================
# Console output
# ======================================================================

def print_gradient_results(results):
    """Print gradient independence analysis results."""
    print('\n--- A. Gradient Independence ---')

    if results.get('status') != 'completed':
        print(f"  Skipped: {results.get('reason', 'unknown')}")
        return

    print(f"  Aleatoric params (temp_predictor): {results['num_aleatoric_params']}")
    print(f"  Epistemic params (learned_temp):   {results['num_epistemic_params']}")
    print(f"  Batches analyzed: {results['num_batches']}")

    grad_a = results.get('grad_aleatoric_loss_wrt_epistemic_params', {})
    if grad_a:
        print(f"  ||grad(L_aleatoric) w.r.t. epistemic params||:")
        print(f"    Mean L2 norm: {grad_a.get('mean', 'N/A'):.6f}")
        print(f"    Max  L2 norm: {grad_a.get('max', 'N/A'):.6f}")

    grad_e = results.get('grad_epistemic_loss_wrt_aleatoric_params', {})
    if grad_e:
        print(f"  ||grad(L_epistemic) w.r.t. aleatoric params||:")
        print(f"    Mean L2 norm: {grad_e.get('mean', 'N/A'):.6f}")
        print(f"    Max  L2 norm: {grad_e.get('max', 'N/A'):.6f}")


def print_bounds_results(results):
    """Print temperature bounds analysis results."""
    print('\n--- B. Temperature Bounds ---')

    if results.get('status') != 'completed':
        print(f"  Skipped: {results.get('reason', 'unknown')}")
        return

    print(f"  Total samples: {results['total_samples']}")

    for name, key in [('tau_a (aleatoric)', 'tau_a'), ('tau_e (epistemic)', 'tau_e')]:
        stats = results.get(key, {})
        print(f"\n  {name}:")
        print(f"    Min:     {stats.get('min', 'N/A'):.6f}" if isinstance(stats.get('min'), (int, float)) and not np.isnan(stats.get('min', float('nan'))) else f"    Min:     N/A")
        print(f"    Max:     {stats.get('max', 'N/A'):.6f}" if isinstance(stats.get('max'), (int, float)) and not np.isnan(stats.get('max', float('nan'))) else f"    Max:     N/A")
        print(f"    Mean:    {stats.get('mean', 'N/A'):.6f}" if isinstance(stats.get('mean'), (int, float)) and not np.isnan(stats.get('mean', float('nan'))) else f"    Mean:    N/A")
        print(f"    Std:     {stats.get('std', 'N/A'):.6f}" if isinstance(stats.get('std'), (int, float)) and not np.isnan(stats.get('std', float('nan'))) else f"    Std:     N/A")
        print(f"    %NaN:    {stats.get('pct_nan', 0):.2f}%")
        print(f"    %Inf:    {stats.get('pct_inf', 0):.2f}%")

        if stats.get('tau_min') is not None:
            print(f"    tau_min: {stats['tau_min']:.4f}")
            if stats.get('bound_satisfied') is not None:
                status = 'PASS' if stats['bound_satisfied'] else 'FAIL'
                violations = stats.get('bound_violations', 0)
                print(f"    Bound check (tau >= tau_min): {status} ({violations} violations)")


def print_convergence_results(results):
    """Print convergence analysis results."""
    print('\n--- C. Convergence ---')

    if results.get('status') != 'completed':
        print(f"  Skipped: {results.get('reason', 'unknown')}")
        return

    print(f"  Results file: {results.get('results_file', 'N/A')}")

    if results.get('temperature_dynamics') is not None:
        print('  Temperature dynamics: available')
        if 'tau_e_over_epochs' in results:
            tau_e = results['tau_e_over_epochs']
            print(f"    tau_e epochs recorded: {len(tau_e)}")
            if tau_e:
                print(f"    tau_e initial: {tau_e[0]:.4f}" if isinstance(tau_e[0], (int, float)) else "")
                print(f"    tau_e final:   {tau_e[-1]:.4f}" if isinstance(tau_e[-1], (int, float)) else "")
    else:
        print('  Temperature dynamics: not available')

    loss_info = results.get('loss_history')
    if loss_info is not None and isinstance(loss_info, dict) and 'num_epochs' in loss_info:
        print(f"  Loss history: {loss_info['num_epochs']} epochs")
        if 'initial_loss' in loss_info:
            print(f"    Initial loss: {loss_info['initial_loss']:.4f}")
            print(f"    Final loss:   {loss_info['final_loss']:.4f}")
            print(f"    Min loss:     {loss_info['min_loss']:.4f}")
            print(f"    Slope:        {loss_info['slope']:.6f}")
            trend = 'decreasing' if loss_info.get('is_decreasing_trend') else 'not decreasing'
            print(f"    Trend:        {trend}")
            print(f"    Decreasing steps: {loss_info.get('pct_decreasing_steps', 0):.1f}%")
    else:
        print('  Loss history: not available')


# ======================================================================
# Main
# ======================================================================

def main():
    """Run the theoretical analysis pipeline."""
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(42)
    device = get_device()

    print('=' * 70)
    print('Theoretical Analysis of UCAT Decomposition')
    print('=' * 70)
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Dataset:    {args.dataset}')
    print(f'Device:     {device}')
    print()

    # Collect all results
    all_results = {
        'checkpoint': args.checkpoint,
        'dataset': args.dataset,
    }

    # --- Load model (if checkpoint provided) ---
    model = None
    test_loader = None

    if args.checkpoint and os.path.isfile(args.checkpoint):
        print('Loading model...')
        model, config = load_model(args.checkpoint, device)
        all_results['model_config'] = config

        print('Loading data...')
        dataset_info = get_dataset_info(args.dataset)
        _, _, test_loader, _ = get_dataloaders(
            dataset_name=args.dataset,
            root=args.data_root,
            batch_size=args.batch_size,
            num_workers=4,
        )
        print(f'  Test set loaded ({args.dataset})')
    elif args.checkpoint:
        print(f'Warning: Checkpoint file not found: {args.checkpoint}')
        print('  Skipping analyses A and B (require model + data).')
    else:
        print('No checkpoint provided. Skipping analyses A and B.')

    # --- A. Gradient Independence ---
    if model is not None and test_loader is not None:
        print('\nRunning gradient independence analysis...')
        gradient_results = analyze_gradient_independence(model, test_loader, device)
    else:
        gradient_results = {
            'status': 'skipped',
            'reason': 'Model or test data not available',
        }
    all_results['gradient_independence'] = gradient_results
    print_gradient_results(gradient_results)

    # --- B. Temperature Bounds ---
    if model is not None and test_loader is not None:
        print('\nRunning temperature bounds analysis...')
        bounds_results = analyze_temperature_bounds(model, test_loader, device)
    else:
        bounds_results = {
            'status': 'skipped',
            'reason': 'Model or test data not available',
        }
    all_results['temperature_bounds'] = bounds_results
    print_bounds_results(bounds_results)

    # --- C. Convergence ---
    results_file = args.results_file
    if results_file is None and args.checkpoint:
        results_file = auto_detect_results_file(args.checkpoint)
        if results_file:
            print(f'\nAuto-detected results file: {results_file}')
        else:
            print('\nNo results file detected for convergence analysis.')

    convergence_results = analyze_convergence(results_file)
    all_results['convergence'] = convergence_results
    print_convergence_results(convergence_results)

    # --- Generate figure ---
    print('\nGenerating figure...')
    fig_path = os.path.join(args.save_dir, 'theoretical_analysis.png')
    generate_figure(gradient_results, bounds_results, convergence_results, fig_path)

    # --- Save JSON results ---
    # Remove internal array data before saving (too large for JSON readability)
    save_results = {}
    for k, v in all_results.items():
        if isinstance(v, dict):
            save_results[k] = {
                kk: vv for kk, vv in v.items()
                if not kk.startswith('_')
            }
        else:
            save_results[k] = v

    json_path = os.path.join(args.save_dir, 'theoretical_results.json')
    with open(json_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f'\nResults saved to: {json_path}')

    print('\nTheoretical analysis complete.')
    print(f'Outputs in: {os.path.abspath(args.save_dir)}')


if __name__ == '__main__':
    main()
