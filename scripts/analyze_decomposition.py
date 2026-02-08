#!/usr/bin/env python3
"""Decomposition validation analysis for Enhanced UCAT.

Produces:
- tau_a vs tau_e scatter with significance annotations (Fig A)
- Blur response curves (Fig B)
- Temperature distribution histograms (Fig E)
- Decomposition validation metrics JSON

Statistical tests for tau_a / tau_e independence:
- Pearson r with p-value and bootstrap 95% CI
- Spearman rank correlation with p-value
- Distance correlation (nonlinear independence test)
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
from scipy.stats import pearsonr, spearmanr

from src.models.efficient_vit import EfficientEuroSATViT
from src.models.baseline import BaselineViT
from src.data.datasets import get_dataloaders, get_dataset_info
from src.data.blur import apply_all_blur_levels
from src.training.losses import UCATLoss
from src.utils.helpers import set_seed, get_device


# ======================================================================
# Distance correlation (Szekely et al., 2007)
# ======================================================================

def _distance_matrix(x):
    """Compute pairwise Euclidean distance matrix for 1-D array."""
    x = x.reshape(-1, 1).astype(np.float64)
    return np.abs(x - x.T)


def _centered_distance_matrix(D):
    """Double-center a distance matrix."""
    row_mean = D.mean(axis=1, keepdims=True)
    col_mean = D.mean(axis=0, keepdims=True)
    grand_mean = D.mean()
    return D - row_mean - col_mean + grand_mean


def distance_correlation(x, y):
    """Compute distance correlation between two 1-D arrays.

    Distance correlation (dCor) detects both linear AND nonlinear
    dependencies.  dCor = 0 iff x and y are statistically independent
    (for finite second moments).

    Reference: Szekely, Rizzo & Bakirov (2007), Annals of Statistics.

    Parameters
    ----------
    x, y : np.ndarray
        1-D arrays of equal length.

    Returns
    -------
    float
        Distance correlation in [0, 1].
    """
    n = len(x)
    if n < 4:
        return 0.0

    A = _centered_distance_matrix(_distance_matrix(x))
    B = _centered_distance_matrix(_distance_matrix(y))

    dcov_xy = np.sqrt(max((A * B).sum() / (n * n), 0.0))
    dcov_xx = np.sqrt(max((A * A).sum() / (n * n), 0.0))
    dcov_yy = np.sqrt(max((B * B).sum() / (n * n), 0.0))

    if dcov_xx * dcov_yy < 1e-15:
        return 0.0

    return float(dcov_xy / np.sqrt(dcov_xx * dcov_yy))


def distance_correlation_permutation_test(x, y, n_permutations=1000, rng=None):
    """Permutation test for distance correlation significance.

    Parameters
    ----------
    x, y : np.ndarray
        1-D arrays of equal length.
    n_permutations : int
        Number of permutations.
    rng : np.random.Generator or None
        Random number generator.

    Returns
    -------
    tuple of (float, float)
        (distance_correlation, p_value).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    observed = distance_correlation(x, y)

    count_ge = 0
    for _ in range(n_permutations):
        y_perm = rng.permutation(y)
        perm_dcor = distance_correlation(x, y_perm)
        if perm_dcor >= observed:
            count_ge += 1

    p_value = (count_ge + 1) / (n_permutations + 1)
    return observed, p_value


# ======================================================================
# Bootstrap confidence interval for correlations
# ======================================================================

def bootstrap_correlation_ci(x, y, corr_func, n_bootstrap=10000, ci=0.95, rng=None):
    """Compute bootstrap CI for a correlation statistic.

    Parameters
    ----------
    x, y : np.ndarray
        1-D arrays of equal length.
    corr_func : callable
        Function that takes (x, y) and returns a scalar correlation.
    n_bootstrap : int
        Number of bootstrap resamples.
    ci : float
        Confidence level.
    rng : np.random.Generator or None
        Random number generator.

    Returns
    -------
    tuple of (float, float)
        (lower_bound, upper_bound) of the CI.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(x)
    if n < 4:
        val = corr_func(x, y)
        return val, val

    boot_corrs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_corrs[i] = corr_func(x[idx], y[idx])

    alpha = 1.0 - ci
    lower = float(np.percentile(boot_corrs, 100 * alpha / 2))
    upper = float(np.percentile(boot_corrs, 100 * (1 - alpha / 2)))
    return lower, upper


def _pearson_scalar(x, y):
    """Pearson r as a plain scalar (for bootstrap)."""
    if x.std() < 1e-15 or y.std() < 1e-15:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _spearman_scalar(x, y):
    """Spearman rho as a plain scalar (for bootstrap)."""
    rho, _ = spearmanr(x, y)
    return float(rho) if not np.isnan(rho) else 0.0


# ======================================================================
# Model loading and data collection
# ======================================================================

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


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='eurosat',
                        choices=['eurosat', 'cifar100', 'resisc45'],
                        help='Dataset to evaluate on')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./analysis_results/decomposition')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_bootstrap', type=int, default=10000,
                        help='Number of bootstrap resamples for CIs')
    parser.add_argument('--n_dcor_perms', type=int, default=1000,
                        help='Number of permutations for distance correlation test')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(42)
    device = get_device()
    rng = np.random.default_rng(42)

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

    tau_a = data['tau_a']
    tau_e = data['tau_e']

    # ==================================================================
    # Independence tests
    # ==================================================================
    print("\nRunning independence tests on tau_a vs tau_e...")

    independence_tests = {}

    # For large N, subsample for distance correlation (O(N^2) cost)
    max_dcor_samples = 5000
    if len(tau_a) > max_dcor_samples:
        idx = rng.choice(len(tau_a), size=max_dcor_samples, replace=False)
        tau_a_sub = tau_a[idx]
        tau_e_sub = tau_e[idx]
    else:
        tau_a_sub = tau_a
        tau_e_sub = tau_e

    # 1. Pearson r with p-value
    if tau_a.std() > 1e-15 and tau_e.std() > 1e-15:
        pearson_r, pearson_p = pearsonr(tau_a, tau_e)
        pearson_ci_lo, pearson_ci_hi = bootstrap_correlation_ci(
            tau_a, tau_e, _pearson_scalar,
            n_bootstrap=args.n_bootstrap, rng=rng,
        )
    else:
        pearson_r, pearson_p = 0.0, 1.0
        pearson_ci_lo, pearson_ci_hi = 0.0, 0.0

    independence_tests['pearson'] = {
        'r': float(pearson_r),
        'p_value': float(pearson_p),
        'ci_95_lower': float(pearson_ci_lo),
        'ci_95_upper': float(pearson_ci_hi),
        'n_samples': len(tau_a),
    }
    print(f"  Pearson r  = {pearson_r:.4f}  (p={pearson_p:.2e}, "
          f"95% CI [{pearson_ci_lo:.4f}, {pearson_ci_hi:.4f}])")

    # 2. Spearman rank correlation with p-value
    if tau_a.std() > 1e-15 and tau_e.std() > 1e-15:
        spearman_rho, spearman_p = spearmanr(tau_a, tau_e)
        spearman_ci_lo, spearman_ci_hi = bootstrap_correlation_ci(
            tau_a, tau_e, _spearman_scalar,
            n_bootstrap=args.n_bootstrap, rng=rng,
        )
    else:
        spearman_rho, spearman_p = 0.0, 1.0
        spearman_ci_lo, spearman_ci_hi = 0.0, 0.0

    independence_tests['spearman'] = {
        'rho': float(spearman_rho),
        'p_value': float(spearman_p),
        'ci_95_lower': float(spearman_ci_lo),
        'ci_95_upper': float(spearman_ci_hi),
        'n_samples': len(tau_a),
    }
    print(f"  Spearman rho = {spearman_rho:.4f}  (p={spearman_p:.2e}, "
          f"95% CI [{spearman_ci_lo:.4f}, {spearman_ci_hi:.4f}])")

    # 3. Distance correlation with permutation test
    print(f"  Computing distance correlation ({len(tau_a_sub)} samples, "
          f"{args.n_dcor_perms} permutations)...")
    dcor_val, dcor_p = distance_correlation_permutation_test(
        tau_a_sub, tau_e_sub,
        n_permutations=args.n_dcor_perms, rng=rng,
    )
    independence_tests['distance_correlation'] = {
        'dcor': float(dcor_val),
        'p_value': float(dcor_p),
        'n_samples': len(tau_a_sub),
        'n_permutations': args.n_dcor_perms,
    }
    print(f"  dCor       = {dcor_val:.4f}  (p={dcor_p:.4f}, "
          f"n={len(tau_a_sub)}, perms={args.n_dcor_perms})")

    # Summary verdict with Bonferroni correction for 3 independence tests
    # Family-wise alpha = 0.05, per-test threshold = 0.05 / 3 = 0.0167
    n_independence_tests = 3
    alpha_family = 0.05
    alpha_per_test = alpha_family / n_independence_tests  # 0.0167

    # Bonferroni-corrected p-values
    pearson_p_corrected = min(pearson_p * n_independence_tests, 1.0)
    spearman_p_corrected = min(spearman_p * n_independence_tests, 1.0)
    dcor_p_corrected = min(dcor_p * n_independence_tests, 1.0)

    # Pass criteria: effect size small AND corrected p non-significant
    passes_pearson = abs(pearson_r) < 0.3 and pearson_p_corrected > alpha_family
    passes_spearman = abs(spearman_rho) < 0.3 and spearman_p_corrected > alpha_family
    passes_dcor = dcor_val < 0.3 and dcor_p_corrected > alpha_family
    overall_pass = passes_pearson and passes_spearman and passes_dcor

    independence_tests['bonferroni'] = {
        'n_tests': n_independence_tests,
        'alpha_family': alpha_family,
        'alpha_per_test': alpha_per_test,
        'pearson_p_corrected': float(pearson_p_corrected),
        'spearman_p_corrected': float(spearman_p_corrected),
        'dcor_p_corrected': float(dcor_p_corrected),
    }
    independence_tests['verdict'] = {
        'pearson_pass': passes_pearson,
        'spearman_pass': passes_spearman,
        'dcor_pass': passes_dcor,
        'overall_independence': overall_pass,
        'criteria': (
            f'|r| < 0.3 AND |rho| < 0.3 AND dCor < 0.3; '
            f'p-values Bonferroni-corrected (k={n_independence_tests}, '
            f'alpha_family={alpha_family})'
        ),
    }
    verdict_str = "PASS" if overall_pass else "FAIL"
    print(f"\n  Independence verdict: {verdict_str}")
    print(f"    (Bonferroni-corrected for k={n_independence_tests} tests, "
          f"alpha_family={alpha_family})")
    print(f"    Pearson  |r|<0.3 & p_corr>{alpha_family}:  "
          f"{'PASS' if passes_pearson else 'FAIL'} "
          f"(|r|={abs(pearson_r):.4f}, p_corr={pearson_p_corrected:.4f})")
    print(f"    Spearman |rho|<0.3 & p_corr>{alpha_family}: "
          f"{'PASS' if passes_spearman else 'FAIL'} "
          f"(|rho|={abs(spearman_rho):.4f}, p_corr={spearman_p_corrected:.4f})")
    print(f"    dCor     <0.3 & p_corr>{alpha_family}:     "
          f"{'PASS' if passes_dcor else 'FAIL'} "
          f"(dCor={dcor_val:.4f}, p_corr={dcor_p_corrected:.4f})")

    # ==================================================================
    # Blur response
    # ==================================================================
    print("\nAnalysing blur response...")
    blur_data = blur_response_analysis(model, test_loader, device)

    # ==================================================================
    # Save results
    # ==================================================================
    results = {
        'independence_tests': independence_tests,
        'mean_tau_a': float(tau_a.mean()),
        'mean_tau_e': float(tau_e.mean()),
        'std_tau_a': float(tau_a.std()),
        'std_tau_e': float(tau_e.std()),
        'blur_response': blur_data,
        'accuracy': float((data['preds'] == data['labels']).mean()),
    }

    with open(os.path.join(args.save_dir, 'decomposition_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # ==================================================================
    # Figures
    # ==================================================================
    sns.set_style('whitegrid')

    # Fig A: tau_a vs tau_e scatter with significance annotations
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(tau_a, tau_e, c=data['labels'],
                         cmap='tab10', alpha=0.5, s=10)
    ax.set_xlabel(r'$\tau_a$ (Aleatoric)', fontsize=11)
    ax.set_ylabel(r'$\tau_e$ (Epistemic)', fontsize=11)

    # Annotation box with all three test results (Bonferroni-corrected)
    textstr = (
        f"Pearson r = {pearson_r:.3f} ($p_{{corr}}$={pearson_p_corrected:.2e})\n"
        f"Spearman $\\rho$ = {spearman_rho:.3f} ($p_{{corr}}$={spearman_p_corrected:.2e})\n"
        f"dCor = {dcor_val:.3f} ($p_{{corr}}$={dcor_p_corrected:.3f})\n"
        f"Independence: {verdict_str} (Bonferroni k={n_independence_tests})"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.03, 0.97, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=props)

    ax.set_title('Temperature Decomposition Independence')
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
    axes[0].hist(tau_a, bins=50, alpha=0.7, color='steelblue')
    axes[0].set_xlabel(r'$\tau_a$')
    axes[0].set_title('Aleatoric Temperature Distribution')
    axes[1].hist(tau_e, bins=50, alpha=0.7, color='coral')
    axes[1].set_xlabel(r'$\tau_e$')
    axes[1].set_title('Epistemic Temperature Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'fig_e_temp_distributions.png'), dpi=300)
    plt.close()

    print(f"\nResults saved to {args.save_dir}")
    print(f"  Independence: {verdict_str}")
    print(f"  tau_a-blur correlation:  {blur_data.get('tau_a_blur_corr', 'N/A')}")


if __name__ == '__main__':
    main()
