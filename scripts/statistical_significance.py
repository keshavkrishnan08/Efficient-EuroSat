#!/usr/bin/env python3
"""Statistical significance tests for TNNLS paper results.

Computes paired t-tests (with proper seed-based pairing), Bonferroni
correction for multiple comparisons, Cohen's d effect sizes, and
bootstrap 95% confidence intervals.  Generates LaTeX tables.

Usage:
    python statistical_significance.py --results_dir ./results
    python statistical_significance.py --save_dir ./analysis_results/significance
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import re
from collections import defaultdict

import numpy as np
from scipy.stats import ttest_rel


# ======================================================================
# Argument parsing
# ======================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Compute statistical significance tests for TNNLS paper results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--results_dir', type=str, default='./results',
        help='Directory containing JSON result files',
    )
    parser.add_argument(
        '--save_dir', type=str, default='./analysis_results/significance',
        help='Directory to save significance analysis outputs',
    )
    parser.add_argument(
        '--n_bootstrap', type=int, default=10000,
        help='Number of bootstrap resamples for confidence intervals',
    )
    return parser.parse_args()


# ======================================================================
# Result loading and grouping
# ======================================================================

SEED_PATTERN = re.compile(r'[_\-](?:s|seed)(\d+)$')


def strip_seed_suffix(name):
    """Strip the seed suffix from a method name.

    Examples:
        'all_combined_s42'   -> 'all_combined'
        'baseline_seed123'   -> 'baseline'
        'ucat_only_s456'     -> 'ucat_only'
        'my_method'          -> 'my_method'
    """
    cleaned = SEED_PATTERN.sub('', name)
    return cleaned


def extract_seed(result):
    """Extract the random seed from a result dict.

    Checks ``result['args']['seed']`` first (written by train.py),
    then falls back to parsing the experiment/method name.

    Returns
    -------
    int or None
        The seed value, or None if not determinable.
    """
    # Primary: stored in args dict
    args = result.get('args', {})
    if 'seed' in args:
        return int(args['seed'])

    # Fallback: parse from experiment name
    raw_name = extract_method_name(result)
    m = SEED_PATTERN.search(raw_name)
    if m:
        return int(m.group(1))

    # Default seed used by the project
    return 42


def load_all_results(results_dir):
    """Load all JSON result files from the given directory (recursively).

    Parameters
    ----------
    results_dir : str
        Path to directory containing JSON result files.

    Returns
    -------
    list of dict
        List of parsed result dictionaries, each augmented with
        a 'source_file' key.
    """
    all_results = []

    if not os.path.isdir(results_dir):
        print(f"Warning: results directory '{results_dir}' does not exist.")
        return all_results

    for root, _dirs, files in os.walk(results_dir):
        for fname in sorted(files):
            if not fname.endswith('.json'):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, 'r') as f:
                    data = json.load(f)
                data['source_file'] = fpath
                all_results.append(data)
            except (json.JSONDecodeError, IOError) as exc:
                print(f"Warning: could not load {fpath}: {exc}")

    return all_results


def extract_method_name(result):
    """Extract the method name from a result dict.

    Looks for common keys used across the project's result formats.
    Falls back to the source filename (sans extension and seed suffix).
    """
    for key in ('method', 'config_name', 'experiment', 'name'):
        if key in result and result[key]:
            return str(result[key])

    # Fallback: derive from filename
    source = result.get('source_file', '')
    if source:
        basename = os.path.splitext(os.path.basename(source))[0]
        return basename

    return 'unknown'


def extract_test_accuracy(result):
    """Extract test accuracy from a result dict.

    Returns
    -------
    float or None
        Test accuracy value, or None if not found.
    """
    if 'test' in result and isinstance(result['test'], dict):
        if 'test_acc' in result['test']:
            return float(result['test']['test_acc'])

    for key in ('test_accuracy', 'test_acc', 'accuracy'):
        if key in result and result[key] is not None:
            return float(result[key])

    return None


def extract_dataset(result):
    """Extract the dataset name from a result dict."""
    return result.get('dataset', 'eurosat')


def group_results_by_seed(results):
    """Group results by (dataset, method) with explicit seed tracking.

    Parameters
    ----------
    results : list of dict
        Loaded result dictionaries.

    Returns
    -------
    dict
        Mapping of ``(dataset, method) -> {seed: test_acc}``.
    """
    groups = defaultdict(dict)

    for result in results:
        raw_name = extract_method_name(result)
        method = strip_seed_suffix(raw_name)
        dataset = extract_dataset(result)
        acc = extract_test_accuracy(result)
        seed = extract_seed(result)

        if acc is None or seed is None:
            continue

        key = (dataset, method)
        groups[key][seed] = acc

    return dict(groups)


# ======================================================================
# Statistical computations
# ======================================================================

def compute_method_stats(seed_acc_map):
    """Compute summary statistics from a seed -> accuracy mapping.

    Parameters
    ----------
    seed_acc_map : dict
        Mapping of seed -> test accuracy.

    Returns
    -------
    dict
        Dictionary with keys: mean, std, min, max, count, seeds.
    """
    seeds = sorted(seed_acc_map.keys())
    accs = np.array([seed_acc_map[s] for s in seeds], dtype=np.float64)
    return {
        'mean': float(np.mean(accs)),
        'std': float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        'min': float(np.min(accs)),
        'max': float(np.max(accs)),
        'count': len(accs),
        'seeds': seeds,
    }


def pair_by_seed(seed_map_a, seed_map_b):
    """Pair accuracy values by matching seeds.

    Parameters
    ----------
    seed_map_a : dict
        {seed: accuracy} for method A.
    seed_map_b : dict
        {seed: accuracy} for method B.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray, list)
        Paired accuracy arrays and the list of common seeds.
        Both arrays are in the same seed order.
    """
    common_seeds = sorted(set(seed_map_a.keys()) & set(seed_map_b.keys()))
    accs_a = np.array([seed_map_a[s] for s in common_seeds], dtype=np.float64)
    accs_b = np.array([seed_map_b[s] for s in common_seeds], dtype=np.float64)
    return accs_a, accs_b, common_seeds


def paired_ttest(accs_a, accs_b):
    """Run a paired t-test between two arrays of paired accuracy values.

    Parameters
    ----------
    accs_a : np.ndarray
        Accuracy values for method A (paired by seed).
    accs_b : np.ndarray
        Accuracy values for method B (paired by seed).

    Returns
    -------
    tuple of (float, float)
        (t-statistic, p-value).  Returns (NaN, NaN) if the test
        cannot be computed.
    """
    if len(accs_a) < 2 or len(accs_b) < 2 or len(accs_a) != len(accs_b):
        return float('nan'), float('nan')

    if np.allclose(accs_a, accs_b):
        return 0.0, 1.0

    try:
        t_stat, p_value = ttest_rel(accs_a, accs_b)
        return float(t_stat), float(p_value)
    except Exception:
        return float('nan'), float('nan')


def cohens_d(accs_a, accs_b):
    """Compute Cohen's d effect size for paired samples.

    Uses the standard deviation of differences as the denominator
    (Cohen's d_z for paired designs).

    Parameters
    ----------
    accs_a, accs_b : np.ndarray
        Paired accuracy arrays.

    Returns
    -------
    float
        Cohen's d_z effect size.  Returns NaN if computation fails.
    """
    diffs = accs_a - accs_b
    sd = np.std(diffs, ddof=1)
    if sd < 1e-15:
        return 0.0
    return float(np.mean(diffs) / sd)


def effect_size_label(d):
    """Interpret Cohen's d magnitude (Cohen, 1988)."""
    d_abs = abs(d)
    if np.isnan(d_abs):
        return ''
    if d_abs < 0.2:
        return 'negligible'
    if d_abs < 0.5:
        return 'small'
    if d_abs < 0.8:
        return 'medium'
    return 'large'


def bootstrap_ci(accs, n_bootstrap=10000, ci=0.95, rng=None):
    """Compute bootstrap confidence interval for the mean.

    Parameters
    ----------
    accs : np.ndarray
        Accuracy values.
    n_bootstrap : int
        Number of bootstrap resamples.
    ci : float
        Confidence level (default 0.95).
    rng : np.random.Generator or None
        Random number generator.

    Returns
    -------
    tuple of (float, float)
        (lower_bound, upper_bound) of the CI.
    """
    if len(accs) < 2:
        m = float(np.mean(accs))
        return m, m

    if rng is None:
        rng = np.random.default_rng(42)

    boot_means = np.empty(n_bootstrap)
    n = len(accs)
    for i in range(n_bootstrap):
        sample = rng.choice(accs, size=n, replace=True)
        boot_means[i] = sample.mean()

    alpha = 1.0 - ci
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lower, upper


def bonferroni_correct(p_value, n_comparisons):
    """Apply Bonferroni correction to a p-value.

    Parameters
    ----------
    p_value : float
        Raw p-value.
    n_comparisons : int
        Number of comparisons in the family.

    Returns
    -------
    float
        Corrected p-value, clamped to [0, 1].
    """
    if np.isnan(p_value):
        return float('nan')
    return float(min(p_value * n_comparisons, 1.0))


def significance_marker(p_value):
    """Return a significance marker string for a given p-value.

    Returns '***' if p < 0.001, '**' if p < 0.01, '*' if p < 0.05,
    '' otherwise (or if p is NaN).
    """
    if np.isnan(p_value):
        return ''
    if p_value < 0.001:
        return '***'
    if p_value < 0.01:
        return '**'
    if p_value < 0.05:
        return '*'
    return ''


# ======================================================================
# Key comparisons
# ======================================================================

SINGLE_ABLATIONS = [
    'ucat_only',
    'early_exit_only',
    'dropout_only',
    'residual_only',
    'annealing_only',
]

KEY_COMPARISONS = [
    ('baseline', 'all_combined', 'Baseline vs All Combined'),
]

for ablation in SINGLE_ABLATIONS:
    KEY_COMPARISONS.append(
        ('baseline', ablation, f'Baseline vs {ablation}')
    )

KEY_COMPARISONS.append(
    ('all_combined', 'decomp_with_losses', 'All Combined vs Decomposed')
)


def run_comparisons(seed_groups, dataset='eurosat', n_bootstrap=10000):
    """Run all key comparisons for a given dataset with proper seed pairing.

    Parameters
    ----------
    seed_groups : dict
        Mapping of ``(dataset, method) -> {seed: accuracy}``.
    dataset : str
        Dataset name to filter on.
    n_bootstrap : int
        Number of bootstrap resamples for CIs.

    Returns
    -------
    list of dict
        Each dict has keys: method_a, method_b, description,
        n_pairs, seeds_used, t_stat, p_value_raw, p_value_corrected,
        marker_raw, marker_corrected, cohens_d, effect_label.
    """
    n_total_comparisons = len(KEY_COMPARISONS)
    comparison_results = []
    rng = np.random.default_rng(42)

    for method_a, method_b, desc in KEY_COMPARISONS:
        key_a = (dataset, method_a)
        key_b = (dataset, method_b)

        seed_map_a = seed_groups.get(key_a, {})
        seed_map_b = seed_groups.get(key_b, {})

        if not seed_map_a or not seed_map_b:
            continue

        # Pair by matching seeds
        accs_a, accs_b, common_seeds = pair_by_seed(seed_map_a, seed_map_b)

        if len(common_seeds) < 2:
            continue

        t_stat, p_raw = paired_ttest(accs_a, accs_b)
        p_corrected = bonferroni_correct(p_raw, n_total_comparisons)
        d = cohens_d(accs_a, accs_b)

        # Bootstrap CI on the mean difference
        diffs = accs_a - accs_b
        ci_lower, ci_upper = bootstrap_ci(diffs, n_bootstrap=n_bootstrap, rng=rng)

        comparison_results.append({
            'method_a': method_a,
            'method_b': method_b,
            'description': desc,
            'n_pairs': len(common_seeds),
            'seeds_used': common_seeds,
            'mean_a': float(np.mean(accs_a)),
            'mean_b': float(np.mean(accs_b)),
            'mean_diff': float(np.mean(diffs)),
            'diff_ci_lower': ci_lower,
            'diff_ci_upper': ci_upper,
            't_statistic': t_stat,
            'p_value_raw': p_raw,
            'p_value_corrected': p_corrected,
            'marker_raw': significance_marker(p_raw),
            'marker_corrected': significance_marker(p_corrected),
            'cohens_d': d,
            'effect_label': effect_size_label(d),
            'n_comparisons_bonferroni': n_total_comparisons,
        })

    return comparison_results


# ======================================================================
# LaTeX table generation
# ======================================================================

def generate_latex_table(method_stats, comparisons_vs_baseline, dataset='eurosat'):
    """Generate a LaTeX table with method accuracies, CIs, effect sizes, and p-values.

    Parameters
    ----------
    method_stats : dict
        Mapping of method_name -> stats dict (mean, std, ci_lower, ci_upper, ...).
    comparisons_vs_baseline : dict
        Mapping of method_name -> comparison result dict.
    dataset : str
        Dataset name for the table caption.

    Returns
    -------
    str
        LaTeX table source.
    """
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'  \centering')
    lines.append(
        r'  \caption{Statistical significance of ablation results on '
        + dataset.replace('_', r'\_')
        + r'. $p$-values are Bonferroni-corrected for '
        + str(len(KEY_COMPARISONS))
        + r' comparisons.}'
    )
    lines.append(r'  \label{tab:significance_' + dataset + r'}')
    lines.append(r'  \begin{tabular}{lccccc}')
    lines.append(r'    \toprule')
    lines.append(
        r'    Method & Accuracy (\%) & 95\% CI & '
        r"Cohen's $d$ & $p_{\mathrm{corr}}$ \\"
    )
    lines.append(r'    \midrule')

    sorted_methods = sorted(method_stats.keys())
    if 'baseline' in sorted_methods:
        sorted_methods.remove('baseline')
        sorted_methods.insert(0, 'baseline')

    for method in sorted_methods:
        stats = method_stats[method]
        if stats['count'] < 1:
            continue

        mean_pct = stats['mean'] * 100 if stats['mean'] <= 1.0 else stats['mean']
        std_pct = stats['std'] * 100 if stats['mean'] <= 1.0 else stats['std']
        acc_str = f'{mean_pct:.2f} $\\pm$ {std_pct:.2f}'

        # CI
        ci_l = stats.get('ci_lower', stats['mean'])
        ci_u = stats.get('ci_upper', stats['mean'])
        ci_l_pct = ci_l * 100 if ci_l <= 1.0 else ci_l
        ci_u_pct = ci_u * 100 if ci_u <= 1.0 else ci_u
        ci_str = f'[{ci_l_pct:.2f}, {ci_u_pct:.2f}]'

        if method == 'baseline':
            d_str = '---'
            p_str = '---'
        elif method in comparisons_vs_baseline:
            comp = comparisons_vs_baseline[method]
            p_corr = comp['p_value_corrected']
            marker = comp['marker_corrected']
            d_val = comp['cohens_d']

            if np.isnan(p_corr):
                p_str = 'N/A'
            else:
                p_str = f'{p_corr:.4f}{marker}'

            if np.isnan(d_val):
                d_str = 'N/A'
            else:
                d_str = f'{d_val:.3f}'
        else:
            d_str = 'N/A'
            p_str = 'N/A'

        method_tex = method.replace('_', r'\_')
        lines.append(f'    {method_tex} & {acc_str} & {ci_str} & {d_str} & {p_str} \\\\')

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'  \vspace{0.5em}')
    lines.append(
        r'  \raggedright\footnotesize '
        r'Significance: {*}\,$p<0.05$, {**}\,$p<0.01$, {***}\,$p<0.001$ '
        r'(Bonferroni-corrected). '
        r"Cohen's $d$: $|d|<0.2$ negligible, $<0.5$ small, $<0.8$ medium, "
        r'$\geq0.8$ large.'
    )
    lines.append(r'\end{table}')

    return '\n'.join(lines)


# ======================================================================
# Console output
# ======================================================================

def print_results_table(method_stats, comparisons_vs_baseline):
    """Print a formatted results table to stdout."""
    print()
    print('=' * 100)
    print('STATISTICAL SIGNIFICANCE RESULTS')
    print('=' * 100)

    header = (
        f"{'Method':<25} {'Acc (mean +/- std)':>22} "
        f"{'95% CI':>18} {'Cohen d':>10} {'p(raw)':>10} {'p(corr)':>10} {'Sig':>5}"
    )
    print(header)
    print('-' * 100)

    sorted_methods = sorted(method_stats.keys())
    if 'baseline' in sorted_methods:
        sorted_methods.remove('baseline')
        sorted_methods.insert(0, 'baseline')

    for method in sorted_methods:
        stats = method_stats[method]
        if stats['count'] < 1:
            continue

        mean_pct = stats['mean'] * 100 if stats['mean'] <= 1.0 else stats['mean']
        std_pct = stats['std'] * 100 if stats['mean'] <= 1.0 else stats['std']
        acc_str = f'{mean_pct:.2f}+/-{std_pct:.2f} (n={stats["count"]})'

        ci_l = stats.get('ci_lower', stats['mean'])
        ci_u = stats.get('ci_upper', stats['mean'])
        ci_l_pct = ci_l * 100 if ci_l <= 1.0 else ci_l
        ci_u_pct = ci_u * 100 if ci_u <= 1.0 else ci_u
        ci_str = f'[{ci_l_pct:.2f},{ci_u_pct:.2f}]'

        if method == 'baseline':
            d_str = '---'
            p_raw_str = '---'
            p_corr_str = '---'
            sig = ''
        elif method in comparisons_vs_baseline:
            comp = comparisons_vs_baseline[method]
            p_raw = comp['p_value_raw']
            p_corr = comp['p_value_corrected']
            d_val = comp['cohens_d']
            sig = comp['marker_corrected']

            p_raw_str = f'{p_raw:.4f}' if not np.isnan(p_raw) else 'N/A'
            p_corr_str = f'{p_corr:.4f}' if not np.isnan(p_corr) else 'N/A'
            d_str = f'{d_val:.3f}' if not np.isnan(d_val) else 'N/A'
        else:
            d_str = 'N/A'
            p_raw_str = 'N/A'
            p_corr_str = 'N/A'
            sig = ''

        print(
            f'{method:<25} {acc_str:>22} {ci_str:>18} '
            f'{d_str:>10} {p_raw_str:>10} {p_corr_str:>10} {sig:>5}'
        )

    print('=' * 100)
    print('Significance: * p<0.05, ** p<0.01, *** p<0.001 (Bonferroni-corrected)')
    print("Cohen's d: |d|<0.2 negligible, <0.5 small, <0.8 medium, >=0.8 large")
    print()


def print_comparisons(comparisons):
    """Print the detailed comparison results."""
    if not comparisons:
        print('No paired comparisons could be computed.')
        return

    print()
    print('PAIRED T-TEST COMPARISONS (seed-matched)')
    print('-' * 110)
    header = (
        f"{'Comparison':<35} {'Seeds':>10} {'n':>3} "
        f"{'t-stat':>8} {'p(raw)':>8} {'p(corr)':>8} "
        f"{'d':>7} {'Effect':>12} {'Sig':>5}"
    )
    print(header)
    print('-' * 110)

    for comp in comparisons:
        desc = comp['description']
        seeds_str = str(comp['seeds_used'])
        n = comp['n_pairs']
        t = comp['t_statistic']
        p_raw = comp['p_value_raw']
        p_corr = comp['p_value_corrected']
        d = comp['cohens_d']
        effect = comp['effect_label']
        sig = comp['marker_corrected']

        t_str = f'{t:.4f}' if not np.isnan(t) else 'N/A'
        pr_str = f'{p_raw:.4f}' if not np.isnan(p_raw) else 'N/A'
        pc_str = f'{p_corr:.4f}' if not np.isnan(p_corr) else 'N/A'
        d_str = f'{d:.3f}' if not np.isnan(d) else 'N/A'

        print(
            f'{desc:<35} {seeds_str:>10} {n:>3} '
            f'{t_str:>8} {pr_str:>8} {pc_str:>8} '
            f'{d_str:>7} {effect:>12} {sig:>5}'
        )

    print('-' * 110)
    print()


# ======================================================================
# Main
# ======================================================================

def main():
    """Run the statistical significance analysis pipeline."""
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Load results
    print(f'Loading results from: {args.results_dir}')
    all_results = load_all_results(args.results_dir)
    print(f'  Loaded {len(all_results)} result files')

    if not all_results:
        print('No result files found. Exiting.')
        return

    # Group by (dataset, method) with explicit seed tracking
    seed_groups = group_results_by_seed(all_results)
    print(f'  Found {len(seed_groups)} (dataset, method) groups')

    # Identify unique datasets
    datasets = sorted(set(key[0] for key in seed_groups.keys()))
    print(f'  Datasets: {datasets}')

    # Show seed inventory
    print('\n  Seed inventory:')
    for (ds, method), seed_map in sorted(seed_groups.items()):
        seeds = sorted(seed_map.keys())
        print(f'    ({ds}, {method}): seeds={seeds}, n={len(seeds)}')

    # Aggregate output
    full_output = {
        'results_dir': os.path.abspath(args.results_dir),
        'total_files_loaded': len(all_results),
        'n_bootstrap': args.n_bootstrap,
        'n_comparisons_bonferroni': len(KEY_COMPARISONS),
        'datasets': {},
    }

    all_latex_tables = []
    rng = np.random.default_rng(42)

    for dataset in datasets:
        print(f'\n--- Dataset: {dataset} ---')

        # Collect method stats with bootstrap CIs
        method_stats = {}
        for (ds, method), seed_map in seed_groups.items():
            if ds != dataset:
                continue
            if len(seed_map) < 2:
                print(f'  Skipping {method}: only {len(seed_map)} seed(s) (need >= 2)')
                continue

            stats = compute_method_stats(seed_map)
            accs = np.array([seed_map[s] for s in sorted(seed_map.keys())])
            ci_lower, ci_upper = bootstrap_ci(accs, n_bootstrap=args.n_bootstrap, rng=rng)
            stats['ci_lower'] = ci_lower
            stats['ci_upper'] = ci_upper
            method_stats[method] = stats

        print(f'  Methods with >= 2 seeds: {list(method_stats.keys())}')

        if not method_stats:
            print('  No methods with sufficient seeds. Skipping dataset.')
            continue

        # Run key comparisons with proper seed pairing
        comparisons = run_comparisons(
            seed_groups, dataset=dataset, n_bootstrap=args.n_bootstrap,
        )

        # Build lookup: method_b -> comparison result (for p-value vs baseline)
        comparisons_vs_baseline = {}
        for comp in comparisons:
            if comp['method_a'] == 'baseline':
                comparisons_vs_baseline[comp['method_b']] = comp

        # Print tables
        print_results_table(method_stats, comparisons_vs_baseline)
        print_comparisons(comparisons)

        # Generate LaTeX table
        latex_table = generate_latex_table(
            method_stats, comparisons_vs_baseline, dataset=dataset,
        )
        all_latex_tables.append(latex_table)
        print('LaTeX table generated.')

        # Store in output dict
        ds_output = {
            'method_stats': {},
            'comparisons': [],
        }
        for method, stats in method_stats.items():
            entry = dict(stats)
            if method in comparisons_vs_baseline:
                comp = comparisons_vs_baseline[method]
                entry['p_value_raw_vs_baseline'] = comp['p_value_raw']
                entry['p_value_corrected_vs_baseline'] = comp['p_value_corrected']
                entry['t_stat_vs_baseline'] = comp['t_statistic']
                entry['cohens_d_vs_baseline'] = comp['cohens_d']
                entry['effect_label_vs_baseline'] = comp['effect_label']
                entry['significance_raw'] = comp['marker_raw']
                entry['significance_corrected'] = comp['marker_corrected']
            ds_output['method_stats'][method] = entry

        # Serialize comparisons (convert seeds to strings for JSON)
        for comp in comparisons:
            comp_copy = dict(comp)
            comp_copy['seeds_used'] = [int(s) for s in comp_copy['seeds_used']]
            ds_output['comparisons'].append(comp_copy)

        full_output['datasets'][dataset] = ds_output

    # Save JSON results
    json_path = os.path.join(args.save_dir, 'significance_results.json')
    with open(json_path, 'w') as f:
        json.dump(full_output, f, indent=2)
    print(f'\nSignificance results saved to: {json_path}')

    # Save LaTeX table
    tex_path = os.path.join(args.save_dir, 'significance_table.tex')
    with open(tex_path, 'w') as f:
        f.write('% Auto-generated significance tables\n')
        f.write('% Generated by scripts/statistical_significance.py\n')
        f.write('% p-values are Bonferroni-corrected for '
                f'{len(KEY_COMPARISONS)} comparisons\n\n')
        for table in all_latex_tables:
            f.write(table)
            f.write('\n\n')
    print(f'LaTeX table saved to: {tex_path}')


if __name__ == '__main__':
    main()
