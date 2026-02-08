#!/usr/bin/env python3
"""Statistical significance tests for TNNLS paper results.

Computes paired t-tests and generates LaTeX tables comparing ablation
methods against the baseline for the EfficientEuroSAT project.

Usage:
    python statistical_significance.py --results_dir ./results
    python statistical_significance.py --results_dir ./ablation_results/individual
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
    return parser.parse_args()


# ======================================================================
# Result loading and grouping
# ======================================================================

def strip_seed_suffix(name):
    """Strip the seed suffix from a method name.

    Examples:
        'all_combined_s42'   -> 'all_combined'
        'baseline_seed123'   -> 'baseline'
        'ucat_only_s456'     -> 'ucat_only'
        'my_method'          -> 'my_method'
    """
    # Match patterns like _s42, _s123, _seed42, _seed123 at end of string
    cleaned = re.sub(r'[_\-](?:s|seed)\d+$', '', name)
    return cleaned


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

    Checks for nested ``results['test']['test_acc']`` first, then
    common top-level keys.

    Returns
    -------
    float or None
        Test accuracy value, or None if not found.
    """
    # Primary: results['test']['test_acc']
    if 'test' in result and isinstance(result['test'], dict):
        if 'test_acc' in result['test']:
            return float(result['test']['test_acc'])

    # Alternatives used in ablation results
    for key in ('test_accuracy', 'test_acc', 'accuracy'):
        if key in result and result[key] is not None:
            return float(result[key])

    return None


def extract_dataset(result):
    """Extract the dataset name from a result dict."""
    return result.get('dataset', 'eurosat')


def group_results(results):
    """Group results by (dataset, method) after stripping seed suffixes.

    Parameters
    ----------
    results : list of dict
        Loaded result dictionaries.

    Returns
    -------
    dict
        Mapping of ``(dataset, method_name) -> list of test_acc values``.
    dict
        Mapping of ``(dataset, method_name) -> list of result dicts``
        (for further inspection if needed).
    """
    acc_groups = defaultdict(list)
    result_groups = defaultdict(list)

    for result in results:
        raw_name = extract_method_name(result)
        method = strip_seed_suffix(raw_name)
        dataset = extract_dataset(result)
        acc = extract_test_accuracy(result)

        if acc is None:
            continue

        key = (dataset, method)
        acc_groups[key].append(acc)
        result_groups[key].append(result)

    return dict(acc_groups), dict(result_groups)


# ======================================================================
# Statistical computations
# ======================================================================

def compute_method_stats(accuracies):
    """Compute summary statistics for a list of accuracy values.

    Parameters
    ----------
    accuracies : list of float
        Test accuracy values across seeds.

    Returns
    -------
    dict
        Dictionary with keys: mean, std, min, max, count.
    """
    arr = np.array(accuracies, dtype=np.float64)
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'count': len(arr),
    }


def paired_ttest(accs_a, accs_b):
    """Run a paired t-test between two sets of accuracy values.

    Both arrays must have the same length (paired by seed).  If either
    has fewer than 2 elements or lengths differ, returns NaN.

    Parameters
    ----------
    accs_a : list of float
        Accuracy values for method A (one per seed).
    accs_b : list of float
        Accuracy values for method B (one per seed).

    Returns
    -------
    tuple of (float, float)
        (t-statistic, p-value).  Returns (NaN, NaN) if the test
        cannot be computed.
    """
    a = np.array(accs_a, dtype=np.float64)
    b = np.array(accs_b, dtype=np.float64)

    if len(a) < 2 or len(b) < 2 or len(a) != len(b):
        return float('nan'), float('nan')

    # Guard against identical arrays (t-test undefined)
    if np.allclose(a, b):
        return 0.0, 1.0

    try:
        t_stat, p_value = ttest_rel(a, b)
        return float(t_stat), float(p_value)
    except Exception:
        return float('nan'), float('nan')


def significance_marker(p_value):
    """Return a significance marker string for a given p-value.

    Returns
    -------
    str
        '***' if p < 0.001, '**' if p < 0.01, '*' if p < 0.05,
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

# Ablation names that are expected in typical results
SINGLE_ABLATIONS = [
    'ucat_only',
    'early_exit_only',
    'dropout_only',
    'residual_only',
    'annealing_only',
]

KEY_COMPARISONS = [
    # (method_a, method_b, description)
    ('baseline', 'all_combined', 'Baseline vs All Combined'),
]

# Baseline vs each single ablation
for ablation in SINGLE_ABLATIONS:
    KEY_COMPARISONS.append(
        ('baseline', ablation, f'Baseline vs {ablation}')
    )

# All combined vs decomposition variant
KEY_COMPARISONS.append(
    ('all_combined', 'decomp_with_losses', 'All Combined vs Decomposed')
)


def run_comparisons(acc_groups, dataset='eurosat'):
    """Run all key comparisons for a given dataset.

    Parameters
    ----------
    acc_groups : dict
        Mapping of ``(dataset, method) -> [accuracies]``.
    dataset : str
        Dataset name to filter on.

    Returns
    -------
    list of dict
        Each dict has keys: method_a, method_b, description,
        t_stat, p_value, marker.
    """
    comparison_results = []

    for method_a, method_b, desc in KEY_COMPARISONS:
        key_a = (dataset, method_a)
        key_b = (dataset, method_b)

        accs_a = acc_groups.get(key_a, [])
        accs_b = acc_groups.get(key_b, [])

        if not accs_a or not accs_b:
            continue

        # Align by taking the minimum shared length (paired by seed order)
        n = min(len(accs_a), len(accs_b))
        if n < 2:
            continue

        t_stat, p_value = paired_ttest(accs_a[:n], accs_b[:n])
        marker = significance_marker(p_value)

        comparison_results.append({
            'method_a': method_a,
            'method_b': method_b,
            'description': desc,
            'n_pairs': n,
            't_statistic': t_stat,
            'p_value': p_value,
            'significance': marker,
        })

    return comparison_results


# ======================================================================
# LaTeX table generation
# ======================================================================

def generate_latex_table(method_stats, comparisons_vs_baseline, dataset='eurosat'):
    """Generate a LaTeX table with method accuracies and p-values vs baseline.

    Parameters
    ----------
    method_stats : dict
        Mapping of method_name -> stats dict (mean, std, ...).
    comparisons_vs_baseline : dict
        Mapping of method_name -> dict with p_value and marker keys.
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
        + r'.}'
    )
    lines.append(r'  \label{tab:significance_' + dataset + r'}')
    lines.append(r'  \begin{tabular}{lcc}')
    lines.append(r'    \toprule')
    lines.append(r'    Method & Accuracy (mean $\pm$ std) & $p$-value vs baseline \\')
    lines.append(r'    \midrule')

    # Sort methods: baseline first, then alphabetically
    sorted_methods = sorted(method_stats.keys())
    if 'baseline' in sorted_methods:
        sorted_methods.remove('baseline')
        sorted_methods.insert(0, 'baseline')

    for method in sorted_methods:
        stats = method_stats[method]
        if stats['count'] < 1:
            continue

        # Format accuracy as percentage
        mean_pct = stats['mean'] * 100 if stats['mean'] <= 1.0 else stats['mean']
        std_pct = stats['std'] * 100 if stats['mean'] <= 1.0 else stats['std']
        acc_str = f'{mean_pct:.2f} $\\pm$ {std_pct:.2f}'

        # p-value string
        if method == 'baseline':
            p_str = '---'
        elif method in comparisons_vs_baseline:
            comp = comparisons_vs_baseline[method]
            p_val = comp['p_value']
            marker = comp['significance']
            if np.isnan(p_val):
                p_str = 'N/A'
            else:
                p_str = f'{p_val:.4f}{marker}'
        else:
            p_str = 'N/A'

        # Escape underscores for LaTeX
        method_tex = method.replace('_', r'\_')
        lines.append(f'    {method_tex} & {acc_str} & {p_str} \\\\')

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'\end{table}')

    return '\n'.join(lines)


# ======================================================================
# Console output
# ======================================================================

def print_results_table(method_stats, comparisons_vs_baseline):
    """Print a formatted results table to stdout."""
    print()
    print('=' * 80)
    print('STATISTICAL SIGNIFICANCE RESULTS')
    print('=' * 80)

    header = f"{'Method':<25} {'Accuracy (mean +/- std)':>25} {'p-value vs baseline':>20}"
    print(header)
    print('-' * 80)

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
        acc_str = f'{mean_pct:.2f} +/- {std_pct:.2f} (n={stats["count"]})'

        if method == 'baseline':
            p_str = '---'
        elif method in comparisons_vs_baseline:
            comp = comparisons_vs_baseline[method]
            p_val = comp['p_value']
            marker = comp['significance']
            if np.isnan(p_val):
                p_str = 'N/A'
            else:
                p_str = f'{p_val:.4f} {marker}'
        else:
            p_str = 'N/A'

        print(f'{method:<25} {acc_str:>25} {p_str:>20}')

    print('=' * 80)
    print('Significance: * p<0.05, ** p<0.01, *** p<0.001')
    print()


def print_comparisons(comparisons):
    """Print the detailed comparison results."""
    if not comparisons:
        print('No paired comparisons could be computed.')
        return

    print()
    print('PAIRED T-TEST COMPARISONS')
    print('-' * 80)
    header = f"{'Comparison':<40} {'n':>4} {'t-stat':>10} {'p-value':>10} {'Sig':>5}"
    print(header)
    print('-' * 80)

    for comp in comparisons:
        desc = comp['description']
        n = comp['n_pairs']
        t = comp['t_statistic']
        p = comp['p_value']
        sig = comp['significance']

        t_str = f'{t:.4f}' if not np.isnan(t) else 'N/A'
        p_str = f'{p:.4f}' if not np.isnan(p) else 'N/A'

        print(f'{desc:<40} {n:>4} {t_str:>10} {p_str:>10} {sig:>5}')

    print('-' * 80)
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

    # Group by (dataset, method)
    acc_groups, result_groups = group_results(all_results)
    print(f'  Found {len(acc_groups)} (dataset, method) groups')

    # Identify unique datasets
    datasets = sorted(set(key[0] for key in acc_groups.keys()))
    print(f'  Datasets: {datasets}')

    # Aggregate output
    full_output = {
        'results_dir': os.path.abspath(args.results_dir),
        'total_files_loaded': len(all_results),
        'datasets': {},
    }

    all_latex_tables = []

    for dataset in datasets:
        print(f'\n--- Dataset: {dataset} ---')

        # Collect method stats for this dataset
        method_stats = {}
        for (ds, method), accs in acc_groups.items():
            if ds != dataset:
                continue
            if len(accs) < 3:
                print(f'  Skipping {method}: only {len(accs)} seeds (need >= 3)')
                continue
            method_stats[method] = compute_method_stats(accs)

        print(f'  Methods with >= 3 seeds: {list(method_stats.keys())}')

        if not method_stats:
            print('  No methods with sufficient seeds. Skipping dataset.')
            continue

        # Run key comparisons
        comparisons = run_comparisons(acc_groups, dataset=dataset)

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
            'comparisons': comparisons,
        }
        for method, stats in method_stats.items():
            entry = dict(stats)
            if method in comparisons_vs_baseline:
                entry['p_value_vs_baseline'] = comparisons_vs_baseline[method]['p_value']
                entry['t_stat_vs_baseline'] = comparisons_vs_baseline[method]['t_statistic']
                entry['significance_vs_baseline'] = comparisons_vs_baseline[method]['significance']
            ds_output['method_stats'][method] = entry

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
        f.write('% Generated by scripts/statistical_significance.py\n\n')
        for table in all_latex_tables:
            f.write(table)
            f.write('\n\n')
    print(f'LaTeX table saved to: {tex_path}')


if __name__ == '__main__':
    main()
