from .metrics import compute_accuracy, compute_top_k_accuracy, compute_per_class_accuracy, compute_all_metrics
from .latency import benchmark_latency, benchmark_with_early_exit, compare_latency
from .early_exit_stats import collect_exit_statistics, analyze_exit_by_difficulty
from .confusion import compute_confusion_matrix, plot_confusion_matrix, get_most_confused_pairs
from .ucat_analysis import (
    compute_temperature_entropy_correlation,
    analyze_correct_vs_incorrect,
    plot_temperature_entropy,
)
from .ood_detection import (
    collect_temperatures,
    compute_ood_metrics,
    run_ood_analysis,
)
from .calibration import (
    compute_ece,
    evaluate_calibration,
    compare_calibration,
    plot_reliability_diagram,
    plot_calibration_comparison,
)
from .robustness import (
    evaluate_corruption,
    run_robustness_analysis,
    summarize_robustness,
    plot_robustness,
)

__all__ = [
    'compute_accuracy',
    'compute_top_k_accuracy',
    'compute_per_class_accuracy',
    'compute_all_metrics',
    'benchmark_latency',
    'benchmark_with_early_exit',
    'compare_latency',
    'collect_exit_statistics',
    'analyze_exit_by_difficulty',
    'compute_confusion_matrix',
    'plot_confusion_matrix',
    'get_most_confused_pairs',
    'compute_temperature_entropy_correlation',
    'analyze_correct_vs_incorrect',
    'plot_temperature_entropy',
    'collect_temperatures',
    'compute_ood_metrics',
    'run_ood_analysis',
    'compute_ece',
    'evaluate_calibration',
    'compare_calibration',
    'plot_reliability_diagram',
    'plot_calibration_comparison',
    'evaluate_corruption',
    'run_robustness_analysis',
    'summarize_robustness',
    'plot_robustness',
]
