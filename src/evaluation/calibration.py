"""Prediction calibration analysis for UCAT models.

Evaluates how well model confidence aligns with actual accuracy using
Expected Calibration Error (ECE) and related metrics.  UCAT models are
expected to produce better-calibrated predictions because the
uncertainty-aware temperature mechanism prevents overconfidence.

Metrics:
    - ECE: Expected Calibration Error (weighted bin gap)
    - MCE: Maximum Calibration Error (worst bin gap)
    - Reliability diagram (optional plot)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_ece(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 15,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Expected Calibration Error and Maximum Calibration Error.

    Parameters
    ----------
    confidences : numpy.ndarray
        Per-sample maximum softmax probabilities, shape ``(N,)``.
    accuracies : numpy.ndarray
        Per-sample binary correctness indicators, shape ``(N,)``.
    n_bins : int
        Number of equal-width confidence bins.

    Returns
    -------
    tuple
        ``(ece, mce, bin_confidences, bin_accuracies, bin_counts)``
        where ``ece`` and ``mce`` are scalars, and the arrays are
        per-bin statistics of length ``n_bins``.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_confidences = np.zeros(n_bins)
    bin_accuracies = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lo = bin_boundaries[i]
        hi = bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        count = mask.sum()
        bin_counts[i] = count
        if count > 0:
            bin_confidences[i] = confidences[mask].mean()
            bin_accuracies[i] = accuracies[mask].mean()

    n_total = len(confidences)
    ece = float(
        np.sum(bin_counts / max(n_total, 1) * np.abs(bin_accuracies - bin_confidences))
    )
    mce = float(np.max(np.abs(bin_accuracies - bin_confidences)))

    return ece, mce, bin_confidences, bin_accuracies, bin_counts


def evaluate_calibration(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
    n_bins: int = 15,
) -> Dict[str, Any]:
    """Evaluate prediction calibration of a model.

    Parameters
    ----------
    model : nn.Module
        Trained model.
    dataloader : DataLoader
        Evaluation data loader.
    device : str
        Device for computation.
    n_bins : int
        Number of calibration bins.

    Returns
    -------
    dict
        ``'ece'``, ``'mce'``, ``'accuracy'``, ``'mean_confidence'``,
        ``'bin_confidences'``, ``'bin_accuracies'``, ``'bin_counts'``.
    """
    model.eval()
    all_confidences: List[float] = []
    all_correct: List[float] = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probs = torch.softmax(logits, dim=-1)

            confidence, predicted = probs.max(dim=-1)
            correct = (predicted == labels).float()

            all_confidences.extend(confidence.cpu().numpy().tolist())
            all_correct.extend(correct.cpu().numpy().tolist())

    confidences = np.array(all_confidences)
    accuracies = np.array(all_correct)

    ece, mce, bin_conf, bin_acc, bin_counts = compute_ece(
        confidences, accuracies, n_bins=n_bins,
    )

    overall_acc = float(accuracies.mean())
    mean_conf = float(confidences.mean())

    return {
        "ece": ece,
        "mce": mce,
        "accuracy": overall_acc,
        "mean_confidence": mean_conf,
        "overconfidence_gap": mean_conf - overall_acc,
        "bin_confidences": bin_conf,
        "bin_accuracies": bin_acc,
        "bin_counts": bin_counts,
    }


def compare_calibration(
    baseline_results: Dict[str, Any],
    ucat_results: Dict[str, Any],
) -> Dict[str, float]:
    """Compare calibration between baseline and UCAT model.

    Parameters
    ----------
    baseline_results : dict
        Output of :func:`evaluate_calibration` for the baseline model.
    ucat_results : dict
        Output of :func:`evaluate_calibration` for the UCAT model.

    Returns
    -------
    dict
        ``'baseline_ece'``, ``'ucat_ece'``, ``'ece_reduction'``,
        ``'ece_reduction_pct'``, and similar for MCE.
    """
    b_ece = baseline_results["ece"]
    u_ece = ucat_results["ece"]
    b_mce = baseline_results["mce"]
    u_mce = ucat_results["mce"]

    return {
        "baseline_ece": b_ece,
        "ucat_ece": u_ece,
        "ece_reduction": b_ece - u_ece,
        "ece_reduction_pct": (b_ece - u_ece) / max(b_ece, 1e-8) * 100,
        "baseline_mce": b_mce,
        "ucat_mce": u_mce,
        "mce_reduction": b_mce - u_mce,
        "mce_reduction_pct": (b_mce - u_mce) / max(b_mce, 1e-8) * 100,
    }


def plot_reliability_diagram(
    results: Dict[str, Any],
    save_path: str = "figures/reliability_diagram.png",
    label: str = "Model",
) -> None:
    """Plot a reliability diagram (calibration curve).

    Parameters
    ----------
    results : dict
        Output of :func:`evaluate_calibration`.
    save_path : str
        Path to save the figure.
    label : str
        Legend label for the calibration curve.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import os

    bin_conf = results["bin_confidences"]
    bin_acc = results["bin_accuracies"]
    bin_counts = results["bin_counts"]
    ece = results["ece"]

    mask = bin_counts > 0

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]},
    )

    # Reliability diagram
    ax1.bar(
        bin_conf[mask], bin_acc[mask],
        width=1.0 / len(bin_conf), alpha=0.6,
        edgecolor="black", linewidth=0.5,
        label=f"{label} (ECE={ece:.4f})",
    )
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax1.set_xlabel("Confidence", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title("Reliability Diagram", fontsize=14)
    ax1.legend(fontsize=11)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # Histogram of confidence distribution
    n_bins = len(bin_counts)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax2.bar(bin_centers, bin_counts, width=1.0 / n_bins, alpha=0.6, color="gray")
    ax2.set_xlabel("Confidence", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved reliability diagram to {save_path}")


def plot_calibration_comparison(
    baseline_results: Dict[str, Any],
    ucat_results: Dict[str, Any],
    save_path: str = "figures/calibration_comparison.png",
) -> None:
    """Plot side-by-side reliability diagrams for baseline vs UCAT.

    Parameters
    ----------
    baseline_results : dict
        Baseline model calibration results.
    ucat_results : dict
        UCAT model calibration results.
    save_path : str
        Path to save the figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import os

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, results, title in [
        (ax1, baseline_results, f"Baseline (ECE={baseline_results['ece']:.4f})"),
        (ax2, ucat_results, f"UCAT (ECE={ucat_results['ece']:.4f})"),
    ]:
        bin_conf = results["bin_confidences"]
        bin_acc = results["bin_accuracies"]
        bin_counts = results["bin_counts"]
        mask = bin_counts > 0

        ax.bar(
            bin_conf[mask], bin_acc[mask],
            width=1.0 / len(bin_conf), alpha=0.6,
            edgecolor="black", linewidth=0.5,
        )
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_xlabel("Confidence", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.suptitle("Calibration: Baseline vs UCAT", fontsize=14)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved calibration comparison to {save_path}")
