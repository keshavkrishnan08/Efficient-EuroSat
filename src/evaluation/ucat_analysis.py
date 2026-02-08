"""UCAT (Uncertainty-Calibrated Attention Temperature) analysis tools.

Provides functions to evaluate the correlation between learned attention
temperatures and prediction uncertainty, supporting the core claim that
UCAT trains temperatures to reflect model confidence.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


def compute_temperature_entropy_correlation(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Compute correlation between attention temperature and prediction entropy.

    Parameters
    ----------
    model : nn.Module
        Trained EfficientEuroSATViT model.
    dataloader : DataLoader
        Evaluation data loader.
    device : str
        Device for computation.

    Returns
    -------
    dict
        ``'temperatures'``, ``'entropies'``, ``'correct'`` (numpy arrays),
        ``'correlation'``, ``'p_value'`` (floats).
    """
    model.eval()

    all_temps: List[float] = []
    all_entropies: List[float] = []
    all_correct: List[float] = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits, temps = model(images, return_temperatures=True)

            # Prediction entropy
            probs = torch.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)

            # Correctness
            preds = logits.argmax(dim=-1)
            correct = (preds == labels).float()

            all_temps.extend(temps.cpu().numpy().tolist())
            all_entropies.extend(entropy.cpu().numpy().tolist())
            all_correct.extend(correct.cpu().numpy().tolist())

    temps_arr = np.array(all_temps)
    entropies_arr = np.array(all_entropies)
    correct_arr = np.array(all_correct)

    # Pearson correlation
    if temps_arr.std() > 1e-8 and entropies_arr.std() > 1e-8:
        correlation = float(np.corrcoef(temps_arr, entropies_arr)[0, 1])
    else:
        correlation = 0.0

    return {
        "temperatures": temps_arr,
        "entropies": entropies_arr,
        "correct": correct_arr,
        "correlation": correlation,
    }


def analyze_correct_vs_incorrect(results: Dict[str, Any]) -> Dict[str, float]:
    """Compare mean temperature for correct vs incorrect predictions.

    Parameters
    ----------
    results : dict
        Output of :func:`compute_temperature_entropy_correlation`.

    Returns
    -------
    dict
        ``'temp_correct'``, ``'temp_incorrect'``, ``'ratio'``.
    """
    temps = results["temperatures"]
    correct = results["correct"]

    mask_correct = correct == 1.0
    mask_incorrect = correct == 0.0

    temp_correct = float(temps[mask_correct].mean()) if mask_correct.any() else 0.0
    temp_incorrect = float(temps[mask_incorrect].mean()) if mask_incorrect.any() else 0.0
    ratio = temp_incorrect / max(temp_correct, 1e-8)

    return {
        "temp_correct": temp_correct,
        "temp_incorrect": temp_incorrect,
        "ratio": ratio,
    }


def plot_temperature_entropy(
    results: Dict[str, Any],
    save_path: str = "figures/ucat_correlation.png",
) -> None:
    """Plot temperature vs entropy scatter with regression line.

    Parameters
    ----------
    results : dict
        Output of :func:`compute_temperature_entropy_correlation`.
    save_path : str
        File path for the saved figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    temps = results["temperatures"]
    entropies = results["entropies"]
    correct = results["correct"]
    corr = results["correlation"]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["green" if c else "red" for c in correct]
    ax.scatter(entropies, temps, c=colors, alpha=0.4, s=8)

    # Regression line
    if len(entropies) > 2:
        z = np.polyfit(entropies, temps, 1)
        p = np.poly1d(z)
        x_line = np.linspace(entropies.min(), entropies.max(), 100)
        ax.plot(x_line, p(x_line), "b--", linewidth=2, label=f"r = {corr:.3f}")

    ax.set_xlabel("Prediction Entropy (Uncertainty)", fontsize=12)
    ax.set_ylabel("Attention Temperature (\u03c4)", fontsize=12)
    ax.set_title("UCAT: Temperature\u2013Uncertainty Correlation", fontsize=14)
    ax.legend(fontsize=11)

    fig.tight_layout()

    import os
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved UCAT correlation plot to {save_path}")
