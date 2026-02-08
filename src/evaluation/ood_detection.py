"""Out-of-distribution (OOD) detection using UCAT attention temperatures.

UCAT trains attention temperatures to reflect prediction uncertainty.
This module exploits that property for OOD detection *without any
additional training*: in-distribution inputs (EuroSAT) produce low
temperatures, while out-of-distribution inputs (e.g., DTD textures)
produce high temperatures.

Metrics reported:
    - AUROC: area under ROC curve (temperature as OOD score)
    - AUPR: area under precision-recall curve
    - FPR@95: false positive rate at 95% true positive rate
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def collect_temperatures(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
    component: str = "total",
) -> np.ndarray:
    """Run inference and collect per-sample attention temperatures.

    Parameters
    ----------
    model : nn.Module
        Trained EfficientEuroSATViT model with learned temperatures.
    dataloader : DataLoader
        Data loader for the dataset to evaluate.
    device : str
        Device for computation.
    component : str
        Which temperature component to collect: ``'total'``, ``'aleatoric'``,
        or ``'epistemic'``. Only ``'total'`` is valid for non-decomposed models.

    Returns
    -------
    numpy.ndarray
        1-D array of per-sample mean temperatures.
    """
    model.eval()
    temps: List[float] = []
    is_decomposed = getattr(model, 'use_decomposition', False)

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            result = model(images, return_temperatures=True)

            if is_decomposed and len(result) == 4:
                logits, tau_total, tau_a, tau_e = result
                if component == "aleatoric":
                    batch_temps = tau_a
                elif component == "epistemic":
                    batch_temps = tau_e.expand(logits.shape[0])
                else:
                    batch_temps = tau_total
            else:
                _, batch_temps = result

            temps.extend(batch_temps.cpu().numpy().tolist())

    return np.array(temps)


def compute_ood_metrics(
    in_temps: np.ndarray,
    ood_temps: np.ndarray,
) -> Dict[str, float]:
    """Compute OOD detection metrics using temperature as the score.

    Higher temperature is treated as more likely OOD.

    Parameters
    ----------
    in_temps : numpy.ndarray
        Temperatures from in-distribution data.
    ood_temps : numpy.ndarray
        Temperatures from out-of-distribution data.

    Returns
    -------
    dict
        ``'auroc'``, ``'aupr'``, ``'fpr_at_95_tpr'``,
        ``'mean_in_temp'``, ``'mean_ood_temp'``, ``'temp_gap'``.
    """
    # Labels: 0 = in-distribution, 1 = OOD
    labels = np.concatenate([
        np.zeros(len(in_temps)),
        np.ones(len(ood_temps)),
    ])
    scores = np.concatenate([in_temps, ood_temps])

    # AUROC
    auroc = _compute_auroc(labels, scores)

    # AUPR
    aupr = _compute_aupr(labels, scores)

    # FPR at 95% TPR
    fpr_at_95 = _compute_fpr_at_tpr(labels, scores, target_tpr=0.95)

    mean_in = float(in_temps.mean())
    mean_ood = float(ood_temps.mean())

    return {
        "auroc": auroc,
        "aupr": aupr,
        "fpr_at_95_tpr": fpr_at_95,
        "mean_in_temp": mean_in,
        "mean_ood_temp": mean_ood,
        "temp_gap": mean_ood - mean_in,
    }


def _compute_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUROC using the trapezoidal rule (no sklearn dependency)."""
    # Sort by descending score
    order = np.argsort(-scores)
    labels_sorted = labels[order]

    n_pos = labels.sum()
    n_neg = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.0

    tpr_list = []
    fpr_list = []
    tp = 0
    fp = 0

    for label in labels_sorted:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)

    # Trapezoidal integration
    fpr_arr = np.array(fpr_list)
    tpr_arr = np.array(tpr_list)

    auroc = float(np.trapz(tpr_arr, fpr_arr))
    return auroc


def _compute_aupr(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute area under precision-recall curve."""
    order = np.argsort(-scores)
    labels_sorted = labels[order]

    n_pos = labels.sum()
    if n_pos == 0:
        return 0.0

    tp = 0
    fp = 0
    precisions = []
    recalls = []

    for label in labels_sorted:
        if label == 1:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp)
        recall = tp / n_pos
        precisions.append(precision)
        recalls.append(recall)

    recalls_arr = np.array(recalls)
    precisions_arr = np.array(precisions)

    aupr = float(np.trapz(precisions_arr, recalls_arr))
    return aupr


def _compute_fpr_at_tpr(
    labels: np.ndarray,
    scores: np.ndarray,
    target_tpr: float = 0.95,
) -> float:
    """Compute FPR at a given TPR threshold."""
    order = np.argsort(-scores)
    labels_sorted = labels[order]

    n_pos = labels.sum()
    n_neg = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.0

    tp = 0
    fp = 0

    for label in labels_sorted:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        if tpr >= target_tpr:
            return float(fp / n_neg)

    return float(fp / n_neg)


def get_ood_dataloader(
    ood_dataset: str = "dtd",
    root: str = "./data",
    batch_size: int = 64,
    num_workers: int = 4,
    img_size: int = 224,
) -> DataLoader:
    """Create a DataLoader for an OOD dataset.

    Parameters
    ----------
    ood_dataset : str
        Name of the OOD dataset. Supported: ``'dtd'``.
    root : str
        Root directory for data storage.
    batch_size : int
        Mini-batch size.
    num_workers : int
        Number of data loading workers.
    img_size : int
        Image resolution for transforms.

    Returns
    -------
    DataLoader
        DataLoader yielding ``(images, labels)`` tuples.
    """
    from ..data.transforms import get_test_transform

    transform = get_test_transform(img_size=img_size)

    if ood_dataset == "dtd":
        from torchvision.datasets import DTD

        dataset = DTD(
            root=root,
            split="test",
            download=True,
            transform=transform,
        )
    else:
        raise ValueError(f"Unsupported OOD dataset: {ood_dataset}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def run_ood_analysis(
    model: nn.Module,
    in_loader: DataLoader,
    ood_loader: DataLoader,
    device: str = "cpu",
    save_path: Optional[str] = "figures/ood_detection.png",
) -> Dict[str, Any]:
    """Run full OOD detection analysis and optionally save a histogram.

    Parameters
    ----------
    model : nn.Module
        Trained EfficientEuroSATViT model.
    in_loader : DataLoader
        In-distribution (EuroSAT) evaluation data.
    ood_loader : DataLoader
        Out-of-distribution data.
    device : str
        Device for computation.
    save_path : str or None
        If provided, save a temperature histogram to this path.

    Returns
    -------
    dict
        OOD metrics and raw temperature arrays.
    """
    in_temps = collect_temperatures(model, in_loader, device=device)
    ood_temps = collect_temperatures(model, ood_loader, device=device)

    metrics = compute_ood_metrics(in_temps, ood_temps)

    print(f"[OOD Detection] AUROC: {metrics['auroc']:.4f}")
    print(f"[OOD Detection] AUPR: {metrics['aupr']:.4f}")
    print(f"[OOD Detection] FPR@95: {metrics['fpr_at_95_tpr']:.4f}")
    print(f"[OOD Detection] Mean temp (in): {metrics['mean_in_temp']:.4f}")
    print(f"[OOD Detection] Mean temp (OOD): {metrics['mean_ood_temp']:.4f}")
    print(f"[OOD Detection] Temp gap: {metrics['temp_gap']:.4f}")

    if save_path is not None:
        _plot_ood_histogram(in_temps, ood_temps, metrics, save_path)

    return {**metrics, "in_temps": in_temps, "ood_temps": ood_temps}


def _plot_ood_histogram(
    in_temps: np.ndarray,
    ood_temps: np.ndarray,
    metrics: Dict[str, float],
    save_path: str,
) -> None:
    """Plot overlapping histograms of in-distribution vs OOD temperatures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import os

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        in_temps, bins=50, alpha=0.6, density=True,
        label=f"In-distribution (EuroSAT, \u03bc={in_temps.mean():.3f})",
        color="steelblue",
    )
    ax.hist(
        ood_temps, bins=50, alpha=0.6, density=True,
        label=f"OOD (DTD, \u03bc={ood_temps.mean():.3f})",
        color="coral",
    )

    ax.set_xlabel("Attention Temperature (\u03c4)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        f"OOD Detection via UCAT Temperature "
        f"(AUROC={metrics['auroc']:.3f})",
        fontsize=14,
    )
    ax.legend(fontsize=11)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved OOD histogram to {save_path}")
