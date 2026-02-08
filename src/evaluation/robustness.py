"""Robustness analysis using UCAT attention temperatures.

Evaluates whether learned attention temperatures reliably increase when
inputs are corrupted or degraded.  If temperatures serve as difficulty
indicators, they should be higher for corrupted (harder) inputs and
lower for clean (easier) inputs.

Corruption types tested:
    - Gaussian blur
    - Color jitter
    - Gaussian noise
    - Brightness shift
    - Contrast reduction
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# -----------------------------------------------------------------------
# Corruption transforms
# -----------------------------------------------------------------------

CORRUPTION_TRANSFORMS = {
    "gaussian_blur": transforms.GaussianBlur(kernel_size=7, sigma=(2.0, 2.0)),
    "color_jitter": transforms.ColorJitter(
        brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3,
    ),
    "gaussian_noise": None,  # handled separately (requires tensor input)
    "brightness_up": transforms.ColorJitter(brightness=(1.5, 1.5)),
    "contrast_down": transforms.ColorJitter(contrast=(0.3, 0.3)),
}


class _GaussianNoise:
    """Add Gaussian noise to a tensor image."""

    def __init__(self, std: float = 0.15):
        self.std = std

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if not isinstance(img, torch.Tensor):
            img = transforms.functional.to_tensor(img)
        return (img + torch.randn_like(img) * self.std).clamp(0, 1)


class _CorruptedDataset(Dataset):
    """Wraps a dataset and applies a corruption transform before the
    original transform pipeline.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        corruption: Any,
    ) -> None:
        self.base_dataset = base_dataset
        self.corruption = corruption

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img, label = self.base_dataset[idx]
        if isinstance(img, torch.Tensor) and callable(self.corruption):
            img = self.corruption(img)
        return img, label


def evaluate_corruption(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
) -> Dict[str, float]:
    """Evaluate model on a single data loader, returning accuracy and temperature.

    Parameters
    ----------
    model : nn.Module
        Trained EfficientEuroSATViT model.
    dataloader : DataLoader
        Data loader (clean or corrupted).
    device : str
        Device for computation.

    Returns
    -------
    dict
        ``'accuracy'``, ``'mean_temp'``, ``'std_temp'``.
    """
    model.eval()
    correct = 0
    total = 0
    temps: List[float] = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits, batch_temps = model(images, return_temperatures=True)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            temps.extend(batch_temps.cpu().numpy().tolist())

    temps_arr = np.array(temps)
    return {
        "accuracy": correct / max(total, 1),
        "mean_temp": float(temps_arr.mean()),
        "std_temp": float(temps_arr.std()),
    }


def run_robustness_analysis(
    model: nn.Module,
    test_dataset: Dataset,
    batch_size: int = 64,
    num_workers: int = 4,
    device: str = "cpu",
    noise_std: float = 0.15,
) -> Dict[str, Dict[str, float]]:
    """Run robustness analysis across all corruption types.

    Parameters
    ----------
    model : nn.Module
        Trained EfficientEuroSATViT model.
    test_dataset : Dataset
        Clean test dataset (already has transforms applied).
    batch_size : int
        Mini-batch size.
    num_workers : int
        Data loading workers.
    device : str
        Device for computation.
    noise_std : float
        Standard deviation for Gaussian noise corruption.

    Returns
    -------
    dict
        Mapping from corruption name (plus ``'clean'``) to metrics dict
        containing ``'accuracy'``, ``'mean_temp'``, ``'std_temp'``.
    """
    results: Dict[str, Dict[str, float]] = {}

    # Clean baseline
    clean_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    results["clean"] = evaluate_corruption(model, clean_loader, device=device)
    print(f"[Robustness] clean: acc={results['clean']['accuracy']:.4f}, "
          f"temp={results['clean']['mean_temp']:.4f}")

    # Corrupted versions
    corruption_fns = {
        "gaussian_blur": transforms.GaussianBlur(kernel_size=7, sigma=(2.0, 2.0)),
        "color_jitter": transforms.ColorJitter(
            brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3,
        ),
        "gaussian_noise": _GaussianNoise(std=noise_std),
        "brightness_up": transforms.ColorJitter(brightness=(1.5, 1.5)),
        "contrast_down": transforms.ColorJitter(contrast=(0.3, 0.3)),
    }

    for name, corruption in corruption_fns.items():
        corrupted_ds = _CorruptedDataset(test_dataset, corruption)
        corrupted_loader = DataLoader(
            corrupted_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )
        results[name] = evaluate_corruption(
            model, corrupted_loader, device=device,
        )
        acc = results[name]["accuracy"]
        temp = results[name]["mean_temp"]
        delta = temp - results["clean"]["mean_temp"]
        print(f"[Robustness] {name}: acc={acc:.4f}, "
              f"temp={temp:.4f} (\u0394={delta:+.4f})")

    return results


def summarize_robustness(
    results: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """Compute summary statistics from robustness analysis.

    Parameters
    ----------
    results : dict
        Output of :func:`run_robustness_analysis`.

    Returns
    -------
    dict
        ``'clean_acc'``, ``'mean_corrupted_acc'``, ``'acc_drop'``,
        ``'clean_temp'``, ``'mean_corrupted_temp'``, ``'temp_increase'``,
        ``'temp_acc_correlation'``.
    """
    clean_acc = results["clean"]["accuracy"]
    clean_temp = results["clean"]["mean_temp"]

    corrupted_names = [k for k in results if k != "clean"]
    corrupted_accs = [results[k]["accuracy"] for k in corrupted_names]
    corrupted_temps = [results[k]["mean_temp"] for k in corrupted_names]

    mean_corr_acc = float(np.mean(corrupted_accs))
    mean_corr_temp = float(np.mean(corrupted_temps))

    # Correlation between accuracy drop and temperature increase
    acc_drops = [clean_acc - a for a in corrupted_accs]
    temp_increases = [t - clean_temp for t in corrupted_temps]

    if len(acc_drops) > 2:
        corr = float(np.corrcoef(acc_drops, temp_increases)[0, 1])
    else:
        corr = 0.0

    return {
        "clean_acc": clean_acc,
        "mean_corrupted_acc": mean_corr_acc,
        "acc_drop": clean_acc - mean_corr_acc,
        "clean_temp": clean_temp,
        "mean_corrupted_temp": mean_corr_temp,
        "temp_increase": mean_corr_temp - clean_temp,
        "temp_acc_correlation": corr,
    }


def plot_robustness(
    results: Dict[str, Dict[str, float]],
    save_path: str = "figures/robustness_analysis.png",
) -> None:
    """Plot accuracy and temperature across corruption types.

    Parameters
    ----------
    results : dict
        Output of :func:`run_robustness_analysis`.
    save_path : str
        Path to save the figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import os

    names = list(results.keys())
    accs = [results[n]["accuracy"] for n in names]
    temps = [results[n]["mean_temp"] for n in names]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, accs, width, label="Accuracy", color="steelblue")
    ax1.set_ylabel("Accuracy", fontsize=12, color="steelblue")
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, temps, width, label="Temperature", color="coral")
    ax2.set_ylabel("Temperature (\u03c4)", fontsize=12, color="coral")

    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha="right")
    ax1.set_title("Robustness: Accuracy vs Temperature by Corruption", fontsize=14)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=11)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved robustness plot to {save_path}")
