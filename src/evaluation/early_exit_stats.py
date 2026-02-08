"""
Early exit statistics collection and analysis for EfficientEuroSAT.

Provides utilities to profile how the model's early exit mechanism
behaves across different inputs -- which layers samples exit at,
whether "easy" samples exit earlier than "hard" ones, and what
proportion of total computation is saved.

These statistics are essential for validating that the early exit
controller is functioning as intended and for reporting efficiency
gains in the IEEE paper.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_logits(
    output: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
) -> torch.Tensor:
    """Extract classification logits from model output.

    Parameters
    ----------
    output : torch.Tensor or tuple
        Raw model output (plain logits or ``(logits, exit_info)``).

    Returns
    -------
    torch.Tensor
        Logits tensor.
    """
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def _get_exit_layer(
    output: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    model: nn.Module,
    num_layers: int,
) -> Optional[int]:
    """Attempt to determine the exit layer from model output or attributes.

    Parameters
    ----------
    output : torch.Tensor or tuple
        Raw model output.
    model : nn.Module
        The model, which may expose exit layer tracking attributes.
    num_layers : int
        Total number of layers in the model (used as the default when
        no early exit is detected).

    Returns
    -------
    int or None
        Exit layer index, or ``None`` if it cannot be determined.
    """
    # Check tuple output for exit layer info
    if isinstance(output, (tuple, list)) and len(output) > 1:
        exit_info = output[1]
        if isinstance(exit_info, (int, float)):
            return int(exit_info)
        if isinstance(exit_info, torch.Tensor) and exit_info.numel() == 1:
            return int(exit_info.item())

    # Check model attribute
    if hasattr(model, "last_exit_layer"):
        val = model.last_exit_layer
        if isinstance(val, torch.Tensor):
            return int(val.item()) if val.numel() == 1 else None
        if isinstance(val, (int, float)):
            return int(val)

    return None


def _infer_num_layers(model: nn.Module) -> int:
    """Try to determine the total number of transformer layers.

    Checks common attribute names used in ViT implementations.

    Parameters
    ----------
    model : nn.Module
        The model to inspect.

    Returns
    -------
    int
        Estimated number of layers, defaulting to ``12`` if no
        attribute is found.
    """
    for attr in ("num_layers", "depth", "n_layers", "num_blocks"):
        if hasattr(model, attr):
            val = getattr(model, attr)
            if isinstance(val, int) and val > 0:
                return val

    # Try to count named sub-modules that look like transformer blocks
    block_keywords = ("block", "layer", "encoder_layer")
    for keyword in block_keywords:
        blocks = [
            name for name, _ in model.named_modules()
            if keyword in name.lower()
        ]
        if blocks:
            return len(blocks)

    return 12  # sensible default for ViT-style models


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def collect_exit_statistics(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Collect exit layer statistics from the model's early exit mechanism.

    Runs the model on the entire dataloader and records which layer
    each sample exits at, enabling analysis of computational savings.

    Parameters
    ----------
    model : nn.Module
        Trained model with early exit support.
    dataloader : DataLoader
        Evaluation data loader yielding ``(images, labels)`` batches.
    device : str, optional
        Device for inference.  Default is ``'cuda'``.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``'exit_distribution'`` (Counter): Maps layer index to the
          number of samples that exited at that layer.
        - ``'per_class_exit'`` (dict): Maps class index to a Counter
          of exit layers for that class.
        - ``'avg_exit_layer'`` (float): Mean exit layer across all
          samples.
        - ``'exit_savings_pct'`` (float): Percentage of computation
          saved relative to running all layers for every sample.
          ``0.0`` means no savings; ``100.0`` means all samples
          exited at layer 0.
    """
    model.eval()
    model.to(device)

    num_layers = _infer_num_layers(model)

    exit_distribution: Counter = Counter()
    per_class_exit: Dict[int, Counter] = {}
    all_exit_layers: List[int] = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            exit_layer = _get_exit_layer(output, model, num_layers)

            # If the model does not report a specific exit layer,
            # assume it ran through all layers (no early exit).
            if exit_layer is None:
                exit_layer = num_layers - 1

            batch_size = images.size(0)
            labels_np = labels.cpu().numpy()

            for i in range(batch_size):
                exit_distribution[exit_layer] += 1
                all_exit_layers.append(exit_layer)

                cls_idx = int(labels_np[i])
                if cls_idx not in per_class_exit:
                    per_class_exit[cls_idx] = Counter()
                per_class_exit[cls_idx][exit_layer] += 1

    # Compute summary statistics
    avg_exit = float(np.mean(all_exit_layers)) if all_exit_layers else 0.0

    # Savings: percentage of layers skipped relative to running all
    # layers for every sample.
    # If every sample exits at the last layer, savings = 0%.
    # If every sample exits at layer 0, savings ~ 100%.
    max_layer = num_layers - 1
    if max_layer > 0 and all_exit_layers:
        layers_saved = sum(max_layer - el for el in all_exit_layers)
        total_possible = max_layer * len(all_exit_layers)
        exit_savings_pct = 100.0 * layers_saved / total_possible
    else:
        exit_savings_pct = 0.0

    return {
        "exit_distribution": exit_distribution,
        "per_class_exit": per_class_exit,
        "avg_exit_layer": avg_exit,
        "exit_savings_pct": exit_savings_pct,
    }


def analyze_exit_by_difficulty(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Analyze exit layers grouped by prediction difficulty.

    Samples are categorised as "easy" (correctly predicted) or "hard"
    (incorrectly predicted).  The hypothesis is that easy samples
    trigger early exit at shallower layers while hard samples require
    deeper processing.

    Parameters
    ----------
    model : nn.Module
        Trained model with early exit support.
    dataloader : DataLoader
        Evaluation data loader yielding ``(images, labels)`` batches.
    device : str, optional
        Device for inference.  Default is ``'cuda'``.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``'correct_exit_layers'`` (list[int]): Exit layers for
          correctly predicted samples.
        - ``'incorrect_exit_layers'`` (list[int]): Exit layers for
          incorrectly predicted samples.
        - ``'correct_avg_exit'`` (float): Mean exit layer for correct
          predictions.
        - ``'incorrect_avg_exit'`` (float): Mean exit layer for
          incorrect predictions.
        - ``'correct_exit_distribution'`` (Counter): Exit layer
          histogram for correct predictions.
        - ``'incorrect_exit_distribution'`` (Counter): Exit layer
          histogram for incorrect predictions.
        - ``'total_correct'`` (int): Number of correct predictions.
        - ``'total_incorrect'`` (int): Number of incorrect predictions.
    """
    model.eval()
    model.to(device)

    num_layers = _infer_num_layers(model)

    correct_exits: List[int] = []
    incorrect_exits: List[int] = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            logits = _extract_logits(output)
            preds = logits.argmax(dim=1)

            exit_layer = _get_exit_layer(output, model, num_layers)
            if exit_layer is None:
                exit_layer = num_layers - 1

            is_correct = (preds == labels).cpu().numpy()

            for i in range(images.size(0)):
                if is_correct[i]:
                    correct_exits.append(exit_layer)
                else:
                    incorrect_exits.append(exit_layer)

    correct_avg = float(np.mean(correct_exits)) if correct_exits else 0.0
    incorrect_avg = float(np.mean(incorrect_exits)) if incorrect_exits else 0.0

    return {
        "correct_exit_layers": correct_exits,
        "incorrect_exit_layers": incorrect_exits,
        "correct_avg_exit": correct_avg,
        "incorrect_avg_exit": incorrect_avg,
        "correct_exit_distribution": Counter(correct_exits),
        "incorrect_exit_distribution": Counter(incorrect_exits),
        "total_correct": len(correct_exits),
        "total_incorrect": len(incorrect_exits),
    }


def plot_exit_distribution(
    exit_stats: Dict[str, Any],
    save_path: Optional[str] = None,
) -> None:
    """Plot a histogram of early exit layer usage.

    Visualises how many samples exit at each transformer layer,
    providing an intuitive overview of the early exit controller's
    behaviour.

    Parameters
    ----------
    exit_stats : dict
        Output of :func:`collect_exit_statistics`.  Must contain
        the ``'exit_distribution'`` key (a ``Counter``).
    save_path : str or None, optional
        If provided, the figure is saved to this path (e.g.,
        ``'exit_dist.png'``).  The figure is closed after saving
        to free memory.  If ``None``, the figure is displayed
        interactively via ``plt.show()``.

    Raises
    ------
    ImportError
        If ``matplotlib`` is not installed.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend for server use
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install matplotlib"
        )

    distribution: Counter = exit_stats["exit_distribution"]

    if not distribution:
        print("No exit data to plot.")
        return

    layers = sorted(distribution.keys())
    counts = [distribution[layer] for layer in layers]
    total = sum(counts)
    percentages = [100.0 * c / total for c in counts]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        [str(l) for l in layers],
        counts,
        color="#2196F3",
        edgecolor="white",
        linewidth=0.8,
    )

    # Annotate each bar with count and percentage
    for bar, count, pct in zip(bars, counts, percentages):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(counts) * 0.01,
            f"{count}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel("Exit Layer", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_title("Early Exit Layer Distribution", fontsize=14)

    avg_exit = exit_stats.get("avg_exit_layer", None)
    savings = exit_stats.get("exit_savings_pct", None)
    subtitle_parts = []
    if avg_exit is not None:
        subtitle_parts.append(f"Avg Exit Layer: {avg_exit:.2f}")
    if savings is not None:
        subtitle_parts.append(f"Computation Saved: {savings:.1f}%")
    if subtitle_parts:
        ax.set_title(
            "Early Exit Layer Distribution\n" + " | ".join(subtitle_parts),
            fontsize=14,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
