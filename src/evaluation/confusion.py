"""
Confusion matrix computation and visualization for EfficientEuroSAT.

Provides utilities to build, visualise, and analyze confusion matrices
for the EuroSAT 10-class satellite land use classification task.  Includes
identification of the most frequently confused class pairs and
per-class accuracy bar charts.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_confusion_matrix(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_classes: int = 10,
    device: str = "cuda",
) -> np.ndarray:
    """Compute the confusion matrix over the full dataloader.

    Entry ``cm[i, j]`` counts the number of samples with true label
    ``i`` that were predicted as class ``j``.

    Parameters
    ----------
    model : nn.Module
        Trained classification model.
    dataloader : DataLoader
        Evaluation data loader yielding ``(images, labels)`` batches.
    num_classes : int, optional
        Number of classes.  Default is ``10`` (EuroSAT).
    device : str, optional
        Device for inference.  Default is ``'cuda'``.

    Returns
    -------
    np.ndarray
        Confusion matrix of shape ``(num_classes, num_classes)`` with
        integer counts.
    """
    model.eval()
    model.to(device)

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = _extract_logits(model(images))
            preds = logits.argmax(dim=1)

            preds_np = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()

            for true_label, pred_label in zip(labels_np, preds_np):
                cm[int(true_label), int(pred_label)] += 1

    return cm


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 15),
) -> None:
    """Plot the confusion matrix as a heatmap.

    Uses ``matplotlib`` and ``seaborn`` (if available) for publication-
    quality visualisation.  When ``seaborn`` is not installed, falls
    back to ``matplotlib.pyplot.imshow``.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix of shape ``(num_classes, num_classes)``.
    class_names : list of str or None, optional
        Human-readable class labels.  If ``None``, integer indices
        are used.
    save_path : str or None, optional
        If provided, the figure is saved to this path and closed.
        Otherwise the figure is displayed via ``plt.show()``.
    figsize : tuple of int, optional
        Figure size in inches.  Default is ``(15, 15)``.

    Raises
    ------
    ImportError
        If ``matplotlib`` is not installed.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install matplotlib"
        )

    num_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    # Normalise rows to get per-class rates for the colour map
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_normalised = np.divide(
        cm.astype(np.float64),
        row_sums,
        out=np.zeros_like(cm, dtype=np.float64),
        where=row_sums != 0,
    )

    fig, ax = plt.subplots(figsize=figsize)

    # Attempt to use seaborn for a polished heatmap
    try:
        import seaborn as sns

        sns.heatmap(
            cm_normalised,
            annot=cm if num_classes <= 20 else False,
            fmt="d" if num_classes <= 20 else "",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            square=True,
            linewidths=0.5,
            cbar_kws={"label": "Prediction Rate"},
            ax=ax,
        )
    except ImportError:
        # Fallback to plain matplotlib
        im = ax.imshow(cm_normalised, cmap="Blues", interpolation="nearest")
        fig.colorbar(im, ax=ax, label="Prediction Rate")

        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels(class_names, rotation=90, fontsize=7)
        ax.set_yticklabels(class_names, fontsize=7)

        # Annotate cells for small matrices
        if num_classes <= 20:
            for i in range(num_classes):
                for j in range(num_classes):
                    color = (
                        "white" if cm_normalised[i, j] > 0.5 else "black"
                    )
                    ax.text(
                        j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color=color, fontsize=7,
                    )

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)

    # Compute and display overall accuracy in subtitle
    total_correct = np.trace(cm)
    total_samples = cm.sum()
    if total_samples > 0:
        accuracy = total_correct / total_samples
        ax.set_title(
            f"Confusion Matrix  (Accuracy: {accuracy:.2%})",
            fontsize=14,
        )

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def get_most_confused_pairs(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    top_k: int = 10,
) -> List[Tuple[str, str, int, float]]:
    """Identify the most commonly confused class pairs.

    Examines off-diagonal entries of the confusion matrix to find
    pairs ``(true_class, predicted_class)`` with the highest
    misclassification counts.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix of shape ``(num_classes, num_classes)``.
    class_names : list of str or None, optional
        Human-readable class labels.  If ``None``, integer strings
        are used.
    top_k : int, optional
        Number of top confused pairs to return.  Default is ``10``.

    Returns
    -------
    list of tuple
        Each tuple contains
        ``(true_class_name, pred_class_name, count, error_rate)``:

        - ``true_class_name`` (str): True class label.
        - ``pred_class_name`` (str): Predicted class label.
        - ``count`` (int): Number of misclassifications.
        - ``error_rate`` (float): Misclassification rate relative
          to the total count of the true class.
    """
    num_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    row_sums = cm.sum(axis=1)

    # Collect all off-diagonal entries
    confused_pairs: List[Tuple[str, str, int, float]] = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                error_rate = (
                    cm[i, j] / row_sums[i] if row_sums[i] > 0 else 0.0
                )
                confused_pairs.append(
                    (class_names[i], class_names[j], int(cm[i, j]), error_rate)
                )

    # Sort by count descending, break ties by error rate descending
    confused_pairs.sort(key=lambda t: (t[2], t[3]), reverse=True)

    return confused_pairs[:top_k]


def plot_per_class_accuracy(
    per_class_acc: Dict[int, float],
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot a horizontal bar chart of per-class accuracy.

    Classes are sorted by accuracy (ascending) so that the weakest
    classes appear at the top of the chart for easy identification.

    Parameters
    ----------
    per_class_acc : dict[int, float]
        Dictionary mapping class index to accuracy in ``[0.0, 1.0]``,
        as returned by
        :func:`~efficient_eurosat.src.evaluation.metrics.compute_per_class_accuracy`.
    class_names : list of str or None, optional
        Human-readable class labels.  If ``None``, integer strings
        are used.
    save_path : str or None, optional
        If provided, the figure is saved to this path and closed.
        Otherwise the figure is displayed via ``plt.show()``.

    Raises
    ------
    ImportError
        If ``matplotlib`` is not installed.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install matplotlib"
        )

    num_classes = len(per_class_acc)
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    # Sort classes by accuracy (ascending) for visual clarity
    sorted_indices = sorted(
        per_class_acc.keys(),
        key=lambda idx: per_class_acc[idx],
    )
    sorted_names = [class_names[i] for i in sorted_indices]
    sorted_accs = [per_class_acc[i] for i in sorted_indices]

    # Compute overall mean accuracy for the reference line
    mean_acc = float(np.mean(sorted_accs)) if sorted_accs else 0.0

    fig, ax = plt.subplots(figsize=(10, max(6, num_classes * 0.35)))

    # Colour bars based on accuracy thresholds
    colors = []
    for acc in sorted_accs:
        if acc >= 0.95:
            colors.append("#4CAF50")   # green -- excellent
        elif acc >= 0.80:
            colors.append("#FF9800")   # orange -- acceptable
        else:
            colors.append("#F44336")   # red -- needs attention

    bars = ax.barh(
        range(len(sorted_names)),
        sorted_accs,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )

    # Annotate bars with percentage values
    for bar, acc in zip(bars, sorted_accs):
        width = bar.get_width()
        ax.text(
            width + 0.01,
            bar.get_y() + bar.get_height() / 2.0,
            f"{acc:.1%}",
            ha="left",
            va="center",
            fontsize=8,
        )

    # Reference line for mean accuracy
    ax.axvline(
        x=mean_acc,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        label=f"Mean: {mean_acc:.1%}",
    )

    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=8)
    ax.set_xlabel("Accuracy", fontsize=12)
    ax.set_title("Per-Class Accuracy", fontsize=14)
    ax.set_xlim(0, 1.15)
    ax.legend(loc="lower right", fontsize=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
