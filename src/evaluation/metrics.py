"""
Classification metrics for EfficientEuroSAT model evaluation.

Provides functions to compute standard classification metrics -- accuracy,
top-k accuracy, per-class accuracy, and F1 score -- on a trained model.
All functions handle the model's potential early exit behavior by accepting
whatever output format the model produces (plain tensor or tuple with exit
signal).

Designed for the EuroSAT satellite land use classification task (10 classes by default).
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_logits(
    output: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
) -> torch.Tensor:
    """Extract classification logits from model output.

    The model may return a plain tensor of logits or a tuple where the
    first element is logits and subsequent elements are auxiliary signals
    (e.g., early exit flags).  This helper normalises both cases.

    Parameters
    ----------
    output : torch.Tensor or tuple of torch.Tensor
        Raw model output.

    Returns
    -------
    torch.Tensor
        Logits tensor of shape ``(B, num_classes)``.
    """
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_accuracy(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cuda",
) -> float:
    """Compute top-1 classification accuracy over the full dataloader.

    Parameters
    ----------
    model : nn.Module
        Trained classification model.
    dataloader : DataLoader
        Evaluation data loader yielding ``(images, labels)`` batches.
    device : str, optional
        Device to run inference on.  Default is ``'cuda'``.

    Returns
    -------
    float
        Top-1 accuracy in ``[0.0, 1.0]``.
    """
    model.eval()
    model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = _extract_logits(model(images))
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0.0


def compute_top_k_accuracy(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    k: int = 5,
    device: str = "cuda",
) -> float:
    """Compute top-k classification accuracy over the full dataloader.

    A prediction is counted as correct if the true label appears among
    the k highest-scoring classes.

    Parameters
    ----------
    model : nn.Module
        Trained classification model.
    dataloader : DataLoader
        Evaluation data loader yielding ``(images, labels)`` batches.
    k : int, optional
        Number of top predictions to consider.  Default is ``5``.
    device : str, optional
        Device to run inference on.  Default is ``'cuda'``.

    Returns
    -------
    float
        Top-k accuracy in ``[0.0, 1.0]``.
    """
    model.eval()
    model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = _extract_logits(model(images))

            # Clamp k to number of classes to avoid runtime errors
            effective_k = min(k, logits.size(1))
            _, top_k_preds = logits.topk(effective_k, dim=1)

            # Check if true label is among the top-k predictions
            correct += (
                top_k_preds == labels.unsqueeze(1)
            ).any(dim=1).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0.0


def compute_per_class_accuracy(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_classes: int = 10,
    device: str = "cuda",
) -> Dict[int, float]:
    """Compute per-class classification accuracy.

    Parameters
    ----------
    model : nn.Module
        Trained classification model.
    dataloader : DataLoader
        Evaluation data loader yielding ``(images, labels)`` batches.
    num_classes : int, optional
        Number of classes in the dataset.  Default is ``10`` (EuroSAT).
    device : str, optional
        Device to run inference on.  Default is ``'cuda'``.

    Returns
    -------
    dict[int, float]
        Dictionary mapping each class index to its accuracy in
        ``[0.0, 1.0]``.  Classes with no samples in the dataloader
        receive an accuracy of ``0.0``.
    """
    model.eval()
    model.to(device)

    class_correct = np.zeros(num_classes, dtype=np.int64)
    class_total = np.zeros(num_classes, dtype=np.int64)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = _extract_logits(model(images))
            preds = logits.argmax(dim=1)

            for cls_idx in range(num_classes):
                cls_mask = labels == cls_idx
                cls_count = cls_mask.sum().item()
                if cls_count > 0:
                    class_total[cls_idx] += cls_count
                    class_correct[cls_idx] += (
                        (preds[cls_mask] == cls_idx).sum().item()
                    )

    per_class_acc: Dict[int, float] = {}
    for cls_idx in range(num_classes):
        if class_total[cls_idx] > 0:
            per_class_acc[cls_idx] = (
                class_correct[cls_idx] / class_total[cls_idx]
            )
        else:
            per_class_acc[cls_idx] = 0.0

    return per_class_acc


def compute_f1_score(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_classes: int = 10,
    device: str = "cuda",
) -> Dict[str, Union[float, List[float]]]:
    """Compute macro and per-class F1 scores.

    F1 is the harmonic mean of precision and recall.  The macro F1 is the
    unweighted mean of per-class F1 scores.

    Parameters
    ----------
    model : nn.Module
        Trained classification model.
    dataloader : DataLoader
        Evaluation data loader yielding ``(images, labels)`` batches.
    num_classes : int, optional
        Number of classes in the dataset.  Default is ``10`` (EuroSAT).
    device : str, optional
        Device to run inference on.  Default is ``'cuda'``.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``'macro_f1'`` (float): Macro-averaged F1 score.
        - ``'per_class_f1'`` (list[float]): F1 score for each class,
          indexed by class index.
    """
    model.eval()
    model.to(device)

    # Accumulate true positives, false positives, false negatives
    tp = np.zeros(num_classes, dtype=np.int64)
    fp = np.zeros(num_classes, dtype=np.int64)
    fn = np.zeros(num_classes, dtype=np.int64)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = _extract_logits(model(images))
            preds = logits.argmax(dim=1)

            preds_np = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()

            for cls_idx in range(num_classes):
                pred_cls = preds_np == cls_idx
                true_cls = labels_np == cls_idx

                tp[cls_idx] += int(np.sum(pred_cls & true_cls))
                fp[cls_idx] += int(np.sum(pred_cls & ~true_cls))
                fn[cls_idx] += int(np.sum(~pred_cls & true_cls))

    # Compute per-class F1
    per_class_f1: List[float] = []
    for cls_idx in range(num_classes):
        precision = (
            tp[cls_idx] / (tp[cls_idx] + fp[cls_idx])
            if (tp[cls_idx] + fp[cls_idx]) > 0
            else 0.0
        )
        recall = (
            tp[cls_idx] / (tp[cls_idx] + fn[cls_idx])
            if (tp[cls_idx] + fn[cls_idx]) > 0
            else 0.0
        )

        if precision + recall > 0:
            f1 = 2.0 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        per_class_f1.append(f1)

    # Macro F1: unweighted mean of per-class F1 scores
    macro_f1 = float(np.mean(per_class_f1)) if per_class_f1 else 0.0

    return {
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
    }


def compute_all_metrics(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_classes: int = 10,
    device: str = "cuda",
) -> Dict[str, object]:
    """Compute a comprehensive set of classification metrics.

    This is a convenience wrapper that runs all individual metric
    functions in a single pass over the data to avoid redundant
    computation.  It collects predictions and labels once, then
    derives all metrics from the accumulated arrays.

    Parameters
    ----------
    model : nn.Module
        Trained classification model.
    dataloader : DataLoader
        Evaluation data loader yielding ``(images, labels)`` batches.
    num_classes : int, optional
        Number of classes in the dataset.  Default is ``10`` (EuroSAT).
    device : str, optional
        Device to run inference on.  Default is ``'cuda'``.

    Returns
    -------
    dict
        Dictionary containing:

        - ``'top1_accuracy'`` (float): Overall top-1 accuracy.
        - ``'top5_accuracy'`` (float): Overall top-5 accuracy.
        - ``'per_class_accuracy'`` (dict[int, float]): Per-class accuracy.
        - ``'macro_f1'`` (float): Macro-averaged F1 score.
        - ``'per_class_f1'`` (list[float]): Per-class F1 scores.
        - ``'num_samples'`` (int): Total number of evaluated samples.
    """
    model.eval()
    model.to(device)

    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_top5_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = _extract_logits(model(images))
            preds = logits.argmax(dim=1)

            # Top-5 accuracy
            effective_k = min(5, logits.size(1))
            _, top_k_preds = logits.topk(effective_k, dim=1)
            all_top5_correct += (
                top_k_preds == labels.unsqueeze(1)
            ).any(dim=1).sum().item()

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            total += labels.size(0)

    # Concatenate all predictions and labels
    preds_arr = np.concatenate(all_preds)
    labels_arr = np.concatenate(all_labels)

    # --- Top-1 accuracy ---
    top1_accuracy = float(np.mean(preds_arr == labels_arr))

    # --- Top-5 accuracy ---
    top5_accuracy = all_top5_correct / total if total > 0 else 0.0

    # --- Per-class accuracy ---
    per_class_acc: Dict[int, float] = {}
    for cls_idx in range(num_classes):
        cls_mask = labels_arr == cls_idx
        cls_count = cls_mask.sum()
        if cls_count > 0:
            per_class_acc[cls_idx] = float(
                np.mean(preds_arr[cls_mask] == cls_idx)
            )
        else:
            per_class_acc[cls_idx] = 0.0

    # --- F1 scores ---
    tp = np.zeros(num_classes, dtype=np.int64)
    fp = np.zeros(num_classes, dtype=np.int64)
    fn = np.zeros(num_classes, dtype=np.int64)

    for cls_idx in range(num_classes):
        pred_cls = preds_arr == cls_idx
        true_cls = labels_arr == cls_idx
        tp[cls_idx] = int(np.sum(pred_cls & true_cls))
        fp[cls_idx] = int(np.sum(pred_cls & ~true_cls))
        fn[cls_idx] = int(np.sum(~pred_cls & true_cls))

    per_class_f1: List[float] = []
    for cls_idx in range(num_classes):
        precision = (
            tp[cls_idx] / (tp[cls_idx] + fp[cls_idx])
            if (tp[cls_idx] + fp[cls_idx]) > 0
            else 0.0
        )
        recall = (
            tp[cls_idx] / (tp[cls_idx] + fn[cls_idx])
            if (tp[cls_idx] + fn[cls_idx]) > 0
            else 0.0
        )
        if precision + recall > 0:
            f1 = 2.0 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        per_class_f1.append(f1)

    macro_f1 = float(np.mean(per_class_f1)) if per_class_f1 else 0.0

    return {
        "top1_accuracy": top1_accuracy,
        "top5_accuracy": top5_accuracy,
        "per_class_accuracy": per_class_acc,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "num_samples": total,
    }
