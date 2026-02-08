"""Training callbacks for EfficientEuroSAT.

This module provides three callback utilities that plug into the
training loop to monitor progress, prevent over-fitting, and persist
the best model weights:

1. EarlyStopping
   Halts training when a monitored validation metric has stopped
   improving for a specified number of epochs (patience).

2. ModelCheckpoint
   Saves model state to disk whenever a tracked metric improves,
   ensuring the best-performing weights are always recoverable.

3. MetricTracker
   Accumulates metric values across training and provides history
   retrieval and best-value queries.
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

__all__ = [
    "EarlyStopping",
    "ModelCheckpoint",
    "MetricTracker",
]


class EarlyStopping:
    """Stop training when a validation metric stops improving.

    After ``patience`` consecutive epochs without an improvement of at
    least ``min_delta``, the callback signals that training should stop.

    Parameters
    ----------
    patience : int, optional
        Number of epochs with no improvement after which training is
        stopped.  Default is ``10``.
    min_delta : float, optional
        Minimum change in the monitored metric to qualify as an
        improvement.  Default is ``0.001``.
    mode : str, optional
        One of ``'max'`` or ``'min'``.  ``'max'`` means higher is
        better (e.g., accuracy); ``'min'`` means lower is better
        (e.g., loss).  Default is ``'max'``.

    Examples
    --------
    >>> es = EarlyStopping(patience=3, mode='max')
    >>> es(0.80)  # new best
    False
    >>> es(0.79)  # no improvement
    False
    >>> es(0.78)  # no improvement
    False
    >>> es(0.77)  # patience exhausted
    True
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = "max",
    ) -> None:
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got '{mode}'")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self._counter: int = 0
        self._best_value: Optional[float] = None

    def _is_improvement(self, metric: float) -> bool:
        """Check whether ``metric`` is an improvement over the best.

        Parameters
        ----------
        metric : float
            Current metric value.

        Returns
        -------
        bool
            ``True`` if the metric improved by at least ``min_delta``.
        """
        if self._best_value is None:
            return True
        if self.mode == "max":
            return metric > self._best_value + self.min_delta
        return metric < self._best_value - self.min_delta

    def __call__(self, metric: float) -> bool:
        """Evaluate whether training should stop.

        Parameters
        ----------
        metric : float
            Current value of the monitored metric.

        Returns
        -------
        bool
            ``True`` if training should stop (patience exhausted);
            ``False`` otherwise.
        """
        if self._is_improvement(metric):
            self._best_value = metric
            self._counter = 0
            return False

        self._counter += 1
        return self._counter >= self.patience

    def reset(self) -> None:
        """Reset the internal state (counter and best value)."""
        self._counter = 0
        self._best_value = None

    @property
    def best_value(self) -> Optional[float]:
        """Best metric value observed so far, or ``None`` if no calls."""
        return self._best_value

    @property
    def epochs_without_improvement(self) -> int:
        """Number of consecutive epochs without improvement."""
        return self._counter

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"patience={self.patience}, "
            f"min_delta={self.min_delta}, "
            f"mode='{self.mode}', "
            f"counter={self._counter})"
        )


class ModelCheckpoint:
    """Save the model whenever a monitored metric improves.

    The checkpoint is saved as a ``.pt`` file containing the model's
    ``state_dict``, along with the epoch number and metric value.  Only
    the single best checkpoint is kept on disk (the previous best is
    overwritten).

    Parameters
    ----------
    save_dir : str or Path
        Directory where checkpoint files are written.  Created
        automatically if it does not exist.
    metric_name : str, optional
        Name of the metric used in the filename and log messages.
        Default is ``'val_acc'``.
    mode : str, optional
        One of ``'max'`` or ``'min'``.  Determines whether a higher or
        lower metric value is considered an improvement.
        Default is ``'max'``.

    Examples
    --------
    >>> ckpt = ModelCheckpoint(save_dir='/tmp/ckpts', metric_name='val_acc')
    >>> model = torch.nn.Linear(10, 2)
    >>> ckpt(model, metric=0.90, epoch=5)  # saves (first call)
    >>> ckpt(model, metric=0.85, epoch=6)  # does not save
    >>> ckpt(model, metric=0.92, epoch=7)  # saves (new best)
    """

    def __init__(
        self,
        save_dir: str,
        metric_name: str = "val_acc",
        mode: str = "max",
    ) -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got '{mode}'")

        self.save_dir = Path(save_dir)
        self.metric_name = metric_name
        self.mode = mode

        self._best_value: Optional[float] = None
        self._best_epoch: Optional[int] = None

        # Ensure the save directory exists.
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _is_improvement(self, metric: float) -> bool:
        """Check whether ``metric`` improves over the current best."""
        if self._best_value is None:
            return True
        if self.mode == "max":
            return metric > self._best_value
        return metric < self._best_value

    def __call__(
        self,
        model: nn.Module,
        metric: float,
        epoch: int,
        model_config: Optional[dict] = None,
    ) -> None:
        """Conditionally save the model if the metric improved.

        Parameters
        ----------
        model : nn.Module
            The model whose ``state_dict`` will be saved.
        metric : float
            Current value of the monitored metric.
        epoch : int
            Current epoch number (used in the filename and metadata).
        model_config : dict or None, optional
            Model configuration to store in the checkpoint for
            reconstruction at load time.
        """
        if not self._is_improvement(metric):
            return

        self._best_value = metric
        self._best_epoch = epoch

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            self.metric_name: metric,
        }
        if model_config is not None:
            checkpoint["model_config"] = model_config

        save_path = self.save_dir / f"best_model_{self.metric_name}.pt"
        torch.save(checkpoint, save_path)
        print(
            f"[ModelCheckpoint] Saved best model at epoch {epoch} "
            f"({self.metric_name}={metric:.4f}) -> {save_path}"
        )

    @property
    def best_value(self) -> Optional[float]:
        """Best metric value observed so far."""
        return self._best_value

    @property
    def best_epoch(self) -> Optional[int]:
        """Epoch at which the best metric was observed."""
        return self._best_epoch

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"save_dir='{self.save_dir}', "
            f"metric_name='{self.metric_name}', "
            f"mode='{self.mode}', "
            f"best={self._best_value})"
        )


class MetricTracker:
    """Track and query training metrics over time.

    Stores a running history for each named metric, supporting
    retrieval of the full history, the best value, and summary
    statistics.

    Examples
    --------
    >>> tracker = MetricTracker()
    >>> tracker.update('val_acc', 0.85)
    >>> tracker.update('val_acc', 0.90)
    >>> tracker.update('val_acc', 0.88)
    >>> tracker.get_best('val_acc', mode='max')
    0.9
    >>> tracker.get_history('val_acc')
    [0.85, 0.9, 0.88]
    """

    def __init__(self) -> None:
        self._history: Dict[str, List[float]] = defaultdict(list)

    def update(self, metric_name: str, value: float) -> None:
        """Record a new value for a named metric.

        Parameters
        ----------
        metric_name : str
            Name of the metric (e.g., ``'train_loss'``, ``'val_acc'``).
        value : float
            The metric value to record.
        """
        self._history[metric_name].append(value)

    def get_history(self, metric_name: str) -> List[float]:
        """Return the full value history for a metric.

        Parameters
        ----------
        metric_name : str
            Name of the metric.

        Returns
        -------
        list of float
            All recorded values in chronological order.  Returns an
            empty list if the metric has never been recorded.
        """
        return list(self._history[metric_name])

    def get_best(self, metric_name: str, mode: str = "max") -> Optional[float]:
        """Return the best value recorded for a metric.

        Parameters
        ----------
        metric_name : str
            Name of the metric.
        mode : str, optional
            ``'max'`` returns the highest value; ``'min'`` returns the
            lowest.  Default is ``'max'``.

        Returns
        -------
        float or None
            Best value, or ``None`` if no values have been recorded.
        """
        history = self._history.get(metric_name)
        if not history:
            return None
        if mode == "max":
            return max(history)
        if mode == "min":
            return min(history)
        raise ValueError(f"mode must be 'max' or 'min', got '{mode}'")

    def get_last(self, metric_name: str) -> Optional[float]:
        """Return the most recently recorded value for a metric.

        Parameters
        ----------
        metric_name : str
            Name of the metric.

        Returns
        -------
        float or None
            Last recorded value, or ``None`` if no values exist.
        """
        history = self._history.get(metric_name)
        if not history:
            return None
        return history[-1]

    @property
    def metric_names(self) -> List[str]:
        """List of all metric names that have been recorded."""
        return list(self._history.keys())

    def reset(self) -> None:
        """Clear all recorded metrics."""
        self._history.clear()

    def __repr__(self) -> str:
        summary = {name: len(vals) for name, vals in self._history.items()}
        return f"{self.__class__.__name__}(metrics={summary})"
