"""Learning rate and temperature schedulers for EfficientEuroSAT training.

This module provides two schedulers:

1. CosineAnnealingWithWarmup
   A learning rate scheduler that combines a linear warmup phase with
   cosine annealing decay.  The warmup prevents early training instability
   when using large learning rates with pre-trained backbones.

2. TemperatureAnnealingScheduler
   A step-based wrapper around ``TemperatureScheduler`` from the
   attention modifications module.  It maintains an internal step counter
   to automatically compute training progress and return the current
   temperature multiplier, decoupling the annealing logic from the
   training loop.
"""

from __future__ import annotations

import math
from typing import Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from ..models.attention_modifications import TemperatureScheduler

__all__ = [
    "CosineAnnealingWithWarmup",
    "TemperatureAnnealingScheduler",
]


class CosineAnnealingWithWarmup(LambdaLR):
    """Learning rate scheduler: linear warmup followed by cosine decay.

    During the warmup phase (epochs ``0`` to ``warmup_epochs - 1``), the
    learning rate increases linearly from a small fraction of the base LR
    to the full base LR.  After warmup, the LR follows a cosine curve
    from the base LR down to ``min_lr``.

    This schedule is commonly used when fine-tuning vision transformers
    and helps stabilise early training while still allowing thorough
    convergence later.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Wrapped optimizer.
    warmup_epochs : int
        Number of epochs for the linear warmup phase.
    total_epochs : int
        Total number of training epochs (warmup + cosine decay).
    min_lr : float, optional
        Minimum learning rate at the end of cosine decay, expressed as
        an absolute value.  The LR multiplier will be computed such that
        ``base_lr * multiplier >= min_lr``.  Default is ``1e-6``.

    Examples
    --------
    >>> import torch
    >>> model = torch.nn.Linear(10, 2)
    >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    >>> scheduler = CosineAnnealingWithWarmup(
    ...     optimizer, warmup_epochs=5, total_epochs=100, min_lr=1e-6
    ... )
    >>> # LR at epoch 0 is a small fraction of base LR
    >>> # LR at epoch 5 equals base LR
    >>> # LR at epoch 100 approaches min_lr
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
    ) -> None:
        if warmup_epochs < 0:
            raise ValueError(
                f"warmup_epochs must be non-negative, got {warmup_epochs}"
            )
        if total_epochs < warmup_epochs:
            raise ValueError(
                f"total_epochs ({total_epochs}) must be >= "
                f"warmup_epochs ({warmup_epochs})"
            )
        if min_lr < 0.0:
            raise ValueError(f"min_lr must be non-negative, got {min_lr}")

        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr

        # Retrieve the base learning rate from the first param group.
        # This is needed to compute the minimum multiplier.
        self._base_lr = optimizer.defaults["lr"]

        # Compute the floor multiplier so that base_lr * multiplier = min_lr.
        # Clamp to avoid negative multipliers if min_lr > base_lr.
        if self._base_lr > 0:
            self._min_mult = max(min_lr / self._base_lr, 0.0)
        else:
            self._min_mult = 0.0

        # Build the lambda and pass to parent.  The lambda captures
        # instance attributes via closure.
        super().__init__(optimizer, lr_lambda=self._lr_lambda)

    def _lr_lambda(self, epoch: int) -> float:
        """Compute the LR multiplier for a given epoch.

        Parameters
        ----------
        epoch : int
            Current epoch index (zero-based).

        Returns
        -------
        float
            Multiplicative factor applied to the base learning rate.
        """
        # --- Warmup phase ---
        if epoch < self.warmup_epochs:
            # Linear ramp from a small fraction to 1.0.
            # At epoch 0 the multiplier is 1 / (warmup_epochs + 1),
            # ensuring it is never zero.
            return (epoch + 1) / (self.warmup_epochs + 1)

        # --- Cosine decay phase ---
        if self.total_epochs <= self.warmup_epochs:
            # Edge case: no cosine phase.
            return 1.0

        cosine_epochs = self.total_epochs - self.warmup_epochs
        progress = (epoch - self.warmup_epochs) / cosine_epochs

        # Cosine multiplier decays from 1.0 to min_mult.
        cosine_mult = self._min_mult + 0.5 * (1.0 - self._min_mult) * (
            1.0 + math.cos(math.pi * progress)
        )
        return cosine_mult

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"warmup_epochs={self.warmup_epochs}, "
            f"total_epochs={self.total_epochs}, "
            f"min_lr={self.min_lr})"
        )


class TemperatureAnnealingScheduler:
    """Step-based wrapper for attention temperature annealing.

    Maintains an internal step counter and wraps
    ``TemperatureScheduler.get_multiplier`` so that the training loop can
    simply call ``step()`` each iteration to advance the schedule and
    retrieve the current temperature multiplier.

    Parameters
    ----------
    tau_max_mult : float, optional
        Temperature multiplier at the start of training.
        Default is ``1.5``.
    tau_min_mult : float, optional
        Temperature multiplier at the end of training.
        Default is ``1.0``.
    power : float, optional
        Exponent controlling the annealing shape.
        Default is ``2``.
    total_steps : int or None, optional
        Total number of training steps.  If ``None``, the scheduler
        returns ``tau_max_mult`` until ``total_steps`` is set via
        ``set_total_steps``.  Default is ``None``.

    Examples
    --------
    >>> sched = TemperatureAnnealingScheduler(total_steps=1000)
    >>> mult_start = sched.step()   # step 0 -> near tau_max_mult
    >>> for _ in range(999):
    ...     mult = sched.step()
    >>> abs(mult - 1.0) < 0.01     # near tau_min_mult at the end
    True
    """

    def __init__(
        self,
        tau_max_mult: float = 1.5,
        tau_min_mult: float = 1.0,
        power: float = 2.0,
        total_steps: Optional[int] = None,
    ) -> None:
        self._scheduler = TemperatureScheduler(
            tau_max_mult=tau_max_mult,
            tau_min_mult=tau_min_mult,
            power=power,
        )
        self._total_steps = total_steps
        self._current_step: int = 0

    @property
    def total_steps(self) -> Optional[int]:
        """Total number of training steps, or ``None`` if not yet set."""
        return self._total_steps

    def set_total_steps(self, total_steps: int) -> None:
        """Set or update the total number of training steps.

        Parameters
        ----------
        total_steps : int
            Must be a positive integer.
        """
        if total_steps < 1:
            raise ValueError(
                f"total_steps must be >= 1, got {total_steps}"
            )
        self._total_steps = total_steps

    def get_progress(self) -> float:
        """Return current training progress as a float in ``[0, 1]``.

        Returns
        -------
        float
            ``0.0`` at the start of training, ``1.0`` at the end.
            If ``total_steps`` has not been set, returns ``0.0``.
        """
        if self._total_steps is None or self._total_steps <= 0:
            return 0.0
        return min(self._current_step / self._total_steps, 1.0)

    def step(self) -> float:
        """Advance the schedule by one step and return the multiplier.

        Returns
        -------
        float
            Current temperature multiplier based on training progress.
        """
        progress = self.get_progress()
        multiplier = self._scheduler.get_multiplier(progress)
        self._current_step += 1
        return multiplier

    def get_current_multiplier(self) -> float:
        """Return the multiplier at the current step without advancing.

        Returns
        -------
        float
            Current temperature multiplier.
        """
        progress = self.get_progress()
        return self._scheduler.get_multiplier(progress)

    def reset(self) -> None:
        """Reset the step counter to zero."""
        self._current_step = 0

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"tau_max_mult={self._scheduler.tau_max_mult}, "
            f"tau_min_mult={self._scheduler.tau_min_mult}, "
            f"power={self._scheduler.power}, "
            f"total_steps={self._total_steps}, "
            f"current_step={self._current_step})"
        )
