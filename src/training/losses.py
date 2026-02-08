"""Loss functions for EfficientEuroSAT training.

This module provides two loss functions tailored for training satellite image
recognition models with the EfficientEuroSAT architecture:

1. LabelSmoothingCrossEntropy
   Standard cross-entropy augmented with label smoothing, which replaces
   hard one-hot targets with a mixture of the ground-truth distribution
   and a uniform distribution.  This acts as a regulariser and improves
   calibration, especially on small-to-medium classification datasets
   such as EuroSAT.

2. EarlyExitLoss
   A composite loss that combines the main classifier head loss with
   weighted auxiliary losses from early-exit branches.  This encourages
   intermediate representations to be discriminative, supporting
   confidence-based early exit at inference time.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "LabelSmoothingCrossEntropy",
    "EarlyExitLoss",
    "UCATLoss",
    "CombinedLoss",
    "AleatoricConsistencyLoss",
    "AleatoricBlurLoss",
    "EpistemicLoss",
    "DecomposedCombinedLoss",
]


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy loss with label smoothing.

    Given a smoothing factor ``epsilon``, the target distribution for a
    ground-truth class ``y`` is:

        q(k) = (1 - epsilon) * delta(k, y) + epsilon / C

    where ``C`` is the number of classes.  This prevents the model from
    becoming overconfident and improves generalisation.

    Parameters
    ----------
    smoothing : float, optional
        Label smoothing factor in ``[0, 1)``.  ``0`` recovers standard
        cross-entropy.  Default is ``0.1``.
    weight : torch.Tensor or None, optional
        A 1-D tensor of shape ``(C,)`` assigning a weight to each class.
        Useful for handling class imbalance (e.g., in EuroSAT where some
        land use categories have far fewer samples).  Default is ``None``
        (uniform weighting).

    Examples
    --------
    >>> criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    >>> pred = torch.randn(8, 10)   # batch=8, 10 EuroSAT classes
    >>> target = torch.randint(0, 10, (8,))
    >>> loss = criterion(pred, target)
    >>> loss.shape
    torch.Size([])
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError(
                f"smoothing must be in [0, 1), got {smoothing}"
            )
        self.smoothing = smoothing
        # Store weight as a buffer so it moves with .to(device) automatically.
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute label-smoothed cross-entropy loss.

        Parameters
        ----------
        pred : torch.Tensor
            Raw logits of shape ``(N, C)`` where ``N`` is the batch size
            and ``C`` is the number of classes.
        target : torch.Tensor
            Ground-truth class indices of shape ``(N,)`` with values in
            ``[0, C)``.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        num_classes = pred.size(-1)

        # Log-softmax for numerical stability.
        log_probs = F.log_softmax(pred, dim=-1)

        # Construct smoothed target distribution.
        # Start with the uniform component: epsilon / C for every class.
        smooth_targets = torch.full_like(log_probs, self.smoothing / num_classes)

        # Add the concentrated mass on the ground-truth class.
        smooth_targets.scatter_(
            dim=1,
            index=target.unsqueeze(1),
            value=1.0 - self.smoothing + self.smoothing / num_classes,
        )

        # Element-wise NLL: -q(k) * log p(k), then sum over classes.
        per_sample_loss = (-smooth_targets * log_probs).sum(dim=-1)

        # Apply per-class weights if provided.
        if self.weight is not None:
            sample_weights = self.weight[target]
            per_sample_loss = per_sample_loss * sample_weights
            return per_sample_loss.sum() / sample_weights.sum()

        return per_sample_loss.mean()

    def extra_repr(self) -> str:
        return (
            f"smoothing={self.smoothing}, "
            f"weighted={self.weight is not None}"
        )


class EarlyExitLoss(nn.Module):
    """Composite loss incorporating auxiliary early-exit branches.

    The total loss is the sum of the main head loss and a weighted
    average of the auxiliary losses from each early-exit branch:

        L = L_main + exit_weight * (1 / num_exits) * sum(L_exit_i)

    This trains intermediate classifier heads so they produce useful
    predictions for the ``EarlyExitController`` to evaluate confidence
    at inference time.

    Parameters
    ----------
    num_exits : int
        Number of auxiliary early-exit branches.
    base_loss_fn : nn.Module
        Loss function to apply to both the main and auxiliary logits
        (e.g., ``LabelSmoothingCrossEntropy`` or ``nn.CrossEntropyLoss``).
    exit_weight : float, optional
        Total weight allocated to all auxiliary losses combined.
        Default is ``0.5``.

    Examples
    --------
    >>> base_loss = LabelSmoothingCrossEntropy(smoothing=0.1)
    >>> criterion = EarlyExitLoss(num_exits=3, base_loss_fn=base_loss)
    >>> main_logits = torch.randn(8, 10)
    >>> exit_logits = [torch.randn(8, 10) for _ in range(3)]
    >>> target = torch.randint(0, 10, (8,))
    >>> loss = criterion(main_logits, exit_logits, target)
    >>> loss.shape
    torch.Size([])
    """

    def __init__(
        self,
        num_exits: int,
        base_loss_fn: nn.Module,
        exit_weight: float = 0.5,
    ) -> None:
        super().__init__()
        if num_exits < 1:
            raise ValueError(
                f"num_exits must be >= 1, got {num_exits}"
            )
        if exit_weight < 0.0:
            raise ValueError(
                f"exit_weight must be non-negative, got {exit_weight}"
            )
        self.num_exits = num_exits
        self.base_loss_fn = base_loss_fn
        self.exit_weight = exit_weight

    def forward(
        self,
        main_logits: torch.Tensor,
        exit_logits_list: List[torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the combined main + auxiliary exit loss.

        Parameters
        ----------
        main_logits : torch.Tensor
            Logits from the main (final) classifier head, shape ``(N, C)``.
        exit_logits_list : list of torch.Tensor
            List of length ``num_exits``, each element a tensor of shape
            ``(N, C)`` containing logits from an auxiliary exit branch.
        target : torch.Tensor
            Ground-truth class indices of shape ``(N,)``.

        Returns
        -------
        torch.Tensor
            Scalar combined loss.

        Raises
        ------
        ValueError
            If the length of ``exit_logits_list`` does not match
            ``num_exits``.
        """
        if len(exit_logits_list) != self.num_exits:
            raise ValueError(
                f"Expected {self.num_exits} exit logit tensors, "
                f"got {len(exit_logits_list)}"
            )

        # Main head loss.
        main_loss = self.base_loss_fn(main_logits, target)

        # Weighted sum of auxiliary exit losses.
        aux_loss = torch.tensor(0.0, device=main_logits.device, dtype=main_logits.dtype)
        weight_per_exit = self.exit_weight / self.num_exits

        for exit_logits in exit_logits_list:
            aux_loss = aux_loss + weight_per_exit * self.base_loss_fn(exit_logits, target)

        return main_loss + aux_loss

    def extra_repr(self) -> str:
        return (
            f"num_exits={self.num_exits}, "
            f"exit_weight={self.exit_weight}"
        )


class UCATLoss(nn.Module):
    """Uncertainty-Calibrated Attention Temperature (UCAT) loss.

    Encourages learned attention temperatures to positively correlate
    with prediction uncertainty (entropy).  The intuition is that the
    model should use **soft** (high-temperature) attention when it is
    uncertain and **sharp** (low-temperature) attention when confident.

    The loss is the *negative* Pearson correlation between the per-sample
    mean attention temperature and the prediction entropy, scaled by
    ``lambda_ucat``.  Minimising this loss therefore maximises the
    desired positive correlation.

    Parameters
    ----------
    lambda_ucat : float, optional
        Weight of the UCAT loss term.  Default is ``0.1``.
    """

    def __init__(self, lambda_ucat: float = 0.1) -> None:
        super().__init__()
        self.lambda_ucat = lambda_ucat

    @staticmethod
    def compute_prediction_entropy(logits: torch.Tensor) -> torch.Tensor:
        """Compute Shannon entropy of the softmax distribution.

        Parameters
        ----------
        logits : torch.Tensor
            Raw logits of shape ``(N, C)``.

        Returns
        -------
        torch.Tensor
            Per-sample entropy of shape ``(N,)``.
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return -torch.sum(probs * log_probs, dim=-1)

    @staticmethod
    def compute_pearson_correlation(
        x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Compute Pearson correlation between two 1-D tensors.

        Returns ``0`` when either vector has zero variance to avoid
        division-by-zero.
        """
        x_c = x - x.mean()
        y_c = y - y.mean()
        num = torch.sum(x_c * y_c)
        den = torch.sqrt(torch.sum(x_c ** 2) * torch.sum(y_c ** 2) + 1e-8)
        return num / den

    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the UCAT loss.

        Parameters
        ----------
        logits : torch.Tensor
            Model predictions of shape ``(N, C)``.
        temperatures : torch.Tensor
            Mean attention temperature per sample of shape ``(N,)``.

        Returns
        -------
        torch.Tensor
            Scalar UCAT loss (negative Pearson correlation scaled by
            ``lambda_ucat``).
        """
        entropy = self.compute_prediction_entropy(logits)
        correlation = self.compute_pearson_correlation(temperatures, entropy)
        return -self.lambda_ucat * correlation

    def extra_repr(self) -> str:
        return f"lambda_ucat={self.lambda_ucat}"


class CombinedLoss(nn.Module):
    """Task loss combined with the UCAT auxiliary objective.

    Computes::

        L_total = L_task + L_ucat

    where ``L_task`` is label-smoothed cross-entropy and ``L_ucat`` is
    the uncertainty-calibrated attention temperature loss.

    Parameters
    ----------
    lambda_ucat : float, optional
        Weight for the UCAT term.  Set to ``0`` to disable.
        Default is ``0.1``.
    label_smoothing : float, optional
        Label smoothing for the task loss.  Default is ``0.1``.
    weight : torch.Tensor or None, optional
        Per-class weights for the task loss.  Default is ``None``.
    """

    def __init__(
        self,
        lambda_ucat: float = 0.1,
        label_smoothing: float = 0.1,
        weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.task_loss = LabelSmoothingCrossEntropy(
            smoothing=label_smoothing, weight=weight
        )
        self.ucat_loss = UCATLoss(lambda_ucat=lambda_ucat)
        self.lambda_ucat = lambda_ucat

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        temperatures: Optional[torch.Tensor] = None,
    ):
        """Compute combined loss.

        Parameters
        ----------
        logits : torch.Tensor
            Shape ``(N, C)``.
        labels : torch.Tensor
            Shape ``(N,)``.
        temperatures : torch.Tensor or None
            Shape ``(N,)``.  If ``None`` or ``lambda_ucat == 0``,
            only the task loss is computed.

        Returns
        -------
        tuple of (total_loss, task_loss, ucat_loss)
            All scalar tensors.
        """
        l_task = self.task_loss(logits, labels)

        if temperatures is not None and self.lambda_ucat > 0:
            l_ucat = self.ucat_loss(logits, temperatures)
        else:
            l_ucat = torch.tensor(0.0, device=logits.device)

        return l_task + l_ucat, l_task, l_ucat

    def extra_repr(self) -> str:
        return f"lambda_ucat={self.lambda_ucat}"


# ---------------------------------------------------------------------------
# Decomposed UCAT losses
# ---------------------------------------------------------------------------

class AleatoricConsistencyLoss(nn.Module):
    """Aleatoric consistency: tau_a should be similar across augmentations of the same image."""

    def forward(
        self,
        tau_a_original: torch.Tensor,
        tau_a_augmented: torch.Tensor,
    ) -> torch.Tensor:
        """Mean squared difference of tau_a between original and augmented."""
        return ((tau_a_original - tau_a_augmented) ** 2).mean()


class AleatoricBlurLoss(nn.Module):
    """Aleatoric blur correlation: tau_a should increase with image blur."""

    def forward(
        self,
        tau_a_values: torch.Tensor,
        blur_levels: torch.Tensor,
    ) -> torch.Tensor:
        """Negative Pearson correlation between tau_a and blur level."""
        return -UCATLoss.compute_pearson_correlation(tau_a_values, blur_levels)


class EpistemicLoss(nn.Module):
    """Epistemic loss: tau_e correlates with class rarity and decays over training."""

    def __init__(self, class_rarity: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        if class_rarity is not None:
            self.register_buffer("class_rarity", class_rarity)
        else:
            self.class_rarity = None

    def forward(
        self,
        tau_e: torch.Tensor,
        predicted_classes: torch.Tensor,
        training_progress: float = 0.0,
    ) -> torch.Tensor:
        """Rarity correlation + training-progress decay."""
        loss = torch.tensor(0.0, device=predicted_classes.device)

        if self.class_rarity is not None and tau_e.numel() > 1:
            rarity_scores = self.class_rarity[predicted_classes]
            tau_e_flat = tau_e.expand(predicted_classes.shape[0]) if tau_e.dim() == 0 else tau_e
            if tau_e_flat.numel() > 1:
                loss = loss - UCATLoss.compute_pearson_correlation(tau_e_flat, rarity_scores)

        tau_e_mean = tau_e.mean() if tau_e.dim() > 0 else tau_e
        loss = loss + training_progress * 0.1 * tau_e_mean

        return loss


class DecomposedCombinedLoss(nn.Module):
    """Full decomposed UCAT loss: L_task + lambda_ucat*L_ucat + lambda_a*L_a + lambda_e*L_e."""

    def __init__(
        self,
        lambda_ucat: float = 0.1,
        lambda_aleatoric: float = 0.05,
        lambda_epistemic: float = 0.05,
        label_smoothing: float = 0.1,
        weight: Optional[torch.Tensor] = None,
        class_rarity: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.task_loss = LabelSmoothingCrossEntropy(smoothing=label_smoothing, weight=weight)
        self.ucat_loss = UCATLoss(lambda_ucat=1.0)
        self.aleatoric_consistency = AleatoricConsistencyLoss()
        self.aleatoric_blur = AleatoricBlurLoss()
        self.epistemic_loss = EpistemicLoss(class_rarity=class_rarity)

        self.lambda_ucat = lambda_ucat
        self.lambda_aleatoric = lambda_aleatoric
        self.lambda_epistemic = lambda_epistemic

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        temperatures: Optional[torch.Tensor] = None,
        tau_a_original: Optional[torch.Tensor] = None,
        tau_a_augmented: Optional[torch.Tensor] = None,
        tau_e: Optional[torch.Tensor] = None,
        training_progress: float = 0.0,
        tau_a_blur_values: Optional[torch.Tensor] = None,
        blur_levels: Optional[torch.Tensor] = None,
    ):
        """Returns (total, task, ucat, aleatoric, epistemic)."""
        device = logits.device
        l_task = self.task_loss(logits, labels)

        l_ucat = torch.tensor(0.0, device=device)
        if temperatures is not None and self.lambda_ucat > 0:
            l_ucat = self.ucat_loss(logits, temperatures)

        l_aleatoric = torch.tensor(0.0, device=device)
        if tau_a_original is not None and tau_a_augmented is not None:
            l_aleatoric = self.aleatoric_consistency(tau_a_original, tau_a_augmented)
        if tau_a_blur_values is not None and blur_levels is not None:
            l_aleatoric = l_aleatoric + self.aleatoric_blur(tau_a_blur_values, blur_levels)

        l_epistemic = torch.tensor(0.0, device=device)
        if tau_e is not None:
            _, predicted = logits.max(dim=1)
            l_epistemic = self.epistemic_loss(tau_e, predicted, training_progress)

        total = (
            l_task
            + self.lambda_ucat * l_ucat
            + self.lambda_aleatoric * l_aleatoric
            + self.lambda_epistemic * l_epistemic
        )

        return total, l_task, l_ucat, l_aleatoric, l_epistemic
