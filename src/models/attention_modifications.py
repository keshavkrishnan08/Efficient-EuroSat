"""Attention mechanism modifications for EfficientEuroSAT.

This module implements five targeted modifications to standard multi-head
self-attention, each designed to improve efficiency, stability, or
representational capacity in satellite land use classification models.

Modifications
-------------
1. LearnableTemperature
    Per-head learned temperature replacing the fixed sqrt(d_k) scaling.
2. EarlyExitController
    Confidence-based early exit during inference.
3. LearnedHeadDropout
    Per-head learned dropout rates for attention weights.
4. LearnedResidualWeight
    Learned interpolation between attention output and residual stream.
5. TemperatureScheduler
    Training-time annealing schedule for temperature multipliers.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "LearnableTemperature",
    "TemperaturePredictor",
    "EarlyExitController",
    "LearnedHeadDropout",
    "LearnedResidualWeight",
    "TemperatureScheduler",
]


class LearnableTemperature(nn.Module):
    """Per-head learned temperature for scaled dot-product attention.

    Standard attention divides logits by a fixed temperature equal to
    sqrt(d_k).  This module replaces that constant with a learned
    parameter for each head, allowing the model to sharpen or soften
    individual attention distributions independently.

    Stability is ensured by parameterising the temperature through
    ``softplus`` and adding a minimum floor ``tau_min``, so the
    effective temperature is always strictly positive:

        tau = softplus(raw_tau) + tau_min

    Parameters
    ----------
    num_heads : int
        Number of attention heads.
    tau_min : float, optional
        Minimum temperature floor to prevent division-by-zero or
        excessively sharp distributions.  Default is ``0.1``.

    Examples
    --------
    >>> lt = LearnableTemperature(num_heads=8, tau_min=0.1)
    >>> tau = lt()          # shape: (8,)
    >>> tau.min().item() >= 0.1
    True
    """

    def __init__(self, num_heads: int, tau_min: float = 0.1) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.tau_min = tau_min

        # Initialise raw parameters to zero so that
        # softplus(0) + tau_min ~ 0.693 + tau_min, a sensible starting
        # temperature close to 1.
        self.raw_tau = nn.Parameter(torch.zeros(num_heads))
        self.last_tau = None  # Cache for UCAT loss

    def forward(self) -> torch.Tensor:
        """Compute the effective temperature for every head.

        Returns
        -------
        torch.Tensor
            Temperature values of shape ``(num_heads,)``, each
            guaranteed to be >= ``tau_min``.
        """
        tau = F.softplus(self.raw_tau) + self.tau_min
        self.last_tau = tau
        return tau

    def get_mean_temperature(self) -> Optional[torch.Tensor]:
        """Return mean temperature across heads for UCAT loss."""
        if self.last_tau is None:
            return None
        return self.last_tau.mean()

    def extra_repr(self) -> str:
        return f"num_heads={self.num_heads}, tau_min={self.tau_min}"


class TemperaturePredictor(nn.Module):
    """Predict input-dependent (aleatoric) temperature from the CLS token.

    Given a CLS token representation, this small MLP predicts per-head
    aleatoric temperatures that vary per image.  High aleatoric
    temperature indicates the image is inherently ambiguous (blur,
    overlapping classes, poor quality).

    Architecture: Linear -> GELU -> Linear -> Softplus + tau_min

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the CLS token (e.g., 192 for ViT-Tiny).
    num_heads : int
        Number of attention heads.
    tau_min : float
        Minimum temperature floor (default 0.1).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        tau_min: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.tau_min = tau_min
        hidden_dim = embed_dim // 4

        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads),
        )

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        """Predict aleatoric temperature from the CLS token.

        Parameters
        ----------
        cls_token : torch.Tensor
            CLS token of shape ``(B, embed_dim)``.

        Returns
        -------
        torch.Tensor
            Aleatoric temperatures of shape ``(B, num_heads)``,
            guaranteed >= ``tau_min``.
        """
        raw = self.net(cls_token)
        return F.softplus(raw) + self.tau_min

    def extra_repr(self) -> str:
        return f"num_heads={self.num_heads}, tau_min={self.tau_min}"


class EarlyExitController:
    """Confidence-based early exit for inference-time efficiency.

    During inference, if the attention distribution across all heads is
    sufficiently peaked (i.e., the model is "confident"), later
    transformer layers can be skipped.  Confidence is measured as the
    mean of the per-head maximum attention weight.

    The controller is inactive during training (``should_exit`` always
    returns ``False``) and enforces a minimum number of layers before
    any exit is permitted.

    Parameters
    ----------
    threshold : float, optional
        Confidence level (in ``[0, 1]``) above which an early exit is
        triggered.  Default is ``0.9``.
    min_layers : int, optional
        Minimum number of layers that must be evaluated before an early
        exit is allowed, regardless of confidence.  Default is ``4``.

    Examples
    --------
    >>> ctrl = EarlyExitController(threshold=0.9, min_layers=4)
    >>> # Simulated high-confidence attention from layer 5
    >>> weights = torch.zeros(1, 8, 10, 10)
    >>> weights[:, :, :, 0] = 1.0  # all mass on first key
    >>> ctrl.should_exit(weights, layer_idx=5)
    True
    """

    def __init__(
        self,
        threshold: float = 0.9,
        min_layers: int = 4,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"threshold must be in [0, 1], got {threshold}"
            )
        if min_layers < 0:
            raise ValueError(
                f"min_layers must be non-negative, got {min_layers}"
            )
        self.threshold = threshold
        self.min_layers = min_layers

    def should_exit(
        self,
        attn_weights: torch.Tensor,
        layer_idx: int,
    ) -> bool:
        """Decide whether the network should exit early.

        Parameters
        ----------
        attn_weights : torch.Tensor
            Attention weight tensor of shape
            ``(batch, num_heads, seq_len, seq_len)``.  Values are
            expected to be non-negative and sum to 1 along the last
            dimension (i.e., post-softmax).
        layer_idx : int
            Zero-based index of the current transformer layer.

        Returns
        -------
        bool
            ``True`` if an early exit should be taken; ``False``
            otherwise.  Always returns ``False`` during training
            (i.e., when ``attn_weights.requires_grad`` is ``True``
            or when the calling module is in training mode).
        """
        # Never exit during training.
        if attn_weights.requires_grad:
            return False

        # Respect the minimum-layer constraint.
        if layer_idx < self.min_layers:
            return False

        # Confidence = mean of per-head max attention weight.
        # attn_weights: (batch, heads, q_len, k_len)
        max_per_query, _ = attn_weights.max(dim=-1)      # (B, H, Q)
        confidence: float = max_per_query.mean().item()   # scalar

        return confidence >= self.threshold

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"threshold={self.threshold}, "
            f"min_layers={self.min_layers})"
        )


class LearnedHeadDropout(nn.Module):
    """Per-head learned dropout for attention weights.

    Instead of applying a single, fixed dropout rate to all heads, this
    module learns an individual dropout probability for each head.  The
    raw logit is passed through ``sigmoid`` and scaled by ``p_max`` so
    that the effective rate is always in ``[0, p_max]``.

    During training, each element of the attention weight matrix is
    independently zeroed with the head's learned probability and the
    surviving weights are rescaled by ``1 / (1 - p)`` (inverted
    dropout) so that expected values are preserved.

    During evaluation, no dropout is applied and the weights are
    returned unchanged.

    Parameters
    ----------
    num_heads : int
        Number of attention heads.
    p_max : float, optional
        Upper bound on the per-head dropout probability.
        Default is ``0.3``.

    Examples
    --------
    >>> hd = LearnedHeadDropout(num_heads=8, p_max=0.3)
    >>> hd.train()
    LearnedHeadDropout(num_heads=8, p_max=0.3)
    >>> w = torch.ones(2, 8, 5, 5)
    >>> out = hd(w)
    >>> out.shape
    torch.Size([2, 8, 5, 5])
    """

    def __init__(self, num_heads: int, p_max: float = 0.3) -> None:
        super().__init__()
        if not 0.0 < p_max < 1.0:
            raise ValueError(f"p_max must be in (0, 1), got {p_max}")
        self.num_heads = num_heads
        self.p_max = p_max

        # Initialise logits to zero -> sigmoid(0) = 0.5 -> p = 0.5 * p_max.
        self.drop_logits = nn.Parameter(torch.zeros(num_heads))

    def _get_drop_probs(self) -> torch.Tensor:
        """Return effective dropout probabilities per head.

        Returns
        -------
        torch.Tensor
            Shape ``(num_heads,)`` with values in ``[0, p_max]``.
        """
        return torch.sigmoid(self.drop_logits) * self.p_max

    def forward(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """Apply per-head learned dropout to attention weights.

        Parameters
        ----------
        attn_weights : torch.Tensor
            Attention weights of shape
            ``(batch, num_heads, seq_len, seq_len)``.

        Returns
        -------
        torch.Tensor
            Attention weights with per-head dropout applied (training)
            or unchanged (evaluation).  Same shape as input.
        """
        if not self.training:
            return attn_weights

        # p: (num_heads,) -> (1, num_heads, 1, 1) for broadcasting.
        p = self._get_drop_probs().view(1, self.num_heads, 1, 1)

        # Generate binary mask: 1 = keep, 0 = drop.
        mask = torch.bernoulli(
            torch.ones_like(attn_weights) - p.expand_as(attn_weights)
        )

        # Inverted dropout: rescale kept values so expected value is
        # unchanged.  Clamp denominator to avoid division by zero in
        # the unlikely event p == 1.
        scale = 1.0 / (1.0 - p).clamp(min=1e-6)

        return attn_weights * mask * scale

    def extra_repr(self) -> str:
        return f"num_heads={self.num_heads}, p_max={self.p_max}"


class LearnedResidualWeight(nn.Module):
    """Learned interpolation between attention output and residual.

    In a standard transformer, the residual connection adds the
    attention output directly: ``x + attn(x)``.  This module learns
    a scalar mixing coefficient ``alpha`` (one per layer instance):

        output = alpha * attn_out + (1 - alpha) * residual

    ``alpha`` is parameterised as ``sigmoid(raw_alpha)`` so it is
    always in ``[0, 1]``.  The raw parameter is initialised to ``0``
    so that ``alpha`` starts at ``0.5`` (equal weighting).

    Examples
    --------
    >>> lrw = LearnedResidualWeight()
    >>> attn_out = torch.randn(2, 10, 512)
    >>> residual = torch.randn(2, 10, 512)
    >>> out = lrw(attn_out, residual)
    >>> out.shape
    torch.Size([2, 10, 512])
    """

    def __init__(self) -> None:
        super().__init__()
        # Initialise to 0 -> sigmoid(0) = 0.5.
        self.raw_alpha = nn.Parameter(torch.zeros(1))

    @property
    def alpha(self) -> torch.Tensor:
        """Effective mixing coefficient in ``[0, 1]``."""
        return torch.sigmoid(self.raw_alpha)

    def forward(
        self,
        attn_out: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the learned weighted combination.

        Parameters
        ----------
        attn_out : torch.Tensor
            Output of the attention sub-layer.  Arbitrary shape.
        residual : torch.Tensor
            Residual (skip-connection) tensor.  Must be
            broadcastable with ``attn_out``.

        Returns
        -------
        torch.Tensor
            ``alpha * attn_out + (1 - alpha) * residual``, same shape
            as ``attn_out``.
        """
        a = self.alpha
        return a * attn_out + (1.0 - a) * residual

    def extra_repr(self) -> str:
        return f"alpha_init=0.5"


class TemperatureScheduler:
    """Training-time annealing schedule for attention temperature.

    During training, attention logits are scaled by a multiplier that
    decreases from ``tau_max_mult`` at the start of training to
    ``tau_min_mult`` at the end, following a power-law schedule:

        multiplier = tau_min_mult
                     + (tau_max_mult - tau_min_mult)
                       * (1 - progress) ** power

    where ``progress`` is in ``[0, 1]`` (0 = start, 1 = end).

    This encourages broad (soft) attention early in training and
    sharper attention later, aiding both optimisation and final
    performance.

    Parameters
    ----------
    tau_max_mult : float, optional
        Temperature multiplier at the start of training.
        Default is ``1.5``.
    tau_min_mult : float, optional
        Temperature multiplier at the end of training.
        Default is ``1.0``.
    power : float, optional
        Exponent controlling the annealing shape.  ``power=1`` gives
        linear decay; ``power=2`` (default) gives quadratic
        (concave-up) decay that stays high longer before dropping.

    Examples
    --------
    >>> sched = TemperatureScheduler(tau_max_mult=1.5, tau_min_mult=1.0)
    >>> sched.get_multiplier(0.0)   # start
    1.5
    >>> sched.get_multiplier(1.0)   # end
    1.0
    >>> 1.0 < sched.get_multiplier(0.5) < 1.5
    True
    """

    def __init__(
        self,
        tau_max_mult: float = 1.5,
        tau_min_mult: float = 1.0,
        power: float = 2.0,
    ) -> None:
        if tau_max_mult < tau_min_mult:
            raise ValueError(
                f"tau_max_mult ({tau_max_mult}) must be >= "
                f"tau_min_mult ({tau_min_mult})"
            )
        if power <= 0:
            raise ValueError(f"power must be positive, got {power}")

        self.tau_max_mult = tau_max_mult
        self.tau_min_mult = tau_min_mult
        self.power = power

    def get_multiplier(self, progress: float) -> float:
        """Compute the temperature multiplier at a given training progress.

        Parameters
        ----------
        progress : float
            Training progress in ``[0, 1]``, where ``0`` is the
            beginning and ``1`` is the end of training.

        Returns
        -------
        float
            Temperature multiplier in
            ``[tau_min_mult, tau_max_mult]``.

        Raises
        ------
        ValueError
            If ``progress`` is outside ``[0, 1]``.
        """
        if not 0.0 <= progress <= 1.0:
            raise ValueError(
                f"progress must be in [0, 1], got {progress}"
            )

        decay = (1.0 - progress) ** self.power
        return self.tau_min_mult + (self.tau_max_mult - self.tau_min_mult) * decay

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"tau_max_mult={self.tau_max_mult}, "
            f"tau_min_mult={self.tau_min_mult}, "
            f"power={self.power})"
        )
