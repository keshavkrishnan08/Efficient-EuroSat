"""
Efficient Satellite Attention Module
=====================================

Combines all five attention modifications into a single unified attention
module (EfficientSatAttention) designed for satellite land use classification.

Modifications integrated:
    1. LearnableTemperature   -- per-head softmax temperature scaling
    2. EarlyExitController    -- confidence-based early layer exit
    3. LearnedHeadDropout     -- input-dependent per-head dropout rates
    4. LearnedResidualWeight  -- learned blending of residual and attention
    5. TemperatureScheduler   -- curriculum-style temperature annealing

Each modification can be independently toggled on or off, making the module
suitable for ablation studies.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention_modifications import (
    LearnableTemperature,
    TemperaturePredictor,
    EarlyExitController,
    LearnedHeadDropout,
    LearnedResidualWeight,
    TemperatureScheduler,
)


class EfficientSatAttention(nn.Module):
    """Multi-head self-attention with five learnable efficiency modifications.

    This module replaces the standard multi-head attention layer inside a
    Vision Transformer block.  It follows the canonical QKV-projection
    pattern while weaving in five optional modifications that can each be
    toggled independently via constructor flags.

    Parameters
    ----------
    dim : int
        Embedding dimension (e.g., 192 for a tiny ViT).
    num_heads : int
        Number of attention heads (e.g., 3).
    qkv_bias : bool
        Whether the QKV projection includes a bias term.
    attn_drop : float
        Base dropout probability applied to attention weights.
    proj_drop : float
        Dropout probability applied after the output projection.
    use_learned_temp : bool
        Enable per-head learnable temperature scaling.
    use_early_exit : bool
        Enable early-exit confidence checking.
    use_learned_dropout : bool
        Enable input-dependent learned head dropout.
    use_learned_residual : bool
        Enable learned residual blending weight.
    use_temp_annealing : bool
        Enable curriculum-style temperature annealing schedule.
    tau_min : float
        Minimum temperature value for the learnable temperature module.
    dropout_max : float
        Maximum per-head dropout rate for learned dropout.
    exit_threshold : float
        Confidence threshold above which early exit is triggered.
    exit_min_layer : int
        Earliest layer index at which early exit is permitted.
    layer_idx : int
        Index of this layer within the full transformer stack.
    num_layers : int
        Total number of layers in the transformer stack.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        # --- modification toggles ---
        use_learned_temp: bool = True,
        use_early_exit: bool = True,
        use_learned_dropout: bool = True,
        use_learned_residual: bool = True,
        use_temp_annealing: bool = True,
        # --- modification hyperparameters ---
        tau_min: float = 0.1,
        dropout_max: float = 0.3,
        exit_threshold: float = 0.9,
        exit_min_layer: int = 4,
        layer_idx: int = 0,
        num_layers: int = 12,
    ) -> None:
        super().__init__()

        assert dim % num_heads == 0, (
            f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        )

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5  # default 1/sqrt(d_k)
        self.layer_idx = layer_idx
        self.num_layers = num_layers

        # Toggle flags (stored for introspection / serialisation)
        self.use_learned_temp = use_learned_temp
        self.use_early_exit = use_early_exit
        self.use_learned_dropout = use_learned_dropout
        self.use_learned_residual = use_learned_residual
        self.use_temp_annealing = use_temp_annealing

        # ---- Core projections ------------------------------------------------
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # ---- Modification 1: Learnable Temperature ---------------------------
        self.learned_temp: Optional[LearnableTemperature] = None
        if use_learned_temp:
            self.learned_temp = LearnableTemperature(
                num_heads=num_heads,
                tau_min=tau_min,
            )

        # ---- Modification 2: Early Exit Controller ---------------------------
        self.early_exit: Optional[EarlyExitController] = None
        if use_early_exit:
            self.early_exit = EarlyExitController(
                threshold=exit_threshold,
                min_layers=exit_min_layer,
            )

        # ---- Modification 3: Learned Head Dropout ----------------------------
        self.learned_dropout: Optional[LearnedHeadDropout] = None
        if use_learned_dropout:
            self.learned_dropout = LearnedHeadDropout(
                num_heads=num_heads,
                p_max=dropout_max,
            )

        # ---- Modification 4: Learned Residual Weight -------------------------
        self.learned_residual: Optional[LearnedResidualWeight] = None
        if use_learned_residual:
            self.learned_residual = LearnedResidualWeight()

        # ---- Modification 5: Temperature Scheduler --------------------------
        self.temp_scheduler: Optional[TemperatureScheduler] = None
        if use_temp_annealing:
            self.temp_scheduler = TemperatureScheduler()

        # ---- Internal state for diagnostics ----------------------------------
        self._last_attn_weights: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        training_progress: float = 0.0,
        return_exit_signal: bool = False,
        tau_aleatoric: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute multi-head self-attention with optional modifications.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, N, C)`` where *B* is batch size,
            *N* is the number of tokens, and *C* equals ``self.dim``.
        training_progress : float
            Scalar in ``[0, 1]`` representing how far training has
            progressed (0 = start, 1 = end).  Used by the temperature
            scheduler for curriculum annealing.
        return_exit_signal : bool
            If ``True`` **and** early exit is enabled, an additional
            boolean tensor indicating whether this layer's output is
            confident enough to skip remaining layers is returned.
        tau_aleatoric : torch.Tensor or None
            Input-dependent aleatoric temperature of shape ``(B, H)``.
            When provided, the epistemic (learned) temperature is combined
            additively: ``tau_total = tau_a + tau_e``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(B, N, C)``.
        torch.Tensor, optional
            Exit signal of shape ``(B,)`` with boolean values, returned
            only when ``return_exit_signal=True`` and early exit is
            enabled.
        """
        B, N, C = x.shape

        # --- QKV projection --------------------------------------------------
        qkv = self.qkv(x)                                    # (B, N, 3*C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                    # (3, B, H, N, d)
        q, k, v = qkv.unbind(0)                              # each (B, H, N, d)

        # --- Temperature computation -----------------------------------------
        # Epistemic temperature: per-head learned parameter (shared across batch)
        tau_e = q.new_ones(1)  # scalar tensor on correct device/dtype

        if self.learned_temp is not None:
            tau_e = self.learned_temp()                       # (H,)

        # Apply annealing only to epistemic component
        if self.temp_scheduler is not None:
            anneal_factor = self.temp_scheduler.get_multiplier(training_progress)
            tau_e = tau_e * anneal_factor

        # Combine with aleatoric temperature if provided
        if tau_aleatoric is not None:
            # tau_aleatoric: (B, H) -> (B, H, 1, 1)
            tau_a = tau_aleatoric.view(B, self.num_heads, 1, 1)
            # tau_e: (H,) -> (1, H, 1, 1)
            if isinstance(tau_e, torch.Tensor) and tau_e.dim() >= 1 and tau_e.numel() > 1:
                tau_e_broad = tau_e.view(1, -1, 1, 1)
            else:
                tau_e_broad = tau_e
            tau = tau_a + tau_e_broad  # (B, H, 1, 1)
        else:
            tau = tau_e
            # Reshape for broadcasting
            if isinstance(tau, torch.Tensor) and tau.dim() >= 1 and tau.numel() > 1:
                tau = tau.view(1, -1, 1, 1)                  # (1, H, 1, 1)

        # --- Attention scores -------------------------------------------------
        attn = (q @ k.transpose(-2, -1)) * self.scale        # (B, H, N, N)
        attn = attn / tau

        attn = attn.softmax(dim=-1)                           # (B, H, N, N)

        # Store attention weights for diagnostics (detach to avoid graph leak)
        self._last_attn_weights = attn.detach()

        # --- Learned head dropout ---------------------------------------------
        if self.learned_dropout is not None:
            attn = self.learned_dropout(attn)

        # Standard attention dropout (applied regardless of learned dropout)
        attn = self.attn_drop(attn)

        # --- Weighted sum of values -------------------------------------------
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)    # (B, N, C)

        # --- Output projection ------------------------------------------------
        out = self.proj(out)
        out = self.proj_drop(out)

        # --- Residual connection ----------------------------------------------
        if self.learned_residual is not None:
            out = self.learned_residual(attn_out=out, residual=x)
        else:
            out = x + out

        # --- Early exit check -------------------------------------------------
        if return_exit_signal and self.early_exit is not None:
            # Compute per-sample confidence from stored attention weights.
            # Confidence = mean of per-head max attention weight per sample.
            attn_w = self._last_attn_weights          # (B, H, N, N)
            max_per_query, _ = attn_w.max(dim=-1)     # (B, H, N)
            confidence = max_per_query.mean(dim=(1, 2))  # (B,)
            return out, confidence

        return out, None

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------
    def get_attention_stats(self) -> Dict[str, object]:
        """Return a dictionary of learned parameter values for logging.

        The dictionary may contain any subset of the following keys
        depending on which modifications are enabled:

        - ``"learned_temperatures"``  -- per-head temperature values
        - ``"head_dropout_rates"``    -- current per-head dropout rates
        - ``"residual_weight"``       -- learned residual blending weight
        - ``"exit_threshold"``        -- current early-exit threshold
        - ``"layer_idx"``             -- index of this layer
        """
        stats: Dict[str, object] = {"layer_idx": self.layer_idx}

        if self.learned_temp is not None:
            with torch.no_grad():
                stats["learned_temperatures"] = self.learned_temp().detach().cpu()

        if self.learned_dropout is not None:
            with torch.no_grad():
                stats["head_dropout_rates"] = (
                    self.learned_dropout.get_dropout_rates().detach().cpu()
                    if hasattr(self.learned_dropout, "get_dropout_rates")
                    else None
                )

        if self.learned_residual is not None:
            with torch.no_grad():
                stats["residual_weight"] = (
                    self.learned_residual.get_weight().detach().cpu()
                    if hasattr(self.learned_residual, "get_weight")
                    else None
                )

        if self.early_exit is not None:
            stats["exit_threshold"] = (
                self.early_exit.threshold
                if hasattr(self.early_exit, "threshold")
                else None
            )

        return stats

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Return the attention weights from the most recent forward pass.

        Returns
        -------
        torch.Tensor or None
            Attention weight tensor of shape ``(B, H, N, N)`` from the
            last call to :meth:`forward`, or ``None`` if ``forward`` has
            not been called yet.  The tensor is detached from the
            computation graph.
        """
        return self._last_attn_weights

    def get_mean_temperature(self) -> Optional[torch.Tensor]:
        """Return mean temperature from the last forward pass for UCAT loss.

        Returns the mean of per-head learned temperatures if learned
        temperature is enabled, otherwise ``None``.
        """
        if self.learned_temp is not None:
            return self.learned_temp.get_mean_temperature()
        return None

    def get_epistemic_temperature(self) -> Optional[torch.Tensor]:
        """Return per-head epistemic temperatures (learned, shared across batch).

        Returns
        -------
        torch.Tensor or None
            Epistemic temperatures of shape ``(num_heads,)`` or ``None``
            if learned temperature is disabled.
        """
        if self.learned_temp is not None:
            return self.learned_temp()
        return None
