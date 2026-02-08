"""
Efficient Vision Transformer for Satellite Land Use Classification (EfficientEuroSAT-ViT).

This module implements a Vision Transformer architecture with five attention
modifications for improved satellite land use classification:

    1. Learned Temperature Scaling
    2. Early Exit via CLS-token confidence
    3. Learned Attention Dropout
    4. Learned Residual Weight (internal to attention)
    5. Temperature Annealing (schedule-based tau decay)

Architecture defaults correspond to ViT-Tiny:
    - Embedding dimension : 192
    - Layers             : 12
    - Heads              : 3
    - MLP ratio          : 4
    - Patch size         : 16x16
    - Input resolution   : 224x224
    - Patches            : 196 (14x14)
    - Parameters         : ~5.7M

References:
    Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for
    Image Recognition at Scale", ICLR 2021.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modified_attention import EfficientSatAttention
from .attention_modifications import TemperaturePredictor


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """Convert an input image into a sequence of flattened patch embeddings.

    A single strided convolution projects each non-overlapping patch of size
    ``patch_size x patch_size`` into an ``embed_dim``-dimensional vector.

    Parameters
    ----------
    img_size : int
        Spatial resolution of the square input image (default: 224).
    patch_size : int
        Side length of each square patch (default: 16).
    in_channels : int
        Number of input image channels (default: 3).
    embed_dim : int
        Dimensionality of each patch embedding (default: 192).
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 192,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 196 for 224/16

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project image patches to embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Input images of shape ``[B, C, H, W]``.

        Returns
        -------
        torch.Tensor
            Patch embeddings of shape ``[B, num_patches, embed_dim]``.
        """
        # x: [B, C, H, W] -> [B, embed_dim, H/P, W/P]
        x = self.proj(x)
        # [B, embed_dim, H/P, W/P] -> [B, embed_dim, num_patches]
        x = x.flatten(2)
        # [B, embed_dim, num_patches] -> [B, num_patches, embed_dim]
        x = x.transpose(1, 2)
        return x


# ---------------------------------------------------------------------------
# MLP (Feed-Forward Network)
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Two-layer feed-forward network used inside each Transformer block.

    Architecture: ``Linear -> GELU -> Dropout -> Linear -> Dropout``.

    Parameters
    ----------
    embed_dim : int
        Input and output dimensionality.
    mlp_ratio : float
        Expansion ratio for the hidden layer (default: 4.0).
    dropout : float
        Dropout probability applied after each linear layer (default: 0.0).
    """

    def __init__(
        self,
        embed_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``[B, N, embed_dim]``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``[B, N, embed_dim]``.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Single Transformer encoder block with EfficientEuroSAT attention.

    Structure::

        x -> LayerNorm -> EfficientSatAttention -> (+residual*) -> LayerNorm -> MLP -> (+residual) -> out

    *When ``use_learned_residual`` is **True** the residual connection around
    the attention sub-layer is handled *internally* by
    :class:`EfficientSatAttention` (Modification 4 -- Learned Residual
    Weight).  In that case the block does **not** add an external residual
    for the attention path.  When ``use_learned_residual`` is **False** a
    standard external residual is applied.

    The MLP path always uses a standard external residual.

    Parameters
    ----------
    embed_dim : int
        Token embedding dimensionality.
    num_heads : int
        Number of attention heads.
    mlp_ratio : float
        MLP hidden-dim expansion factor.
    dropout : float
        Dropout probability for MLP layers.
    use_learned_temp : bool
        Enable learned temperature scaling in attention.
    use_early_exit : bool
        Enable early-exit confidence signal.
    use_learned_dropout : bool
        Enable learned per-head attention dropout.
    use_learned_residual : bool
        Enable learned residual weighting inside attention.
    use_temp_annealing : bool
        Enable temperature annealing schedule.
    tau_min : float
        Minimum temperature for annealing.
    dropout_max : float
        Upper bound for learned attention dropout.
    exit_threshold : float
        Confidence threshold for early exit.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_learned_temp: bool = True,
        use_early_exit: bool = True,
        use_learned_dropout: bool = True,
        use_learned_residual: bool = True,
        use_temp_annealing: bool = True,
        tau_min: float = 0.5,
        dropout_max: float = 0.3,
        exit_threshold: float = 0.9,
        layer_idx: int = 0,
        num_layers: int = 12,
    ) -> None:
        super().__init__()

        self.use_learned_residual = use_learned_residual

        # Pre-norm for attention
        self.norm1 = nn.LayerNorm(embed_dim)

        # Modified multi-head self-attention
        self.attn = EfficientSatAttention(
            dim=embed_dim,
            num_heads=num_heads,
            use_learned_temp=use_learned_temp,
            use_early_exit=use_early_exit,
            use_learned_dropout=use_learned_dropout,
            use_learned_residual=use_learned_residual,
            use_temp_annealing=use_temp_annealing,
            tau_min=tau_min,
            dropout_max=dropout_max,
            exit_threshold=exit_threshold,
            layer_idx=layer_idx,
            num_layers=num_layers,
        )

        # Pre-norm for MLP
        self.norm2 = nn.LayerNorm(embed_dim)

        # Feed-forward network
        self.mlp = MLP(embed_dim=embed_dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        training_progress: float = 0.0,
        return_exit_signal: bool = False,
        tau_aleatoric: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through the Transformer block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``[B, N, embed_dim]``.
        training_progress : float
            Fraction of training completed, in ``[0, 1]``.  Used by
            temperature annealing and other schedule-aware modifications.
        return_exit_signal : bool
            If ``True``, request the attention module to return an early-exit
            confidence signal derived from the CLS token.
        tau_aleatoric : torch.Tensor or None
            Input-dependent aleatoric temperature ``[B, num_heads]``.

        Returns
        -------
        Tuple[torch.Tensor, Optional[torch.Tensor]]
            - Output tensor of shape ``[B, N, embed_dim]``.
            - Early-exit confidence signal of shape ``[B]`` if requested and
              available, otherwise ``None``.
        """
        # ----- Attention sub-layer -----
        normed = self.norm1(x)
        attn_out, exit_signal = self.attn(
            normed,
            training_progress=training_progress,
            return_exit_signal=return_exit_signal,
            tau_aleatoric=tau_aleatoric,
        )

        if self.use_learned_residual:
            # Residual is handled inside EfficientSatAttention via
            # LearnedResidualWeight; attn_out already incorporates the
            # skip connection (alpha * attn + (1-alpha) * input).
            # We must pass the *original* x as well so the internal
            # residual has access to it.  The EfficientSatAttention module
            # receives the pre-normed input but applies the residual with
            # the un-normed x.  To support this, we add x only when the
            # attention module does NOT handle it.  Since the attention
            # module receives `normed` and adds residual internally with
            # `normed`, we need to add the difference (x - normed) to
            # recover the original residual path.
            #
            # Actually -- the standard design for a learned-residual
            # attention is:  output = alpha * Attn(LN(x)) + (1-alpha) * LN(x)
            # which means the norm is baked in.  We therefore need to add
            # back the *pre-norm residual* component.  The cleanest way:
            # the attention module returns alpha*Attn(normed) + (1-alpha)*normed.
            # We replace the standard residual x + Attn(LN(x)) with this,
            # so we do NOT add x again.
            x = attn_out
        else:
            # Standard residual connection
            x = x + attn_out

        # ----- MLP sub-layer (always standard residual) -----
        x = x + self.mlp(self.norm2(x))

        return x, exit_signal


# ---------------------------------------------------------------------------
# EfficientEuroSAT Vision Transformer
# ---------------------------------------------------------------------------

class EfficientEuroSATViT(nn.Module):
    """Vision Transformer with EfficientEuroSAT attention modifications.

    This model integrates five learned attention modifications targeting
    efficient satellite land use classification:

        1. **Learned Temperature Scaling** -- per-head softmax temperature.
        2. **Early Exit** -- CLS-token confidence gates for skipping later
           layers at inference time.
        3. **Learned Attention Dropout** -- per-head, training-adaptive
           dropout on attention weights.
        4. **Learned Residual Weight** -- learnable blending coefficient
           between attention output and skip connection.
        5. **Temperature Annealing** -- cosine schedule that decays
           temperatures from 1 to ``tau_min`` over training.

    Parameters
    ----------
    img_size : int
        Input image spatial resolution.
    patch_size : int
        Patch side length.
    in_channels : int
        Number of input channels.
    num_classes : int
        Number of output classes (10 for EuroSAT).
    embed_dim : int
        Token embedding dimensionality.
    depth : int
        Number of Transformer blocks.
    num_heads : int
        Number of attention heads per block.
    mlp_ratio : float
        MLP hidden-dimension expansion factor.
    dropout : float
        Dropout applied after patch embedding and inside MLP layers.
    use_learned_temp : bool
        Enable Modification 1.
    use_early_exit : bool
        Enable Modification 2.
    use_learned_dropout : bool
        Enable Modification 3.
    use_learned_residual : bool
        Enable Modification 4.
    use_temp_annealing : bool
        Enable Modification 5.
    tau_min : float
        Minimum temperature for annealing.
    dropout_max : float
        Upper bound for learned attention dropout.
    exit_threshold : float
        Confidence threshold for early exit.
    exit_min_layer : int
        Earliest layer at which early exit may be triggered (0-indexed).
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_learned_temp: bool = True,
        use_early_exit: bool = True,
        use_learned_dropout: bool = True,
        use_learned_residual: bool = True,
        use_temp_annealing: bool = True,
        tau_min: float = 0.5,
        dropout_max: float = 0.3,
        exit_threshold: float = 0.9,
        exit_min_layer: int = 6,
        use_decomposition: bool = False,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.depth = depth
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.use_early_exit = use_early_exit
        self.use_learned_temp = use_learned_temp
        self.use_decomposition = use_decomposition
        self.exit_threshold = exit_threshold
        self.exit_min_layer = exit_min_layer
        self.early_exit_enabled = use_early_exit  # runtime toggle

        # --- Patch embedding ---
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches  # 196

        # --- Special tokens & positional embedding ---
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )  # +1 for CLS

        # --- Embedding dropout ---
        self.pos_drop = nn.Dropout(p=dropout)

        # --- Transformer blocks ---
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_learned_temp=use_learned_temp,
                use_early_exit=use_early_exit,
                use_learned_dropout=use_learned_dropout,
                use_learned_residual=use_learned_residual,
                use_temp_annealing=use_temp_annealing,
                tau_min=tau_min,
                dropout_max=dropout_max,
                exit_threshold=exit_threshold,
                layer_idx=idx,
                num_layers=depth,
            )
            for idx in range(depth)
        ])

        # --- Temperature predictor for decomposition ---
        self.temp_predictor: Optional[TemperaturePredictor] = None
        if use_decomposition and use_learned_temp:
            self.temp_predictor = TemperaturePredictor(
                embed_dim=embed_dim,
                num_heads=num_heads,
                tau_min=tau_min,
            )

        # --- Final norm & classification head ---
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # --- Exit statistics tracking ---
        self._exit_layer_counts: Dict[int, int] = defaultdict(int)
        self._total_samples: int = 0

        # --- Weight initialisation ---
        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Apply weight initialisation following ViT conventions."""
        # Positional embedding: truncated normal
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        # CLS token: truncated normal
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Recursively initialise Linear and LayerNorm layers
        self.apply(self._init_module)

    @staticmethod
    def _init_module(module: nn.Module) -> None:
        """Per-module initialisation callback."""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            # Fan-out Kaiming init for the patch projection
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            nn.init.trunc_normal_(module.weight, std=math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        training_progress: float = 0.0,
        return_temperatures: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Full forward pass with optional early exit during inference.

        Parameters
        ----------
        x : torch.Tensor
            Input images of shape ``[B, 3, H, W]``.
        training_progress : float
            Fraction of training completed (``0.0`` to ``1.0``).  Forwarded
            to each :class:`TransformerBlock` to drive schedule-based
            modifications (e.g., temperature annealing).
        return_temperatures : bool
            If ``True``, return a tuple of ``(logits, temperatures)`` where
            temperatures contains the mean learned temperature across all
            blocks, expanded to shape ``[B]``.

        Returns
        -------
        torch.Tensor or tuple
            If ``return_temperatures`` is ``False``: logits of shape
            ``[B, num_classes]``.
            If ``return_temperatures`` is ``True``: a tuple of
            ``(logits, temperatures)`` where temperatures has shape ``[B]``.
        """
        B = x.shape[0]

        # --- Patch embedding + CLS token + positional encoding ---
        x = self.patch_embed(x)  # [B, 196, embed_dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 197, embed_dim]
        x = x + self.pos_embed  # [B, 197, embed_dim]
        x = self.pos_drop(x)

        # --- Reset exit statistics for this batch ---
        if not self.training:
            self._exit_layer_counts = defaultdict(int)
            self._total_samples = 0

        # --- Predict aleatoric temperature if decomposition is enabled ---
        tau_a_per_sample = None
        if self.temp_predictor is not None:
            # Use CLS token as the global image representation
            cls_repr = x[:, 0]  # [B, embed_dim]
            tau_a_per_sample = self.temp_predictor(cls_repr)  # [B, num_heads]

        # --- Transformer blocks ---
        if not self.training and self.early_exit_enabled and self.use_early_exit:
            # Inference with early exit
            x = self._forward_with_early_exit(x, training_progress, tau_a_per_sample)
        else:
            # Training or early exit disabled -- full forward pass
            for block in self.blocks:
                x, _ = block(
                    x,
                    training_progress=training_progress,
                    return_exit_signal=False,
                    tau_aleatoric=tau_a_per_sample,
                )

            # Final norm + classify from CLS token
            x = self.norm(x)
            x = self.head(x[:, 0])  # [B, num_classes]

        if return_temperatures:
            if self.use_decomposition and tau_a_per_sample is not None:
                # Return decomposed: (logits, tau_a_mean, tau_e_mean)
                tau_a_mean = tau_a_per_sample.mean(dim=-1)  # [B]
                tau_e = self._collect_epistemic_temperatures()
                tau_e_mean = tau_e if tau_e is not None else torch.zeros(1, device=x.device)
                # For backward compat, tau_total as the main temperature
                batch_temps = tau_a_mean + tau_e_mean.expand(x.shape[0])
                return x, batch_temps, tau_a_mean, tau_e_mean
            else:
                temps = self._collect_mean_temperatures()
                if temps is not None:
                    batch_temps = temps.expand(x.shape[0])
                else:
                    batch_temps = torch.zeros(x.shape[0], device=x.device)
                return x, batch_temps

        return x

    def _collect_mean_temperatures(self) -> Optional[torch.Tensor]:
        """Collect mean learned temperature across all blocks.

        Returns the average of per-block mean temperatures, or ``None``
        if no blocks have learned temperature enabled.
        """
        temps = []
        for block in self.blocks:
            t = block.attn.get_mean_temperature()
            if t is not None:
                temps.append(t)
        if temps:
            return torch.stack(temps).mean()
        return None

    def _collect_epistemic_temperatures(self) -> Optional[torch.Tensor]:
        """Collect mean epistemic temperature across all blocks.

        Returns scalar mean of all per-head learned temperatures.
        """
        temps = []
        for block in self.blocks:
            t = block.attn.get_epistemic_temperature()
            if t is not None:
                temps.append(t.mean())
        if temps:
            return torch.stack(temps).mean()
        return None

    def _forward_with_early_exit(
        self,
        x: torch.Tensor,
        training_progress: float,
        tau_aleatoric: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Inference-time forward with per-sample early exit.

        After each block (from ``exit_min_layer`` onwards), the exit
        confidence signal is checked.  Samples whose CLS-token confidence
        exceeds ``exit_threshold`` are routed through the classifier
        immediately, and their results are stored.  Remaining samples
        continue through deeper layers.

        Parameters
        ----------
        x : torch.Tensor
            Embedded tokens of shape ``[B, N, embed_dim]``.
        training_progress : float
            Training progress fraction.

        Returns
        -------
        torch.Tensor
            Logits of shape ``[B, num_classes]``.
        """
        B = x.shape[0]
        device = x.device

        # Pre-allocate output logits
        output = torch.zeros(B, self.num_classes, device=device)
        # Track which samples have already exited
        exited = torch.zeros(B, dtype=torch.bool, device=device)

        # Mapping from original batch index to current position
        active_indices = torch.arange(B, device=device)

        for layer_idx, block in enumerate(self.blocks):
            request_exit = (
                layer_idx >= self.exit_min_layer
                and active_indices.numel() > 0
            )

            # Slice tau_aleatoric to match active samples
            tau_a_active = None
            if tau_aleatoric is not None:
                tau_a_active = tau_aleatoric[active_indices] if active_indices.numel() < B else tau_aleatoric

            x, exit_signal = block(
                x,
                training_progress=training_progress,
                return_exit_signal=request_exit,
                tau_aleatoric=tau_a_active,
            )

            if request_exit and exit_signal is not None:
                # exit_signal: [B_active] confidence values
                should_exit = exit_signal >= self.exit_threshold  # [B_active]

                if should_exit.any():
                    # Classify exiting samples from their CLS tokens
                    exit_x = self.norm(x[should_exit])
                    exit_logits = self.head(exit_x[:, 0])  # [n_exit, num_classes]

                    # Store results in the output tensor
                    exit_orig_indices = active_indices[should_exit]
                    output[exit_orig_indices] = exit_logits

                    # Record exit statistics
                    n_exiting = int(should_exit.sum().item())
                    self._exit_layer_counts[layer_idx] += n_exiting
                    self._total_samples += n_exiting

                    # Remove exited samples from the active set
                    keep = ~should_exit
                    x = x[keep]
                    active_indices = active_indices[keep]

                    if active_indices.numel() == 0:
                        return output

        # Remaining samples exit at the final layer
        if active_indices.numel() > 0:
            x = self.norm(x)
            final_logits = self.head(x[:, 0])
            output[active_indices] = final_logits

            n_remaining = active_indices.numel()
            self._exit_layer_counts[self.depth - 1] += n_remaining
            self._total_samples += n_remaining

        return output

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def get_attention_stats(self) -> Dict[str, Any]:
        """Collect learned attention parameters from every layer.

        Returns
        -------
        Dict[str, Any]
            Dictionary with the following keys:

            - ``"temperatures"`` : list of per-head temperature tensors.
            - ``"dropout_rates"`` : list of per-head dropout rate tensors.
            - ``"residual_weights"`` : list of learned residual alpha tensors.
            - ``"exit_confidences"`` : list of exit gate bias / threshold
              parameters (if available).
        """
        stats: Dict[str, List[Any]] = {
            "temperatures": [],
            "dropout_rates": [],
            "residual_weights": [],
            "exit_confidences": [],
        }

        for idx, block in enumerate(self.blocks):
            attn = block.attn

            # Learned temperature (Mod 1)
            if hasattr(attn, "temperature") and attn.temperature is not None:
                stats["temperatures"].append(
                    attn.temperature.detach().clone()
                )
            elif hasattr(attn, "log_temperature") and attn.log_temperature is not None:
                stats["temperatures"].append(
                    attn.log_temperature.detach().exp().clone()
                )

            # Learned dropout rate (Mod 3)
            if hasattr(attn, "dropout_rate") and attn.dropout_rate is not None:
                stats["dropout_rates"].append(
                    attn.dropout_rate.detach().clone()
                )
            elif hasattr(attn, "log_dropout") and attn.log_dropout is not None:
                stats["dropout_rates"].append(
                    attn.log_dropout.detach().clone()
                )

            # Learned residual weight (Mod 4)
            if hasattr(attn, "residual_weight") and attn.residual_weight is not None:
                stats["residual_weights"].append(
                    attn.residual_weight.detach().clone()
                )
            elif hasattr(attn, "alpha") and attn.alpha is not None:
                stats["residual_weights"].append(
                    attn.alpha.detach().clone()
                )

            # Exit confidence parameters (Mod 2)
            if hasattr(attn, "exit_gate") and attn.exit_gate is not None:
                if isinstance(attn.exit_gate, nn.Module):
                    for param in attn.exit_gate.parameters():
                        stats["exit_confidences"].append(
                            param.detach().clone()
                        )
                else:
                    stats["exit_confidences"].append(
                        attn.exit_gate.detach().clone()
                    )

        return stats

    def get_exit_statistics(self) -> Dict[str, Any]:
        """Return early-exit distribution from the most recent forward pass.

        Returns
        -------
        Dict[str, Any]
            Dictionary with keys:

            - ``"exit_layer_counts"`` : ``Dict[int, int]`` mapping layer
              index to number of samples that exited at that layer.
            - ``"total_samples"`` : total number of samples processed.
            - ``"exit_layer_fractions"`` : ``Dict[int, float]`` mapping
              layer index to the fraction of samples exiting there.
            - ``"average_exit_layer"`` : weighted-average exit layer.
        """
        total = self._total_samples if self._total_samples > 0 else 1
        fractions = {
            layer: count / total
            for layer, count in self._exit_layer_counts.items()
        }

        # Weighted average exit layer
        if self._total_samples > 0:
            avg_layer = sum(
                layer * count
                for layer, count in self._exit_layer_counts.items()
            ) / self._total_samples
        else:
            avg_layer = float(self.depth - 1)

        return {
            "exit_layer_counts": dict(self._exit_layer_counts),
            "total_samples": self._total_samples,
            "exit_layer_fractions": fractions,
            "average_exit_layer": avg_layer,
        }

    def set_early_exit(self, enabled: bool) -> None:
        """Enable or disable early exit at inference time.

        Parameters
        ----------
        enabled : bool
            If ``True``, early exit is active during evaluation.  If
            ``False``, all samples pass through every layer.
        """
        self.early_exit_enabled = enabled

    def get_num_params(self) -> int:
        """Return the total number of learnable parameters.

        Returns
        -------
        int
            Total parameter count.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------------

def create_efficient_eurosat_tiny(
    num_classes: int = 10,
    *,
    use_learned_temp: bool = True,
    use_early_exit: bool = True,
    use_learned_dropout: bool = True,
    use_learned_residual: bool = True,
    use_temp_annealing: bool = True,
    tau_min: float = 0.5,
    dropout_max: float = 0.3,
    exit_threshold: float = 0.9,
    exit_min_layer: int = 6,
    use_decomposition: bool = False,
    **kwargs: Any,
) -> EfficientEuroSATViT:
    """Create an EfficientEuroSAT ViT-Tiny model.

    ViT-Tiny configuration: embed_dim=192, depth=12, num_heads=3.

    Parameters
    ----------
    num_classes : int
        Number of output classes (default: 10 for EuroSAT).
    use_learned_temp : bool
        Enable learned temperature scaling (Mod 1).
    use_early_exit : bool
        Enable early exit (Mod 2).
    use_learned_dropout : bool
        Enable learned attention dropout (Mod 3).
    use_learned_residual : bool
        Enable learned residual weight (Mod 4).
    use_temp_annealing : bool
        Enable temperature annealing (Mod 5).
    tau_min : float
        Minimum temperature for annealing.
    dropout_max : float
        Upper bound for learned dropout rate.
    exit_threshold : float
        Confidence threshold for early exit.
    exit_min_layer : int
        Earliest layer for early exit (0-indexed).
    **kwargs
        Additional keyword arguments forwarded to :class:`EfficientEuroSATViT`.

    Returns
    -------
    EfficientEuroSATViT
        Configured ViT-Tiny model.
    """
    return EfficientEuroSATViT(
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        use_learned_temp=use_learned_temp,
        use_early_exit=use_early_exit,
        use_learned_dropout=use_learned_dropout,
        use_learned_residual=use_learned_residual,
        use_temp_annealing=use_temp_annealing,
        tau_min=tau_min,
        dropout_max=dropout_max,
        exit_threshold=exit_threshold,
        exit_min_layer=exit_min_layer,
        use_decomposition=use_decomposition,
        **kwargs,
    )


def create_efficient_eurosat_small(
    num_classes: int = 10,
    *,
    use_learned_temp: bool = True,
    use_early_exit: bool = True,
    use_learned_dropout: bool = True,
    use_learned_residual: bool = True,
    use_temp_annealing: bool = True,
    tau_min: float = 0.5,
    dropout_max: float = 0.3,
    exit_threshold: float = 0.9,
    exit_min_layer: int = 6,
    **kwargs: Any,
) -> EfficientEuroSATViT:
    """Create an EfficientEuroSAT ViT-Small model.

    ViT-Small configuration: embed_dim=384, depth=12, num_heads=6.

    Parameters
    ----------
    num_classes : int
        Number of output classes (default: 10 for EuroSAT).
    use_learned_temp : bool
        Enable learned temperature scaling (Mod 1).
    use_early_exit : bool
        Enable early exit (Mod 2).
    use_learned_dropout : bool
        Enable learned attention dropout (Mod 3).
    use_learned_residual : bool
        Enable learned residual weight (Mod 4).
    use_temp_annealing : bool
        Enable temperature annealing (Mod 5).
    tau_min : float
        Minimum temperature for annealing.
    dropout_max : float
        Upper bound for learned dropout rate.
    exit_threshold : float
        Confidence threshold for early exit.
    exit_min_layer : int
        Earliest layer for early exit (0-indexed).
    **kwargs
        Additional keyword arguments forwarded to :class:`EfficientEuroSATViT`.

    Returns
    -------
    EfficientEuroSATViT
        Configured ViT-Small model.
    """
    return EfficientEuroSATViT(
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        use_learned_temp=use_learned_temp,
        use_early_exit=use_early_exit,
        use_learned_dropout=use_learned_dropout,
        use_learned_residual=use_learned_residual,
        use_temp_annealing=use_temp_annealing,
        tau_min=tau_min,
        dropout_max=dropout_max,
        exit_threshold=exit_threshold,
        exit_min_layer=exit_min_layer,
        **kwargs,
    )


def create_baseline_vit_tiny(
    num_classes: int = 10,
    **kwargs: Any,
) -> EfficientEuroSATViT:
    """Create a baseline ViT-Tiny with **all** modifications disabled.

    This serves as the control model for ablation studies.  The architecture
    is identical to :func:`create_efficient_eurosat_tiny` but none of the five
    EfficientEuroSAT attention modifications are active.

    Parameters
    ----------
    num_classes : int
        Number of output classes (default: 10 for EuroSAT).
    **kwargs
        Additional keyword arguments forwarded to :class:`EfficientEuroSATViT`.

    Returns
    -------
    EfficientEuroSATViT
        ViT-Tiny with all modifications disabled.
    """
    return EfficientEuroSATViT(
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        use_learned_temp=False,
        use_early_exit=False,
        use_learned_dropout=False,
        use_learned_residual=False,
        use_temp_annealing=False,
        **kwargs,
    )
