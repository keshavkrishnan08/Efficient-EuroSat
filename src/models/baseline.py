"""
Baseline Vision Transformer (ViT-Tiny) for Satellite Land Use Classification.

A clean, standard ViT implementation with NO modifications:
- Standard multi-head self-attention with fixed temperature sqrt(d_k)
- Standard residual connections (alpha = 1.0)
- Fixed dropout rate
- No early exit mechanism
- No learned temperature scaling
- No temperature annealing

Used as the comparison baseline for the EfficientEuroSAT ablation study.
Architecture matches EfficientEuroSAT-Tiny: embed_dim=192, layers=12, heads=3.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineAttention(nn.Module):
    """
    Standard multi-head self-attention with fixed temperature scaling.

    Uses the conventional scaling factor of 1/sqrt(d_k) without any learned
    temperature parameters, learned dropout, or other modifications.
    """

    def __init__(self, embed_dim: int, num_heads: int, drop_rate: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(drop_rate)
        self.proj_drop = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, N, C) where B is batch size,
               N is sequence length, C is embed_dim.

        Returns:
            Output tensor of shape (B, N, C).
        """
        B, N, C = x.shape

        # Compute Q, K, V projections
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)  # Each: (B, heads, N, head_dim)

        # Standard scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) / self.scale  # (B, heads, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class BaselineTransformerBlock(nn.Module):
    """
    Standard Transformer encoder block with fixed residual connections.

    Contains multi-head self-attention followed by a feed-forward MLP,
    each with layer normalization and standard (alpha=1) residual connections.
    No early exit or learned residual gating.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = BaselineAttention(embed_dim, num_heads, drop_rate)

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(drop_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, N, C).

        Returns:
            Output tensor of shape (B, N, C).
        """
        # Standard pre-norm residual connection for attention
        x = x + self.attn(self.norm1(x))

        # Standard pre-norm residual connection for MLP
        x = x + self.mlp(self.norm2(x))

        return x


class BaselineViT(nn.Module):
    """
    Baseline Vision Transformer (ViT-Tiny) for satellite land use classification.

    A standard ViT implementation with no attention modifications, serving as
    the control model for ablation studies against EfficientEuroSAT.

    Architecture:
        - Patch embedding via convolution
        - Learnable positional embeddings
        - CLS token for classification
        - Standard Transformer encoder blocks
        - Linear classification head

    Default configuration matches ViT-Tiny:
        embed_dim=192, num_layers=12, num_heads=3
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 192,
        num_layers: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
    ):
        super().__init__()

        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # Patch embedding: project image patches to embedding dimension
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        # CLS token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        self.pos_drop = nn.Dropout(drop_rate)

        # Transformer encoder blocks
        self.blocks = nn.ModuleList(
            [
                BaselineTransformerBlock(embed_dim, num_heads, mlp_ratio, drop_rate)
                for _ in range(num_layers)
            ]
        )

        # Final layer norm and classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights following ViT conventions."""
        # Positional embedding: truncated normal
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Linear and LayerNorm layers
        self.apply(self._init_module_weights)

    @staticmethod
    def _init_module_weights(m: nn.Module):
        """Initialize weights for Linear and LayerNorm modules."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the baseline ViT.

        Args:
            x: Input image tensor of shape (B, C, H, W).

        Returns:
            Classification logits of shape (B, num_classes).
        """
        B = x.shape[0]

        # Patch embedding: (B, C, H, W) -> (B, num_patches, embed_dim)
        x = self.patch_embed(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches + 1, embed_dim)

        # Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Pass through all Transformer blocks (no early exit)
        for block in self.blocks:
            x = block(x)

        # Classification: use CLS token output
        x = self.norm(x[:, 0])  # (B, embed_dim)
        x = self.head(x)  # (B, num_classes)

        return x


def create_baseline_vit(
    num_classes: int = 10,
    img_size: int = 224,
    patch_size: int = 16,
) -> BaselineViT:
    """
    Factory function to create a baseline ViT-Tiny model.

    Creates a standard ViT with no attention modifications, matching the
    EfficientEuroSAT-Tiny architecture dimensions for fair comparison.

    Args:
        num_classes: Number of output classes (default: 10 for EuroSAT).
        img_size: Input image size (default: 224).
        patch_size: Patch size for patch embedding (default: 16).

    Returns:
        A BaselineViT model instance.
    """
    return BaselineViT(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=192,
        num_layers=12,
        num_heads=3,
        mlp_ratio=4.0,
        drop_rate=0.1,
    )
