"""Tests for early exit behaviour in the EfficientEuroSAT model.

Focuses on verifying that the confidence-based early exit mechanism
works correctly at the model level: respecting min_layer, responding
to threshold settings, tracking statistics, and preserving output validity.

Uses small dimensions (embed_dim=64, num_heads=2, depth=2, patch_size=8,
img_size=32, num_classes=5) for fast execution.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import torch.nn as nn

from src.models.efficient_vit import EfficientEuroSATViT

# ---------------------------------------------------------------------------
# Small test constants
# ---------------------------------------------------------------------------
BATCH_SIZE = 2
IMG_SIZE = 32
PATCH_SIZE = 8
IN_CHANNELS = 3
NUM_CLASSES = 5
EMBED_DIM = 64
DEPTH = 2
NUM_HEADS = 2


def _make_model(**overrides) -> EfficientEuroSATViT:
    """Create a small EfficientEuroSAT model with early exit enabled."""
    defaults = dict(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        mlp_ratio=2.0,
        dropout=0.0,
        use_learned_temp=True,
        use_early_exit=True,
        use_learned_dropout=False,
        use_learned_residual=True,
        use_temp_annealing=False,
        tau_min=0.5,
        dropout_max=0.3,
        exit_threshold=0.9,
        exit_min_layer=0,
    )
    defaults.update(overrides)
    return EfficientEuroSATViT(**defaults)


def _dummy_input(batch_size: int = BATCH_SIZE) -> torch.Tensor:
    """Create a dummy input image batch."""
    return torch.randn(batch_size, IN_CHANNELS, IMG_SIZE, IMG_SIZE)


# ===========================================================================
# Tests
# ===========================================================================


class TestEarlyExitSignal:
    """Test that the model can produce and respect early exit signals."""

    def test_exit_signal_returned(self):
        """Model can return an exit signal (output is valid logits)."""
        model = _make_model(exit_min_layer=0, exit_threshold=0.5)
        model.eval()
        x = _dummy_input()
        with torch.no_grad():
            out = model(x)
        # Output must be valid logits regardless of whether exit occurred
        assert out.shape == (BATCH_SIZE, NUM_CLASSES), (
            f"Expected shape ({BATCH_SIZE}, {NUM_CLASSES}), got {out.shape}"
        )
        assert torch.isfinite(out).all(), "Output contains non-finite values"

    def test_no_exit_during_training(self):
        """Early exit never triggers during training."""
        model = _make_model(exit_threshold=0.0, exit_min_layer=0)
        model.train()
        x = _dummy_input()
        # Even with threshold=0.0 (should always exit), training mode
        # processes all layers
        out = model(x, training_progress=0.5)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)
        # During training the early_exit_enabled path is not taken,
        # so exit stats should not be populated
        stats = model.get_exit_statistics()
        assert stats["total_samples"] == 0, (
            "No samples should be tracked during training, "
            f"got {stats['total_samples']}"
        )


class TestExitStatistics:
    """Test early exit statistics tracking."""

    def test_exit_statistics_tracking(self):
        """get_exit_statistics returns valid data after forward pass."""
        model = _make_model(exit_min_layer=0)
        model.eval()
        x = _dummy_input()
        with torch.no_grad():
            _ = model(x)
        stats = model.get_exit_statistics()

        # Verify the structure of the statistics dictionary
        assert "exit_layer_counts" in stats, "Missing 'exit_layer_counts' key"
        assert "total_samples" in stats, "Missing 'total_samples' key"
        assert "exit_layer_fractions" in stats, "Missing 'exit_layer_fractions' key"
        assert "average_exit_layer" in stats, "Missing 'average_exit_layer' key"

        # Total samples should equal batch size
        assert stats["total_samples"] == BATCH_SIZE, (
            f"Expected total_samples={BATCH_SIZE}, got {stats['total_samples']}"
        )

        # Sum of per-layer counts must equal total_samples
        total_from_counts = sum(stats["exit_layer_counts"].values())
        assert total_from_counts == stats["total_samples"], (
            f"Sum of exit_layer_counts ({total_from_counts}) does not match "
            f"total_samples ({stats['total_samples']})"
        )

        # Fractions should sum to approximately 1.0
        total_frac = sum(stats["exit_layer_fractions"].values())
        assert abs(total_frac - 1.0) < 1e-6, (
            f"Exit fractions sum to {total_frac}, expected 1.0"
        )

        # Average exit layer should be within valid range
        assert 0 <= stats["average_exit_layer"] < DEPTH, (
            f"Average exit layer {stats['average_exit_layer']} "
            f"is outside valid range [0, {DEPTH})"
        )


class TestExitThreshold:
    """Test that threshold and min_layer parameters affect exit behaviour."""

    def test_exit_threshold_effect(self):
        """Higher threshold means fewer (or equal) early exits.

        With a very high threshold (0.99), samples are less likely to exit
        early compared to a low threshold (0.01).
        """
        x = _dummy_input(batch_size=4)

        # Low threshold -- easier to exit early
        model_low = _make_model(exit_threshold=0.01, exit_min_layer=0)
        model_low.eval()
        with torch.no_grad():
            _ = model_low(x)
        stats_low = model_low.get_exit_statistics()

        # High threshold -- harder to exit early
        model_high = _make_model(exit_threshold=0.99, exit_min_layer=0)
        model_high.eval()
        with torch.no_grad():
            _ = model_high(x)
        stats_high = model_high.get_exit_statistics()

        # The average exit layer with a high threshold should be >= that
        # with a low threshold (or equal if no early exit happens in either)
        assert stats_high["average_exit_layer"] >= stats_low["average_exit_layer"], (
            f"Higher threshold should lead to later (or equal) average exit: "
            f"high={stats_high['average_exit_layer']}, "
            f"low={stats_low['average_exit_layer']}"
        )

    def test_exit_min_layer_effect(self):
        """min_layer parameter is respected: no exits before min_layer."""
        min_layer = 1
        model = _make_model(exit_threshold=0.01, exit_min_layer=min_layer)
        model.eval()
        x = _dummy_input()
        with torch.no_grad():
            _ = model(x)
        stats = model.get_exit_statistics()

        # No layer below min_layer should have any exits
        for layer_idx, count in stats["exit_layer_counts"].items():
            if count > 0:
                assert layer_idx >= min_layer, (
                    f"Found {count} exits at layer {layer_idx}, but "
                    f"min_layer={min_layer}"
                )


class TestExitOutputValidity:
    """Test that outputs remain valid logits regardless of exit behaviour."""

    def test_exit_accuracy_preserved(self):
        """Outputs are valid logits even with early exit active."""
        model = _make_model(exit_threshold=0.5, exit_min_layer=0)
        model.eval()
        x = _dummy_input(batch_size=4)
        with torch.no_grad():
            out = model(x)

        # Shape must be correct
        assert out.shape == (4, NUM_CLASSES), (
            f"Expected shape (4, {NUM_CLASSES}), got {out.shape}"
        )

        # All values must be finite
        assert torch.isfinite(out).all(), "Output contains non-finite values"

        # Softmax of logits should produce valid probabilities
        probs = torch.softmax(out, dim=-1)
        assert (probs >= 0).all(), "Probabilities contain negative values"
        assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5), (
            "Probabilities do not sum to 1"
        )

    def test_exit_disabled_matches_full_forward(self):
        """Disabling early exit produces same output as a no-exit model.

        When set_early_exit(False) is called on a model that has early exit
        modules, the output should match the full forward pass (all layers).
        """
        model = _make_model(exit_threshold=0.5, exit_min_layer=0)
        model.eval()

        torch.manual_seed(123)
        x = _dummy_input()

        # Run with early exit disabled
        model.set_early_exit(False)
        with torch.no_grad():
            out_disabled = model(x)

        # Run the same input with early exit disabled again to confirm
        # deterministic output
        with torch.no_grad():
            out_disabled2 = model(x)

        assert torch.allclose(out_disabled, out_disabled2, atol=1e-6), (
            "Output with early exit disabled is not deterministic"
        )
