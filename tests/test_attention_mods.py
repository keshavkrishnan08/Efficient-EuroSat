"""Tests for the five individual attention modifications.

Each modification from attention_modifications.py is tested in isolation
using small tensor dimensions for fast execution.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

from src.models.attention_modifications import (
    LearnableTemperature,
    EarlyExitController,
    LearnedHeadDropout,
    LearnedResidualWeight,
    TemperatureScheduler,
)

# ---------------------------------------------------------------------------
# Small test constants
# ---------------------------------------------------------------------------
NUM_HEADS = 2
BATCH_SIZE = 2
SEQ_LEN = 5
EMBED_DIM = 64


# ===========================================================================
# Modification 1 -- LearnableTemperature
# ===========================================================================


class TestLearnableTemperature:
    """Tests for per-head learnable temperature scaling."""

    def setup_method(self):
        self.lt = LearnableTemperature(num_heads=NUM_HEADS, tau_min=0.1)

    def test_learnable_temperature_output_shape(self):
        """tau shape is (num_heads,)."""
        tau = self.lt()
        assert tau.shape == (NUM_HEADS,), (
            f"Expected shape ({NUM_HEADS},), got {tau.shape}"
        )

    def test_learnable_temperature_positive(self):
        """All tau values must be strictly greater than tau_min."""
        tau = self.lt()
        assert (tau > self.lt.tau_min).all(), (
            f"Some tau values are not above tau_min={self.lt.tau_min}: {tau}"
        )

    def test_learnable_temperature_gradient(self):
        """Parameters receive gradients after backward pass."""
        tau = self.lt()
        loss = tau.sum()
        loss.backward()
        assert self.lt.raw_tau.grad is not None, "raw_tau did not receive gradients"
        assert self.lt.raw_tau.grad.shape == (NUM_HEADS,)


# ===========================================================================
# Modification 2 -- EarlyExitController
# ===========================================================================


class TestEarlyExitController:
    """Tests for confidence-based early exit."""

    def setup_method(self):
        self.ctrl = EarlyExitController(threshold=0.9, min_layers=4)

    def _make_attn_weights(self, confidence: float) -> torch.Tensor:
        """Create attention weights with a given approximate confidence.

        Confidence is measured as the mean of per-head max attention weight.
        We create a distribution where the max weight in each row equals
        ``confidence`` and the remaining mass is spread uniformly.
        """
        w = torch.zeros(BATCH_SIZE, NUM_HEADS, SEQ_LEN, SEQ_LEN)
        remaining = 1.0 - confidence
        w.fill_(remaining / (SEQ_LEN - 1))
        w[:, :, :, 0] = confidence
        return w

    def test_early_exit_training_mode(self):
        """should_exit returns False during training (requires_grad=True)."""
        w = self._make_attn_weights(0.99)
        w.requires_grad_(True)
        result = self.ctrl.should_exit(w, layer_idx=10)
        assert result is False, "should_exit must return False during training"

    def test_early_exit_low_confidence(self):
        """should_exit returns False for low confidence scores."""
        w = self._make_attn_weights(0.3)
        result = self.ctrl.should_exit(w, layer_idx=10)
        assert result is False, "should_exit should be False for low confidence"

    def test_early_exit_high_confidence(self):
        """should_exit returns True for high confidence above threshold."""
        w = self._make_attn_weights(0.95)
        result = self.ctrl.should_exit(w, layer_idx=10)
        assert result is True, (
            "should_exit should be True for confidence=0.95 > threshold=0.9"
        )

    def test_early_exit_min_layer(self):
        """should_exit returns False for layers below min_layers."""
        w = self._make_attn_weights(0.99)
        result = self.ctrl.should_exit(w, layer_idx=2)
        assert result is False, (
            "should_exit must return False when layer_idx < min_layers"
        )


# ===========================================================================
# Modification 3 -- LearnedHeadDropout
# ===========================================================================


class TestLearnedHeadDropout:
    """Tests for per-head learned dropout."""

    def setup_method(self):
        self.hd = LearnedHeadDropout(num_heads=NUM_HEADS, p_max=0.3)

    def test_learned_dropout_training(self):
        """Applies dropout during training: output may differ from input."""
        self.hd.train()
        w = torch.ones(BATCH_SIZE, NUM_HEADS, SEQ_LEN, SEQ_LEN)
        torch.manual_seed(42)
        out = self.hd(w)
        # With dropout, at least some values should be zeroed
        # Run multiple trials to handle stochasticity
        found_difference = False
        for seed in range(20):
            torch.manual_seed(seed)
            out = self.hd(torch.ones_like(w))
            if not torch.equal(out, w):
                found_difference = True
                break
        assert found_difference, (
            "LearnedHeadDropout in training mode should modify input "
            "(drop some values) at least sometimes"
        )

    def test_learned_dropout_eval(self):
        """No dropout during eval: output equals input exactly."""
        self.hd.eval()
        w = torch.ones(BATCH_SIZE, NUM_HEADS, SEQ_LEN, SEQ_LEN)
        out = self.hd(w)
        assert torch.equal(out, w), (
            "LearnedHeadDropout in eval mode must return input unchanged"
        )

    def test_learned_dropout_range(self):
        """Learned dropout rates are in [0, p_max]."""
        rates = self.hd._get_drop_probs()
        assert (rates >= 0.0).all(), f"Some dropout rates are negative: {rates}"
        assert (rates <= self.hd.p_max + 1e-6).all(), (
            f"Some dropout rates exceed p_max={self.hd.p_max}: {rates}"
        )


# ===========================================================================
# Modification 4 -- LearnedResidualWeight
# ===========================================================================


class TestLearnedResidualWeight:
    """Tests for learned residual mixing coefficient."""

    def setup_method(self):
        self.lrw = LearnedResidualWeight()

    def test_residual_weight_output_shape(self):
        """Output shape matches input shape."""
        attn_out = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)
        residual = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)
        out = self.lrw(attn_out, residual)
        assert out.shape == attn_out.shape, (
            f"Expected shape {attn_out.shape}, got {out.shape}"
        )

    def test_residual_weight_initialization(self):
        """Alpha should start at 0.5 (sigmoid(0) = 0.5)."""
        alpha = self.lrw.alpha.item()
        assert abs(alpha - 0.5) < 1e-5, (
            f"Expected alpha to start at 0.5, got {alpha}"
        )

    def test_residual_weight_range(self):
        """Alpha must always be in [0, 1]."""
        # Test with several raw_alpha values
        for val in [-10.0, -1.0, 0.0, 1.0, 10.0]:
            self.lrw.raw_alpha.data.fill_(val)
            alpha = self.lrw.alpha.item()
            assert 0.0 <= alpha <= 1.0, (
                f"Alpha {alpha} is out of [0, 1] for raw_alpha={val}"
            )


# ===========================================================================
# Modification 5 -- TemperatureScheduler
# ===========================================================================


class TestTemperatureScheduler:
    """Tests for temperature annealing schedule."""

    def setup_method(self):
        self.sched = TemperatureScheduler(
            tau_max_mult=1.5, tau_min_mult=1.0, power=2.0
        )

    def test_temperature_scheduler_start(self):
        """Multiplier starts at tau_max_mult."""
        mult = self.sched.get_multiplier(0.0)
        assert abs(mult - 1.5) < 1e-6, (
            f"Expected multiplier=1.5 at progress=0.0, got {mult}"
        )

    def test_temperature_scheduler_end(self):
        """Multiplier ends at tau_min_mult."""
        mult = self.sched.get_multiplier(1.0)
        assert abs(mult - 1.0) < 1e-6, (
            f"Expected multiplier=1.0 at progress=1.0, got {mult}"
        )

    def test_temperature_scheduler_monotonic(self):
        """Multiplier is non-increasing over training progress."""
        steps = [i / 100.0 for i in range(101)]
        multipliers = [self.sched.get_multiplier(p) for p in steps]
        for i in range(len(multipliers) - 1):
            assert multipliers[i] >= multipliers[i + 1] - 1e-9, (
                f"Multiplier increased from {multipliers[i]} to "
                f"{multipliers[i+1]} between progress={steps[i]} "
                f"and {steps[i+1]}"
            )
