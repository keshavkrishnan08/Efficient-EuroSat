"""Uncertainty quantification baseline methods for comparison with UCAT.

Implements three standard uncertainty estimation approaches from the
deep learning literature:

    1. **MC Dropout** (Gal & Ghahramani, 2016) -- approximate Bayesian
       inference via stochastic forward passes with dropout active.
    2. **Deep Ensembles** (Lakshminarayanan et al., 2017) -- average
       softmax predictions across independently trained models.
    3. **Post-hoc Temperature Scaling** (Guo et al., 2017) -- learn a
       single scalar temperature on a held-out validation set to
       recalibrate logits.

Each function returns a results dictionary that is compatible with the
evaluation utilities in ``src.evaluation.calibration``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ======================================================================
# 1. MC Dropout
# ======================================================================

def mc_dropout_inference(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    n_forward: int = 30,
) -> Dict[str, Any]:
    """Estimate predictive uncertainty via Monte Carlo Dropout.

    The model is placed in *train* mode so that dropout layers remain
    active, then ``n_forward`` stochastic forward passes are performed
    for every batch.  Because Vision Transformers use LayerNorm (not
    BatchNorm), switching to train mode does not change normalisation
    statistics -- only dropout behaviour is affected.

    Parameters
    ----------
    model : nn.Module
        A trained model with dropout layers (e.g., ``BaselineViT`` or
        ``EfficientEuroSATViT``).
    dataloader : DataLoader
        Evaluation dataloader.
    device : str
        Device identifier (e.g., ``"cpu"`` or ``"cuda"``).
    n_forward : int
        Number of stochastic forward passes per batch (default: 30).

    Returns
    -------
    dict
        Keys:

        - ``accuracy`` : float -- top-1 accuracy from mean predictions.
        - ``predictions`` : numpy.ndarray -- predicted class per sample.
        - ``mean_probs`` : numpy.ndarray -- ``(N, C)`` mean softmax probs.
        - ``predictive_entropy`` : numpy.ndarray -- ``(N,)`` entropy of
          mean predictive distribution, H[E_q[p(y|x)]].
        - ``expected_entropy`` : numpy.ndarray -- ``(N,)`` mean entropy
          across stochastic passes, E_q[H[p(y|x)]].
        - ``mutual_information`` : numpy.ndarray -- ``(N,)`` epistemic
          uncertainty as MI = H[E[p]] - E[H[p]] (Gal & Ghahramani, 2016).
        - ``predictive_variance`` : numpy.ndarray -- ``(N,)`` mean variance
          of softmax predictions across passes (scalar per sample).
        - ``per_class_variance`` : numpy.ndarray -- ``(N, C)`` variance of
          softmax predictions per class across passes.
    """
    model.to(device)
    # Train mode enables dropout; LayerNorm is unaffected.
    model.train()

    # Disable early exit if the model supports it.
    if hasattr(model, "early_exit_enabled"):
        model.early_exit_enabled = False

    all_stacked: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)

            # Collect softmax predictions across N stochastic passes.
            stacked_probs = []
            for _ in range(n_forward):
                output = model(images)
                # Handle tuple outputs (e.g., EfficientEuroSATViT with
                # return_temperatures or early-exit metadata).
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output
                probs = F.softmax(logits, dim=-1)  # (B, C)
                stacked_probs.append(probs.cpu().numpy())

            # Stack: (n_forward, B, C) — keep full stochastic samples
            stacked = np.stack(stacked_probs, axis=0)

            all_stacked.append(stacked)
            all_labels.append(labels.numpy())

    # Concatenate across batches: (n_forward, N, C)
    stacked_all = np.concatenate(all_stacked, axis=1)
    labels_all = np.concatenate(all_labels, axis=0)  # (N,)

    # Mean predictive distribution: E_q[p(y|x)]
    mean_probs_all = stacked_all.mean(axis=0)  # (N, C)

    # --- Predictive entropy: H[E_q[p(y|x)]] ---
    log_mean = np.log(mean_probs_all + 1e-12)
    predictive_entropy = -np.sum(mean_probs_all * log_mean, axis=1)  # (N,)

    # --- Expected entropy: E_q[H[p(y|x)]] ---
    # Entropy of each stochastic pass, then average over passes
    log_stacked = np.log(stacked_all + 1e-12)  # (T, N, C)
    per_pass_entropy = -np.sum(stacked_all * log_stacked, axis=2)  # (T, N)
    expected_entropy = per_pass_entropy.mean(axis=0)  # (N,)

    # --- Mutual information (epistemic uncertainty) ---
    # MI = H[E[p]] - E[H[p]]  (Gal & Ghahramani, 2016; Smith & Gal, 2018)
    mutual_information = predictive_entropy - expected_entropy  # (N,)
    # Clamp to >= 0 (numerical precision)
    mutual_information = np.maximum(mutual_information, 0.0)

    # --- Predictive variance ---
    per_class_variance = stacked_all.var(axis=0)  # (N, C)
    predictive_variance = per_class_variance.mean(axis=1)  # (N,) scalar per sample

    predictions = mean_probs_all.argmax(axis=1)
    accuracy = float((predictions == labels_all).mean())

    return {
        "accuracy": accuracy,
        "predictions": predictions,
        "mean_probs": mean_probs_all,
        "predictive_entropy": predictive_entropy,
        "expected_entropy": expected_entropy,
        "mutual_information": mutual_information,
        "predictive_variance": predictive_variance,
        "per_class_variance": per_class_variance,
    }


# ======================================================================
# 2. Deep Ensembles
# ======================================================================

def _build_model_from_config(
    model_config: Dict[str, Any],
    device: str,
) -> nn.Module:
    """Reconstruct a model from a configuration dictionary.

    Parameters
    ----------
    model_config : dict
        Must contain ``model_type`` (``"efficient_eurosat"`` or
        ``"baseline"``).  Additional keys are forwarded to the
        respective constructor.
    device : str
        Target device.

    Returns
    -------
    nn.Module
        The constructed model on ``device``.
    """
    from src.models.efficient_vit import EfficientEuroSATViT
    from src.models.baseline import BaselineViT

    model_type = model_config.get("model_type", "efficient_eurosat")

    if model_type == "efficient_eurosat":
        model = EfficientEuroSATViT(
            img_size=model_config.get("img_size", 224),
            num_classes=model_config.get("num_classes", 10),
            use_learned_temp=model_config.get("use_learned_temp", True),
            use_early_exit=model_config.get("use_early_exit", True),
            use_learned_dropout=model_config.get("use_learned_dropout", True),
            use_learned_residual=model_config.get("use_learned_residual", True),
            use_temp_annealing=model_config.get("use_temp_annealing", True),
            use_decomposition=model_config.get("use_decomposition", False),
            tau_min=model_config.get("tau_min", 0.1),
            dropout_max=model_config.get("dropout_max", 0.3),
            exit_threshold=model_config.get("exit_threshold", 0.9),
            exit_min_layer=model_config.get("exit_min_layer", 4),
        )
    elif model_type == "baseline":
        model = BaselineViT(
            img_size=model_config.get("img_size", 224),
            num_classes=model_config.get("num_classes", 10),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)
    return model


def ensemble_inference(
    checkpoint_paths: List[str],
    model_config: Dict[str, Any],
    dataloader: DataLoader,
    device: str,
) -> Dict[str, Any]:
    """Estimate uncertainty via a Deep Ensemble.

    Loads each checkpoint independently, runs a full forward pass on the
    evaluation data, and averages the softmax outputs across all models.

    Parameters
    ----------
    checkpoint_paths : list of str
        Paths to the saved checkpoint files.  Each checkpoint is expected
        to contain ``"model_state_dict"`` and optionally
        ``"model_config"``.
    model_config : dict
        Default model configuration used to construct each ensemble
        member.  If a checkpoint contains its own ``model_config``, that
        takes precedence.
    dataloader : DataLoader
        Evaluation dataloader.
    device : str
        Device identifier.

    Returns
    -------
    dict
        Keys:

        - ``accuracy`` : float -- top-1 accuracy from averaged probs.
        - ``mean_probs`` : numpy.ndarray -- ``(N, C)`` averaged softmax.
        - ``predictive_entropy`` : numpy.ndarray -- ``(N,)`` entropy of
          mean predictive distribution.
        - ``mutual_information`` : numpy.ndarray -- ``(N,)`` epistemic
          uncertainty as MI = H[E[p]] - E[H[p]].
        - ``predictive_variance`` : numpy.ndarray -- ``(N,)`` mean variance
          of softmax predictions across ensemble members.
        - ``n_models`` : int -- number of ensemble members.
    """
    n_models = len(checkpoint_paths)
    if n_models == 0:
        raise ValueError("At least one checkpoint path is required.")

    all_member_probs: List[np.ndarray] = []

    for ckpt_path in checkpoint_paths:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

        # Use per-checkpoint config if available, else fall back.
        cfg = checkpoint.get("model_config", model_config)
        model = _build_model_from_config(cfg, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Disable early exit during evaluation.
        if hasattr(model, "early_exit_enabled"):
            model.early_exit_enabled = False

        member_probs: List[np.ndarray] = []
        with torch.no_grad():
            for images, _labels in dataloader:
                images = images.to(device)
                output = model(images)
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output
                probs = F.softmax(logits, dim=-1)
                member_probs.append(probs.cpu().numpy())

        all_member_probs.append(np.concatenate(member_probs, axis=0))

        # Free GPU memory between members.
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Collect labels in a final pass (all members see the same data).
    all_labels: List[np.ndarray] = []
    for _images, labels in dataloader:
        all_labels.append(labels.numpy())
    labels_all = np.concatenate(all_labels, axis=0)

    # Average across ensemble members: (M, N, C) -> (N, C).
    stacked = np.stack(all_member_probs, axis=0)
    mean_probs = stacked.mean(axis=0)

    # Predictive entropy: H[E[p]]
    log_mean = np.log(mean_probs + 1e-12)
    predictive_entropy = -np.sum(mean_probs * log_mean, axis=1)

    # Expected entropy: E[H[p]] (average entropy across members)
    log_stacked = np.log(stacked + 1e-12)  # (M, N, C)
    per_member_entropy = -np.sum(stacked * log_stacked, axis=2)  # (M, N)
    expected_entropy = per_member_entropy.mean(axis=0)  # (N,)

    # Mutual information (epistemic uncertainty)
    mutual_information = np.maximum(predictive_entropy - expected_entropy, 0.0)

    # Predictive variance across ensemble members
    per_class_variance = stacked.var(axis=0)  # (N, C)
    predictive_variance = per_class_variance.mean(axis=1)  # (N,)

    predictions = mean_probs.argmax(axis=1)
    accuracy = float((predictions == labels_all).mean())

    return {
        "accuracy": accuracy,
        "mean_probs": mean_probs,
        "predictive_entropy": predictive_entropy,
        "mutual_information": mutual_information,
        "predictive_variance": predictive_variance,
        "n_models": n_models,
    }


# ======================================================================
# 3. Post-hoc Temperature Scaling (Guo et al., 2017)
# ======================================================================

def _compute_ece_from_probs(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error from probability predictions.

    Uses the standard equal-width binning approach.

    Parameters
    ----------
    probs : numpy.ndarray
        Softmax probabilities of shape ``(N, C)``.
    labels : numpy.ndarray
        Ground-truth labels of shape ``(N,)``.
    n_bins : int
        Number of confidence bins.

    Returns
    -------
    float
        The ECE value.
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(confidences)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        count = mask.sum()
        if count > 0:
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += (count / max(n_total, 1)) * abs(bin_acc - bin_conf)

    return float(ece)


def posthoc_temperature_scaling(
    model: nn.Module,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
) -> Dict[str, Any]:
    """Learn a post-hoc temperature and evaluate calibration improvement.

    Following Guo et al. (2017), a single scalar temperature ``T`` is
    learned on a held-out validation set by minimising the negative
    log-likelihood (NLL) via L-BFGS.  The learned temperature is then
    applied to the test logits, and ECE is computed before and after
    scaling.

    Parameters
    ----------
    model : nn.Module
        A trained model in eval mode.
    val_loader : DataLoader
        Validation dataloader used to learn ``T``.
    test_loader : DataLoader
        Test dataloader used to evaluate calibration.
    device : str
        Device identifier.

    Returns
    -------
    dict
        Keys:

        - ``learned_temperature`` : float -- optimised temperature.
        - ``accuracy`` : float -- test accuracy (unchanged by scaling).
        - ``ece_before`` : float -- test ECE before temperature scaling.
        - ``ece_after`` : float -- test ECE after temperature scaling.
    """
    model.to(device)
    model.eval()

    # Disable early exit during logit collection.
    if hasattr(model, "early_exit_enabled"):
        model.early_exit_enabled = False

    # ------------------------------------------------------------------
    # Step 1: Collect validation logits and labels.
    # ------------------------------------------------------------------
    val_logits_list: List[torch.Tensor] = []
    val_labels_list: List[torch.Tensor] = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            output = model(images)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            val_logits_list.append(logits.cpu())
            val_labels_list.append(labels)

    val_logits = torch.cat(val_logits_list, dim=0)  # (N_val, C)
    val_labels = torch.cat(val_labels_list, dim=0)  # (N_val,)

    # ------------------------------------------------------------------
    # Step 2: Learn temperature T via L-BFGS on validation NLL.
    # ------------------------------------------------------------------
    temperature = nn.Parameter(torch.tensor(1.5))

    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def _closure() -> torch.Tensor:
        optimizer.zero_grad()
        scaled_logits = val_logits / temperature
        loss = F.cross_entropy(scaled_logits, val_labels)
        loss.backward()
        return loss

    optimizer.step(_closure)

    learned_temp = float(temperature.detach().item())

    # ------------------------------------------------------------------
    # Step 3: Collect test logits and labels.
    # ------------------------------------------------------------------
    test_logits_list: List[torch.Tensor] = []
    test_labels_list: List[torch.Tensor] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            output = model(images)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            test_logits_list.append(logits.cpu())
            test_labels_list.append(labels)

    test_logits = torch.cat(test_logits_list, dim=0)  # (N_test, C)
    test_labels = torch.cat(test_labels_list, dim=0)  # (N_test,)

    # ------------------------------------------------------------------
    # Step 4: Compute ECE before and after temperature scaling.
    # ------------------------------------------------------------------
    test_labels_np = test_labels.numpy()

    # Before scaling.
    probs_before = F.softmax(test_logits, dim=-1).numpy()
    ece_before = _compute_ece_from_probs(probs_before, test_labels_np, n_bins=15)

    # After scaling.
    scaled_test_logits = test_logits / learned_temp
    probs_after = F.softmax(scaled_test_logits, dim=-1).numpy()
    ece_after = _compute_ece_from_probs(probs_after, test_labels_np, n_bins=15)

    # Accuracy (identical before and after scaling).
    predictions = probs_before.argmax(axis=1)
    accuracy = float((predictions == test_labels_np).mean())

    return {
        "learned_temperature": learned_temp,
        "accuracy": accuracy,
        "ece_before": ece_before,
        "ece_after": ece_after,
    }
