"""Main training loop for EfficientEuroSAT.

Provides ``EuroSATTrainer``, a self-contained training class that handles:
- Mixed-precision training with ``torch.amp``
- Gradient clipping for training stability
- Cosine annealing LR schedule with linear warmup
- Label-smoothed cross-entropy with optional class weights
- Temperature annealing for learned attention temperatures
- Early stopping and model checkpointing
- Top-1 and Top-5 accuracy tracking
- Optional Weights & Biases logging
- Detailed per-class test evaluation with confusion matrix
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .callbacks import (
    EarlyStopping,
    MetricTracker,
    ModelCheckpoint,
)
from .losses import CombinedLoss, DecomposedCombinedLoss, LabelSmoothingCrossEntropy
from .schedulers import (
    CosineAnnealingWithWarmup,
    TemperatureAnnealingScheduler,
)

__all__ = ["EuroSATTrainer"]


def _topk_accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
    topk: tuple = (1, 5),
) -> List[float]:
    """Compute top-k accuracy for the given k values.

    Parameters
    ----------
    output : torch.Tensor
        Model logits of shape ``(N, C)``.
    target : torch.Tensor
        Ground-truth class indices of shape ``(N,)``.
    topk : tuple of int
        Which top-k accuracies to compute.

    Returns
    -------
    list of float
        Accuracy values (in ``[0, 1]``) for each k.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()  # (maxk, N)
        correct = pred.eq(target.unsqueeze(0).expand_as(pred))

        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            results.append(correct_k.item() / batch_size)
        return results


class EuroSATTrainer:
    """Complete training pipeline for EfficientEuroSAT models.

    Encapsulates the training loop, validation, testing, logging,
    checkpointing, and early stopping into a single reusable class.

    Parameters
    ----------
    model : nn.Module
        The EfficientEuroSAT model to train.
    train_loader : DataLoader
        DataLoader for the training set.
    val_loader : DataLoader
        DataLoader for the validation set.
    test_loader : DataLoader
        DataLoader for the test set.
    optimizer : torch.optim.Optimizer or None, optional
        Optimizer instance.  If ``None``, AdamW is created with
        ``lr=1e-4`` and ``weight_decay=0.05``.
    scheduler : LR scheduler or None, optional
        Learning rate scheduler.  If ``None``,
        ``CosineAnnealingWithWarmup`` is created.
    criterion : nn.Module or None, optional
        Loss function.  If ``None``,
        ``LabelSmoothingCrossEntropy`` is created.
    class_weights : torch.Tensor or None, optional
        Per-class weights for the loss function (shape ``(C,)``).
    device : str, optional
        Device to train on.  Default is ``'cuda'``.
    save_dir : str, optional
        Directory for saving checkpoints.  Default is ``'./checkpoints'``.
    use_amp : bool, optional
        Whether to use automatic mixed precision.  Default is ``True``.
    use_wandb : bool, optional
        Whether to log metrics to Weights & Biases.  Default is ``False``.
    wandb_project : str, optional
        W&B project name.  Default is ``'efficient_eurosat'``.
    wandb_run_name : str or None, optional
        W&B run name.  If ``None``, W&B auto-generates one.

    Examples
    --------
    >>> trainer = EuroSATTrainer(
    ...     model=model,
    ...     train_loader=train_dl,
    ...     val_loader=val_dl,
    ...     test_loader=test_dl,
    ...     device='cuda',
    ... )
    >>> best_metrics = trainer.train(num_epochs=100, patience=10)
    >>> test_results = trainer.test()
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        criterion: Optional[nn.Module] = None,
        class_weights: Optional[torch.Tensor] = None,
        device: str = "cuda",
        save_dir: str = "./checkpoints",
        use_amp: bool = True,
        use_wandb: bool = False,
        wandb_project: str = "efficient_eurosat",
        wandb_run_name: Optional[str] = None,
        lambda_ucat: float = 0.0,
        use_decomposition: bool = False,
        lambda_aleatoric: float = 0.05,
        lambda_epistemic: float = 0.05,
        blur_loss_frequency: int = 10,
        class_rarity: Optional[torch.Tensor] = None,
    ) -> None:
        # Normalise device to a plain type string (e.g. "cuda", "cpu")
        # so that torch.amp.autocast and GradScaler work with "cuda:0" etc.
        if isinstance(device, torch.device):
            self.device = device.type
        elif ":" in str(device):
            self.device = str(device).split(":")[0]
        else:
            self.device = str(device)

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.save_dir = Path(save_dir)
        self.use_amp = use_amp
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.lambda_ucat = lambda_ucat
        self.class_weights = class_weights
        self.use_decomposition = use_decomposition
        self.lambda_aleatoric = lambda_aleatoric
        self.lambda_epistemic = lambda_epistemic
        self.blur_loss_frequency = blur_loss_frequency
        self.class_rarity = class_rarity
        self.temperature_dynamics: List[Dict[str, float]] = []

        # Set up components (use provided or create defaults).
        self.optimizer = optimizer or self._setup_optimizer()
        self.criterion = criterion or self._setup_criterion()
        self.scheduler = scheduler  # set after optimizer is ready
        self._scheduler_provided = scheduler is not None

        # Mixed precision scaler (only useful on CUDA).
        self.scaler = torch.amp.GradScaler(self.device) if use_amp else None

        # Temperature annealing scheduler (configured when training begins).
        self.temp_scheduler: Optional[TemperatureAnnealingScheduler] = None

        # Callbacks and tracking.
        self.metric_tracker = MetricTracker()
        self.checkpoint = ModelCheckpoint(
            save_dir=str(self.save_dir),
            metric_name="val_acc",
            mode="max",
        )

        # Build model config dict for checkpoint reconstruction.
        self._model_config = self._extract_model_config()

        # Gradient clipping max norm.
        self._max_grad_norm: float = 1.0

        # W&B initialisation (deferred to train() to avoid import
        # overhead if wandb is not used).
        self._wandb_run = None

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _extract_model_config(self) -> dict:
        """Extract model configuration for checkpoint reconstruction."""
        m = self.model
        config: Dict[str, Any] = {}
        # Detect model type.
        cls_name = type(m).__name__
        if cls_name == "BaselineViT":
            config["model_type"] = "baseline"
        else:
            config["model_type"] = "efficient_eurosat"
        # Pull constructor-visible attributes when available.
        for attr in (
            "img_size", "num_classes", "use_learned_temp",
            "use_early_exit", "use_learned_dropout",
            "use_learned_residual", "use_temp_annealing",
            "tau_min", "dropout_max", "exit_threshold",
            "exit_min_layer", "use_decomposition",
        ):
            if hasattr(m, attr):
                config[attr] = getattr(m, attr)
        config["lambda_ucat"] = self.lambda_ucat
        return config

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Create a default AdamW optimizer.

        Returns
        -------
        torch.optim.AdamW
            Optimizer with ``lr=1e-4`` and ``weight_decay=0.05``.
        """
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=0.05,
        )

    def _setup_scheduler(self, num_epochs: int) -> Any:
        """Create a default cosine annealing scheduler with warmup.

        Parameters
        ----------
        num_epochs : int
            Total number of training epochs.

        Returns
        -------
        CosineAnnealingWithWarmup
            LR scheduler instance.
        """
        warmup_epochs = max(1, num_epochs // 10)
        return CosineAnnealingWithWarmup(
            optimizer=self.optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=num_epochs,
            min_lr=1e-6,
        )

    def _setup_criterion(self) -> nn.Module:
        """Create the appropriate loss function."""
        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(self.device)

        if self.use_decomposition:
            rarity = self.class_rarity.to(self.device) if self.class_rarity is not None else None
            return DecomposedCombinedLoss(
                lambda_ucat=self.lambda_ucat,
                lambda_aleatoric=self.lambda_aleatoric,
                lambda_epistemic=self.lambda_epistemic,
                label_smoothing=0.1,
                weight=weight,
                class_rarity=rarity,
            )
        elif self.lambda_ucat > 0:
            return CombinedLoss(
                lambda_ucat=self.lambda_ucat,
                label_smoothing=0.1,
                weight=weight,
            )
        return LabelSmoothingCrossEntropy(smoothing=0.1, weight=weight)

    def _init_wandb(self) -> None:
        """Initialise a Weights & Biases run if enabled."""
        if not self.use_wandb:
            return
        try:
            import wandb

            self._wandb_run = wandb.init(
                project=self.wandb_project,
                name=self.wandb_run_name,
                config={
                    "optimizer": self.optimizer.__class__.__name__,
                    "lr": self.optimizer.defaults.get("lr"),
                    "weight_decay": self.optimizer.defaults.get("weight_decay"),
                    "use_amp": self.use_amp,
                    "max_grad_norm": self._max_grad_norm,
                },
            )
        except ImportError:
            print(
                "[EuroSATTrainer] wandb is not installed. "
                "Falling back to console logging only."
            )
            self.use_wandb = False

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log metrics to the console and optionally to W&B.

        Parameters
        ----------
        metrics : dict
            Mapping of metric names to values.
        step : int
            Global step or epoch number for the x-axis.
        """
        parts = [f"{k}: {v:.4f}" for k, v in metrics.items()]
        print(f"[Step {step}] " + " | ".join(parts))

        if self.use_wandb and self._wandb_run is not None:
            import wandb

            wandb.log(metrics, step=step)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        num_epochs: int,
        patience: int = 10,
    ) -> Dict[str, float]:
        """Run the full training loop.

        Parameters
        ----------
        num_epochs : int
            Maximum number of training epochs.
        patience : int, optional
            Early stopping patience (epochs without improvement).
            Default is ``10``.

        Returns
        -------
        dict
            Best metrics achieved during training, including
            ``'best_val_acc'``, ``'best_val_loss'``, and
            ``'best_epoch'``.
        """
        # Set up scheduler if not provided externally.
        if not self._scheduler_provided:
            self.scheduler = self._setup_scheduler(num_epochs)

        # Temperature annealing across all training steps.
        total_steps = num_epochs * len(self.train_loader)
        self.temp_scheduler = TemperatureAnnealingScheduler(
            tau_max_mult=1.5,
            tau_min_mult=1.0,
            power=2.0,
            total_steps=total_steps,
        )

        # Callbacks.
        early_stopping = EarlyStopping(patience=patience, min_delta=0.001, mode="max")

        # W&B.
        self._init_wandb()

        # Ensure save directory exists.
        self.save_dir.mkdir(parents=True, exist_ok=True)

        best_val_acc: float = 0.0
        best_epoch: int = 0

        print(f"[EuroSATTrainer] Starting training for {num_epochs} epochs")
        print(f"[EuroSATTrainer] Device: {self.device} | AMP: {self.use_amp}")
        print(f"[EuroSATTrainer] Train batches: {len(self.train_loader)} | "
              f"Val batches: {len(self.val_loader)}")
        print("-" * 70)

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            # --- Train ---
            train_metrics = self.train_one_epoch(epoch)

            # --- Validate ---
            val_metrics = self.validate()

            # --- LR scheduler step ---
            if self.scheduler is not None:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            # Combine metrics for logging.
            combined = {
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **val_metrics,
                "lr": current_lr,
                "temp_mult": self.temp_scheduler.get_current_multiplier(),
            }

            # Track metrics.
            for name, value in combined.items():
                self.metric_tracker.update(name, value)

            epoch_time = time.time() - epoch_start
            combined["epoch_time_s"] = epoch_time

            self._log_metrics(combined, step=epoch)

            # --- Checkpointing ---
            val_acc = val_metrics["val_acc"]
            self.checkpoint(self.model, val_acc, epoch, model_config=self._model_config)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch

            # --- Early stopping ---
            if early_stopping(val_acc):
                print(
                    f"[EuroSATTrainer] Early stopping triggered at epoch {epoch}. "
                    f"Best val_acc: {best_val_acc:.4f} at epoch {best_epoch}."
                )
                break

        print("-" * 70)
        print(
            f"[EuroSATTrainer] Training complete. "
            f"Best val_acc: {best_val_acc:.4f} at epoch {best_epoch}."
        )

        # Finish W&B run.
        if self.use_wandb and self._wandb_run is not None:
            import wandb

            wandb.finish()

        result = {
            "best_val_acc": best_val_acc,
            "best_val_loss": self.metric_tracker.get_best("val_loss", mode="min") or 0.0,
            "best_epoch": best_epoch,
        }
        if self.temperature_dynamics:
            result["temperature_dynamics"] = self.temperature_dynamics
        return result

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number (1-based, for logging only).

        Returns
        -------
        dict
            Training metrics for this epoch.
        """
        self.model.train()

        running_loss: float = 0.0
        correct: int = 0
        total: int = 0

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Compute training progress for temperature annealing.
            training_progress: float = 0.0
            if self.temp_scheduler is not None:
                training_progress = self.temp_scheduler.get_progress()

            # Forward pass with AMP.
            with torch.amp.autocast(self.device, enabled=self.use_amp):
                if self.use_decomposition:
                    loss, outputs = self._decomposed_forward(
                        images, targets, training_progress, batch_idx
                    )
                elif self.lambda_ucat > 0:
                    outputs, temperatures = self.model(
                        images,
                        training_progress=training_progress,
                        return_temperatures=True,
                    )
                    loss, task_loss, ucat_loss = self.criterion(
                        outputs, targets, temperatures
                    )
                else:
                    outputs = self.model(
                        images, training_progress=training_progress
                    )
                    loss = self.criterion(outputs, targets)

            # Backward pass.
            self.optimizer.zero_grad(set_to_none=True)

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self._max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self._max_grad_norm
                )
                self.optimizer.step()

            # Step the temperature scheduler.
            if self.temp_scheduler is not None:
                self.temp_scheduler.step()

            # Accumulate statistics.
            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            _, predicted = outputs.max(dim=1)
            correct += predicted.eq(targets).sum().item()
            total += batch_size

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)

        # Track temperature dynamics for decomposed models
        if self.use_decomposition and hasattr(self.model, '_collect_epistemic_temperatures'):
            with torch.no_grad():
                tau_e = self.model._collect_epistemic_temperatures()
                tau_e_val = tau_e.item() if tau_e is not None else 0.0
                self.temperature_dynamics.append({
                    "epoch": epoch,
                    "mean_tau_e": tau_e_val,
                })

        return {"loss": epoch_loss, "acc": epoch_acc}

    def _decomposed_forward(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        training_progress: float,
        batch_idx: int,
    ) -> tuple:
        """Forward pass with decomposed UCAT loss computation."""
        from ..data.augmentation_pairs import generate_augmentation_pair
        from ..data.blur import apply_all_blur_levels

        # Main forward with decomposition
        result = self.model(
            images,
            training_progress=training_progress,
            return_temperatures=True,
        )
        # Decomposed returns: (logits, tau_total, tau_a_mean, tau_e_mean)
        outputs, temperatures, tau_a_mean, tau_e_mean = result

        # Augmented forward for aleatoric consistency
        augmented = generate_augmentation_pair(images)
        result_aug = self.model(
            augmented,
            training_progress=training_progress,
            return_temperatures=True,
        )
        _, _, tau_a_aug, _ = result_aug

        # Blur forward (every N batches)
        tau_a_blur_vals = None
        blur_lvls = None
        if batch_idx % self.blur_loss_frequency == 0:
            blurred_all, blur_lvls = apply_all_blur_levels(images)
            with torch.no_grad():
                # Only need tau_a, don't need gradients for blur labels
                pass
            result_blur = self.model(
                blurred_all,
                training_progress=training_progress,
                return_temperatures=True,
            )
            _, _, tau_a_blur_vals, _ = result_blur
            # tau_a_blur_vals is mean across heads per sample

        loss, l_task, l_ucat, l_aleatoric, l_epistemic = self.criterion(
            logits=outputs,
            labels=targets,
            temperatures=temperatures,
            tau_a_original=tau_a_mean,
            tau_a_augmented=tau_a_aug,
            tau_e=tau_e_mean.expand(targets.shape[0]) if tau_e_mean.dim() == 0 else tau_e_mean,
            training_progress=training_progress,
            tau_a_blur_values=tau_a_blur_vals,
            blur_levels=blur_lvls,
        )

        return loss, outputs

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Evaluate the model on the validation set.

        Returns
        -------
        dict
            Validation metrics: ``'val_loss'``, ``'val_acc'``,
            ``'val_top5_acc'``.
        """
        self.model.eval()

        running_loss: float = 0.0
        correct: int = 0
        correct_top5: int = 0
        total: int = 0

        for images, targets in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with torch.amp.autocast(self.device, enabled=self.use_amp):
                outputs = self.model(images)
                if self.lambda_ucat > 0:
                    loss, _, _ = self.criterion(outputs, targets)
                else:
                    loss = self.criterion(outputs, targets)

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size

            top1, top5 = _topk_accuracy(outputs, targets, topk=(1, 5))
            correct += int(top1 * batch_size)
            correct_top5 += int(top5 * batch_size)
            total += batch_size

        val_loss = running_loss / max(total, 1)
        val_acc = correct / max(total, 1)
        val_top5_acc = correct_top5 / max(total, 1)

        return {
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_top5_acc": val_top5_acc,
        }

    # ------------------------------------------------------------------
    # Testing
    # ------------------------------------------------------------------

    @torch.no_grad()
    def test(self) -> Dict[str, Any]:
        """Evaluate the model on the test set with detailed metrics.

        Computes overall accuracy, per-class accuracy, and a full
        confusion matrix.

        Returns
        -------
        dict
            ``'test_acc'`` : float
                Overall top-1 accuracy on the test set.
            ``'test_top5_acc'`` : float
                Overall top-5 accuracy on the test set.
            ``'per_class_acc'`` : numpy.ndarray
                Per-class accuracy array of shape ``(C,)``.
            ``'confusion_matrix'`` : numpy.ndarray
                Confusion matrix of shape ``(C, C)`` where entry
                ``[i, j]`` is the number of samples with true class
                ``i`` predicted as class ``j``.
        """
        self.model.eval()

        all_preds: List[int] = []
        all_targets: List[int] = []
        correct_top5: int = 0
        total: int = 0

        for images, targets in self.test_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with torch.amp.autocast(self.device, enabled=self.use_amp):
                outputs = self.model(images)

            _, predicted = outputs.max(dim=1)
            all_preds.extend(predicted.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

            _, top5 = _topk_accuracy(outputs, targets, topk=(1, 5))
            batch_size = targets.size(0)
            correct_top5 += int(top5 * batch_size)
            total += batch_size

        all_preds_np = np.array(all_preds)
        all_targets_np = np.array(all_targets)

        # Overall accuracy.
        test_acc = float((all_preds_np == all_targets_np).mean())
        test_top5_acc = correct_top5 / max(total, 1)

        # Determine number of classes.
        num_classes = max(all_targets_np.max(), all_preds_np.max()) + 1

        # Confusion matrix: row = true class, col = predicted class.
        confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        for true, pred in zip(all_targets_np, all_preds_np):
            confusion[true, pred] += 1

        # Per-class accuracy.
        per_class_totals = confusion.sum(axis=1)
        per_class_correct = np.diag(confusion)
        # Avoid division by zero for classes with no test samples.
        per_class_acc = np.where(
            per_class_totals > 0,
            per_class_correct / per_class_totals,
            0.0,
        )

        # Log summary.
        print(f"[EuroSATTrainer] Test accuracy: {test_acc:.4f}")
        print(f"[EuroSATTrainer] Test top-5 accuracy: {test_top5_acc:.4f}")
        print(
            f"[EuroSATTrainer] Per-class accuracy: "
            f"mean={per_class_acc.mean():.4f}, "
            f"min={per_class_acc.min():.4f} (class {per_class_acc.argmin()}), "
            f"max={per_class_acc.max():.4f} (class {per_class_acc.argmax()})"
        )

        return {
            "test_acc": test_acc,
            "test_top5_acc": test_top5_acc,
            "per_class_acc": per_class_acc,
            "confusion_matrix": confusion,
        }
