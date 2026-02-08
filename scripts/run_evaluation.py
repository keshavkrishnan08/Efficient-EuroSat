#!/usr/bin/env python3
"""
Master evaluation script for EfficientEuroSAT ablation study (A1: Accuracy Evaluation).

Loads all 10 trained model checkpoints, evaluates each on the EuroSAT test set,
and produces:
    - Per-experiment test accuracy, test loss, and per-class accuracy
    - Seed-averaged accuracy with mean +/- std for the three all_combined runs
    - A formatted console summary table
    - A JSON file with the full results
    - A CSV summary file
    - An ablation bar chart (Fig 7) saved as ablation_accuracy.png

The 10 experiments mirror those defined in run_experiments.py:
    1. baseline           -- Baseline ViT (no modifications)
    2. ucat_only          -- +UCAT loss only
    3. early_exit_only    -- +Early Exit only
    4. dropout_only       -- +Learned Dropout only
    5. residual_only      -- +Learned Residual only
    6. annealing_only     -- +Temperature Annealing only
    7. all_combined_s42   -- All modifications combined, seed 42
    8. all_no_ucat        -- All modifications minus UCAT, seed 42
    9. all_combined_s123  -- All modifications combined, seed 123
   10. all_combined_s456  -- All modifications combined, seed 456

Usage:
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --checkpoints_dir ./checkpoints --batch_size 128
    python scripts/run_evaluation.py --device cpu
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import csv
import json
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.data.eurosat import get_eurosat_dataloaders, EUROSAT_CLASS_NAMES
from src.models.efficient_vit import EfficientEuroSATViT
from src.models.baseline import BaselineViT
from src.utils.helpers import set_seed, get_device, count_parameters


# ---------------------------------------------------------------------------
# Experiment names (must match run_experiments.py)
# ---------------------------------------------------------------------------

EXPERIMENT_NAMES = [
    "baseline",
    "ucat_only",
    "early_exit_only",
    "dropout_only",
    "residual_only",
    "annealing_only",
    "all_combined_s42",
    "all_no_ucat",
    "all_combined_s123",
    "all_combined_s456",
]

# Seed experiments used for mean +/- std computation
SEED_EXPERIMENTS = [
    "all_combined_s42",
    "all_combined_s123",
    "all_combined_s456",
]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Master evaluation: load all trained checkpoints and produce "
                    "accuracy results + ablation bar chart (Fig 7).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoints_dir", type=str, default="./checkpoints",
        help="Root directory containing per-experiment checkpoint folders.",
    )
    parser.add_argument(
        "--data_root", type=str, default="./data",
        help="Root directory for the EuroSAT dataset.",
    )
    parser.add_argument(
        "--save_dir", type=str, default="./analysis_results/accuracy",
        help="Directory to save evaluation outputs (JSON, CSV, figures).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Compute device (e.g. 'cpu', 'cuda', 'mps'). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size for test-set evaluation.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (used for reproducible data splits).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_from_checkpoint(checkpoint_path, device):
    """Reconstruct a model from a saved checkpoint and load its weights.

    Parameters
    ----------
    checkpoint_path : str
        Path to ``best_model_val_acc.pt``.
    device : torch.device
        Target device.

    Returns
    -------
    tuple
        ``(model, model_type, model_config, checkpoint)`` where *model* is
        placed on *device* in eval mode.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_config = checkpoint.get("model_config", {})
    model_type = model_config.get("model_type", "efficient_eurosat")

    if model_type == "baseline":
        model = BaselineViT(
            img_size=model_config.get("img_size", 224),
            num_classes=model_config.get("num_classes", 10),
        )
    else:
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

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, model_type, model_config, checkpoint


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, test_loader, device, num_classes=10):
    """Evaluate a model on the test set.

    Returns
    -------
    dict
        Dictionary with keys: ``test_accuracy``, ``test_loss``,
        ``per_class_accuracy`` (list of floats), ``num_samples``,
        ``num_correct``.
    """
    all_preds = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # Handle potential dict output from EfficientEuroSATViT
            if isinstance(outputs, dict):
                logits = outputs["logits"]
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()
            num_batches += 1

            preds = logits.argmax(dim=-1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    avg_loss = total_loss / max(num_batches, 1)

    overall_acc = float(np.mean(all_preds == all_labels))

    per_class_acc = []
    for c in range(num_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            per_class_acc.append(float(np.mean(all_preds[mask] == c)))
        else:
            per_class_acc.append(0.0)

    return {
        "test_accuracy": overall_acc,
        "test_loss": avg_loss,
        "per_class_accuracy": per_class_acc,
        "num_samples": int(len(all_labels)),
        "num_correct": int(np.sum(all_preds == all_labels)),
    }


# ---------------------------------------------------------------------------
# Ablation bar chart (Fig 7)
# ---------------------------------------------------------------------------

def _bar_color(name):
    """Return a colour for the ablation bar chart based on experiment name."""
    if name == "baseline":
        return "#888888"           # gray
    if name in ("ucat_only", "early_exit_only", "dropout_only",
                "residual_only", "annealing_only"):
        return "#4682B4"           # steel blue (individual modifications)
    if name == "all_no_ucat":
        return "#E8850C"           # orange
    if name == "all_combined_s42":
        return "#2E8B57"           # sea green
    if name == "all_combined_s123":
        return "#3CB371"           # medium sea green
    if name == "all_combined_s456":
        return "#66CDAA"           # medium aquamarine
    return "#4682B4"


def generate_ablation_chart(results, save_path, baseline_accuracy=None):
    """Generate and save the ablation bar chart (Fig 7).

    Parameters
    ----------
    results : list of dict
        Each dict must contain ``name`` and ``test_accuracy``.
    save_path : str
        File path to save the PNG figure.
    baseline_accuracy : float or None
        If provided, draw a horizontal dashed line at this accuracy.
    """
    names = [r["name"] for r in results]
    accuracies = [r["test_accuracy"] * 100.0 for r in results]
    colors = [_bar_color(n) for n in names]

    fig, ax = plt.subplots(figsize=(14, 6))

    bars = ax.bar(range(len(names)), accuracies, color=colors, edgecolor="black",
                  linewidth=0.6, width=0.7)

    # Add value labels on each bar
    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.3,
            f"{acc:.1f}%",
            ha="center", va="bottom", fontsize=8, fontweight="bold",
        )

    # Baseline reference line
    if baseline_accuracy is not None:
        ax.axhline(
            y=baseline_accuracy * 100.0,
            color="#888888", linestyle="--", linewidth=1.2, alpha=0.7,
            label=f"Baseline ({baseline_accuracy * 100.0:.1f}%)",
        )
        ax.legend(loc="lower right", fontsize=9)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax.set_title("Ablation Study: Test Accuracy by Configuration", fontsize=13,
                 fontweight="bold")

    # Y-axis range: show context around the accuracies
    if accuracies:
        y_min = max(0, min(accuracies) - 5)
        y_max = min(100, max(accuracies) + 3)
        ax.set_ylim(y_min, y_max)

    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

def print_results_table(results, seed_stats=None):
    """Print a formatted summary table to stdout.

    Parameters
    ----------
    results : list of dict
        Per-experiment results (must have ``name``, ``test_accuracy``,
        ``test_loss``, ``status``).
    seed_stats : dict or None
        If provided, contains ``mean``, ``std``, ``n`` for seed experiments.
    """
    width = 90
    print()
    print("=" * width)
    print("  A1: ACCURACY EVALUATION RESULTS")
    print("=" * width)

    header = (
        f"  {'#':<4} {'Experiment':<24} {'Status':<10} "
        f"{'Test Acc (%)':<14} {'Test Loss':<12}"
    )
    print(header)
    print("  " + "-" * (width - 4))

    for idx, r in enumerate(results, 1):
        status = r.get("status", "OK")
        if status == "OK":
            acc_str = f"{r['test_accuracy'] * 100.0:.2f}"
            loss_str = f"{r['test_loss']:.4f}"
        else:
            acc_str = "--"
            loss_str = "--"
        print(
            f"  {idx:<4} {r['name']:<24} {status:<10} "
            f"{acc_str:<14} {loss_str:<12}"
        )

    print("  " + "-" * (width - 4))

    if seed_stats is not None:
        print(
            f"  Seed experiments (all_combined, n={seed_stats['n']}):  "
            f"mean = {seed_stats['mean'] * 100.0:.2f}%  "
            f"std = {seed_stats['std'] * 100.0:.2f}%"
        )

    evaluated = sum(1 for r in results if r.get("status") == "OK")
    skipped = sum(1 for r in results if r.get("status") == "SKIPPED")
    print(f"  Evaluated: {evaluated} / {len(results)}  |  Skipped: {skipped}")
    print("=" * width)
    print()


def print_per_class_table(results):
    """Print per-class accuracy breakdown for all evaluated experiments."""
    evaluated = [r for r in results if r.get("status") == "OK"]
    if not evaluated:
        return

    num_classes = len(EUROSAT_CLASS_NAMES)
    col_w = 10

    print()
    print("  PER-CLASS ACCURACY (%)")
    print("  " + "-" * (24 + num_classes * (col_w + 1)))

    # Header
    hdr = f"  {'Experiment':<24}"
    for cname in EUROSAT_CLASS_NAMES:
        short = cname[:col_w - 1]
        hdr += f" {short:>{col_w}}"
    print(hdr)
    print("  " + "-" * (24 + num_classes * (col_w + 1)))

    for r in evaluated:
        line = f"  {r['name']:<24}"
        for c in range(num_classes):
            val = r["per_class_accuracy"][c] * 100.0
            line += f" {val:>{col_w}.1f}"
        print(line)

    print()


# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------

def _json_serialisable(obj):
    """Convert numpy / torch types for JSON serialisation."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    return obj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Seed for reproducible data splits
    set_seed(args.seed)

    # Device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = get_device()

    # Output directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Banner
    print()
    print("#" * 72)
    print("#  EfficientEuroSAT -- A1: Master Accuracy Evaluation")
    print("#" * 72)
    print(f"  Checkpoints dir : {args.checkpoints_dir}")
    print(f"  Data root       : {args.data_root}")
    print(f"  Save dir        : {args.save_dir}")
    print(f"  Device          : {device}")
    print(f"  Batch size      : {args.batch_size}")
    print(f"  Seed            : {args.seed}")
    print()

    # ------------------------------------------------------------------
    # Load EuroSAT test data (shared across all experiments)
    # ------------------------------------------------------------------
    print("Loading EuroSAT test data...")
    _, _, test_loader, _ = get_eurosat_dataloaders(
        root=args.data_root,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(f"  Test batches: {len(test_loader)}")
    print()

    # ------------------------------------------------------------------
    # Evaluate each experiment
    # ------------------------------------------------------------------
    all_results = []

    for exp_name in EXPERIMENT_NAMES:
        ckpt_path = os.path.join(
            args.checkpoints_dir, exp_name, "best_model_val_acc.pt"
        )

        if not os.path.isfile(ckpt_path):
            print(f"[SKIP] {exp_name:24s} -- checkpoint not found: {ckpt_path}")
            all_results.append({
                "name": exp_name,
                "status": "SKIPPED",
                "test_accuracy": None,
                "test_loss": None,
                "per_class_accuracy": None,
                "num_samples": None,
                "num_correct": None,
                "model_type": None,
                "num_parameters": None,
                "epoch": None,
            })
            continue

        print(f"[EVAL] {exp_name:24s} -- loading {ckpt_path}")
        t0 = time.time()

        model, model_type, model_config, checkpoint = load_model_from_checkpoint(
            ckpt_path, device
        )

        # Disable early exit during evaluation so every sample passes through
        # all layers for a fair accuracy comparison.
        if hasattr(model, "set_early_exit"):
            model.set_early_exit(False)

        total_params, _ = count_parameters(model)

        eval_result = evaluate_model(model, test_loader, device)
        elapsed = time.time() - t0

        record = {
            "name": exp_name,
            "status": "OK",
            "test_accuracy": eval_result["test_accuracy"],
            "test_loss": eval_result["test_loss"],
            "per_class_accuracy": eval_result["per_class_accuracy"],
            "num_samples": eval_result["num_samples"],
            "num_correct": eval_result["num_correct"],
            "model_type": model_type,
            "num_parameters": total_params,
            "epoch": checkpoint.get("epoch", None),
            "eval_time_s": round(elapsed, 2),
        }
        all_results.append(record)

        print(
            f"       accuracy = {eval_result['test_accuracy'] * 100.0:.2f}%  |  "
            f"loss = {eval_result['test_loss']:.4f}  |  "
            f"params = {total_params:,}  |  "
            f"time = {elapsed:.1f}s"
        )

        # Free GPU memory between models
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Seed experiment statistics (mean +/- std)
    # ------------------------------------------------------------------
    seed_accs = [
        r["test_accuracy"]
        for r in all_results
        if r["name"] in SEED_EXPERIMENTS and r["status"] == "OK"
    ]

    seed_stats = None
    if len(seed_accs) >= 2:
        seed_stats = {
            "mean": float(np.mean(seed_accs)),
            "std": float(np.std(seed_accs, ddof=0)),
            "n": len(seed_accs),
            "values": seed_accs,
        }
    elif len(seed_accs) == 1:
        seed_stats = {
            "mean": seed_accs[0],
            "std": 0.0,
            "n": 1,
            "values": seed_accs,
        }

    # ------------------------------------------------------------------
    # Print summary tables
    # ------------------------------------------------------------------
    print_results_table(all_results, seed_stats=seed_stats)
    print_per_class_table(all_results)

    # ------------------------------------------------------------------
    # Generate ablation bar chart (Fig 7)
    # ------------------------------------------------------------------
    evaluated_results = [r for r in all_results if r["status"] == "OK"]

    if evaluated_results:
        chart_path = os.path.join(args.save_dir, "ablation_accuracy.png")

        # Determine baseline accuracy for the reference line
        baseline_acc = None
        for r in all_results:
            if r["name"] == "baseline" and r["status"] == "OK":
                baseline_acc = r["test_accuracy"]
                break

        generate_ablation_chart(evaluated_results, chart_path,
                                baseline_accuracy=baseline_acc)
        print(f"  Ablation bar chart saved to: {chart_path}")
    else:
        print("  No evaluated experiments -- skipping chart generation.")

    # ------------------------------------------------------------------
    # Save results JSON
    # ------------------------------------------------------------------
    output = {
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "experiments": all_results,
        "seed_statistics": seed_stats,
        "class_names": list(EUROSAT_CLASS_NAMES),
    }

    json_path = os.path.join(args.save_dir, "evaluation_results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=_json_serialisable)
    print(f"  Results JSON saved to:       {json_path}")

    # ------------------------------------------------------------------
    # Save CSV summary
    # ------------------------------------------------------------------
    csv_path = os.path.join(args.save_dir, "evaluation_summary.csv")
    fieldnames = [
        "experiment", "status", "test_accuracy_pct", "test_loss",
        "num_parameters", "num_samples", "num_correct", "model_type", "epoch",
    ]
    # Append per-class columns
    for cname in EUROSAT_CLASS_NAMES:
        fieldnames.append(f"class_{cname}_pct")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            row = {
                "experiment": r["name"],
                "status": r["status"],
                "test_accuracy_pct": (
                    f"{r['test_accuracy'] * 100.0:.2f}"
                    if r["test_accuracy"] is not None else ""
                ),
                "test_loss": (
                    f"{r['test_loss']:.4f}"
                    if r["test_loss"] is not None else ""
                ),
                "num_parameters": r.get("num_parameters", ""),
                "num_samples": r.get("num_samples", ""),
                "num_correct": r.get("num_correct", ""),
                "model_type": r.get("model_type", ""),
                "epoch": r.get("epoch", ""),
            }
            if r["per_class_accuracy"] is not None:
                for i, cname in enumerate(EUROSAT_CLASS_NAMES):
                    row[f"class_{cname}_pct"] = f"{r['per_class_accuracy'][i] * 100.0:.2f}"
            else:
                for cname in EUROSAT_CLASS_NAMES:
                    row[f"class_{cname}_pct"] = ""
            writer.writerow(row)

    print(f"  CSV summary saved to:        {csv_path}")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("  EVALUATION COMPLETE")
    print("=" * 72)
    n_ok = sum(1 for r in all_results if r["status"] == "OK")
    n_skip = sum(1 for r in all_results if r["status"] == "SKIPPED")
    print(f"  Experiments evaluated : {n_ok}")
    print(f"  Experiments skipped   : {n_skip}")
    if seed_stats is not None:
        print(
            f"  Seed avg accuracy     : "
            f"{seed_stats['mean'] * 100.0:.2f}% "
            f"+/- {seed_stats['std'] * 100.0:.2f}% "
            f"(n={seed_stats['n']})"
        )
    if evaluated_results:
        best = max(evaluated_results, key=lambda r: r["test_accuracy"])
        print(
            f"  Best experiment       : {best['name']} "
            f"({best['test_accuracy'] * 100.0:.2f}%)"
        )
    print(f"  Output directory      : {args.save_dir}")
    print("=" * 72)
    print()


if __name__ == "__main__":
    main()
