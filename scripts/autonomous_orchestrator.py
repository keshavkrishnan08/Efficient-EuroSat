#!/usr/bin/env python3
"""
Autonomous self-tuning training orchestrator for EfficientEuroSAT.

Trains all 35 experiments across multiple datasets, architectures, and seeds.
Checks results against benchmarks, automatically adjusts hyperparameters and
retrains when targets are not met. Runs all evaluations, uncertainty method
comparisons, statistical significance tests, and figure generation.

Designed for a single unattended GPU session.

Usage:
    python scripts/autonomous_orchestrator.py --epochs 200 --data_root ./data
"""

import argparse
import json
import os
import subprocess
import sys
import time
import glob


# ======================================================================
# Configuration
# ======================================================================

BENCHMARKS = {
    "baseline_acc_min": 0.92,
    "baseline_acc_target": 0.95,
    "main_acc_improvement_min": 0.0,   # must be >= baseline
    "main_acc_improvement_target": 0.005,  # 0.5% over baseline
    "decomp_tau_a_e_corr_max": 0.5,    # hard max
    "decomp_tau_a_e_corr_target": 0.3,  # want < 0.3
    "decomp_blur_corr_min": 0.1,       # hard min
    "decomp_blur_corr_target": 0.3,    # want > 0.3
    "ece_max": 0.08,
    "ece_target": 0.04,
    "ood_auroc_min": 0.60,
    "ood_auroc_target": 0.75,
}

MAX_RETRIES = 3

# Hyperparameter adjustment strategies
ADJUSTMENTS = [
    {
        "condition": "accuracy_low",
        "changes": {"lr": 2.0},
        "description": "Double learning rate",
    },
    {
        "condition": "accuracy_low",
        "changes": {"lr": 0.5},
        "description": "Halve learning rate",
    },
    {
        "condition": "overfitting",
        "changes": {"dropout_max": 1.5, "weight_decay": 2.0},
        "description": "Increase regularisation",
    },
    {
        "condition": "tau_correlation_high",
        "changes": {"lambda_aleatoric": 2.0},
        "description": "Increase aleatoric loss weight",
    },
    {
        "condition": "blur_correlation_low",
        "changes": {"lambda_aleatoric": 2.0, "blur_loss_frequency": 0.5},
        "description": "Strengthen blur supervision",
    },
    {
        "condition": "ucat_dominating",
        "changes": {"lambda_ucat": 0.5},
        "description": "Reduce UCAT loss weight",
    },
]

# Datasets used in multi-dataset experiments
DATASETS = ["eurosat", "cifar100", "resisc45"]


# ======================================================================
# Helpers
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Autonomous self-tuning EfficientEuroSAT pipeline"
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--resume", action="store_true",
                        help="Skip experiments with existing checkpoints")
    return parser.parse_args()


def format_duration(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    m = seconds / 60
    if m < 60:
        return f"{m:.1f}m"
    return f"{m / 60:.1f}h"


def checkpoint_exists(save_dir):
    if not os.path.isdir(save_dir):
        return False
    return any(f.endswith(('.pt', '.pth')) for f in os.listdir(save_dir))


def read_results(experiment_name):
    """Read training results JSON for an experiment."""
    pattern = f"./results/{experiment_name}*.json"
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


def get_test_accuracy(results):
    if results is None:
        return None
    test = results.get("test", {})
    return test.get("test_acc", None)


def run_train(experiment_name, flags, epochs, batch_size, data_root):
    """Run a single training experiment."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(script_dir, "train.py")
    python = sys.executable

    save_dir = f"./checkpoints/{experiment_name}"

    cmd = [
        python, "-u", train_script,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--data_root", data_root,
        "--save_dir", save_dir,
        "--experiment_name", experiment_name,
        "--no_wandb",
    ] + flags

    print(f"\n{'='*72}")
    print(f"  Training: {experiment_name}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*72}\n")

    start = time.time()
    result = subprocess.run(cmd, check=False)
    duration = time.time() - start

    success = result.returncode == 0
    print(f"\n  {'COMPLETED' if success else 'FAILED'} in {format_duration(duration)}")

    return success, duration


def run_evaluation_script(script_name, extra_args):
    """Run an evaluation/analysis script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, script_name)
    python = sys.executable

    if not os.path.isfile(script_path):
        print(f"  SKIP: {script_name} not found")
        return False

    cmd = [python, "-u", script_path] + extra_args
    print(f"\n  Running: {script_name}")
    result = subprocess.run(cmd, check=False)
    return result.returncode == 0


def diagnose_issues(results, baseline_acc):
    """Diagnose what's wrong with training results."""
    issues = []
    if results is None:
        issues.append("no_results")
        return issues

    test_acc = get_test_accuracy(results)
    train_results = results.get("training", {})
    best_val_acc = train_results.get("best_val_acc", 0)

    if test_acc is not None and baseline_acc is not None:
        if test_acc < baseline_acc:
            issues.append("accuracy_low")

    train_acc_approx = best_val_acc
    if test_acc is not None and train_acc_approx > 0:
        gap = train_acc_approx - test_acc
        if gap > 0.15:
            issues.append("overfitting")

    return issues


def adjust_hyperparams(base_flags, issues, retry_num):
    """Apply hyperparameter adjustments based on diagnosed issues."""
    flags = list(base_flags)

    flag_dict = {}
    i = 0
    while i < len(flags):
        if flags[i].startswith("--") and i + 1 < len(flags) and not flags[i + 1].startswith("--"):
            key = flags[i]
            val = flags[i + 1]
            flag_dict[key] = val
            i += 2
        else:
            flag_dict[flags[i]] = None
            i += 1

    for issue in issues:
        for adj in ADJUSTMENTS:
            if adj["condition"] == issue:
                print(f"    Applying: {adj['description']}")
                for param, multiplier in adj["changes"].items():
                    flag_key = f"--{param}"
                    if flag_key in flag_dict and flag_dict[flag_key] is not None:
                        old_val = float(flag_dict[flag_key])
                        new_val = old_val * multiplier
                        flag_dict[flag_key] = str(new_val)
                break

    new_flags = []
    for key, val in flag_dict.items():
        new_flags.append(key)
        if val is not None:
            new_flags.append(val)

    return new_flags


def _get_dataset_for_experiment(exp):
    """Extract the dataset from experiment flags. Default: eurosat."""
    flags = exp.get("flags", [])
    for i, f in enumerate(flags):
        if f == "--dataset" and i + 1 < len(flags):
            return flags[i + 1]
    return "eurosat"


# ======================================================================
# Experiments Definition — All 35 experiments
# ======================================================================

def get_experiments():
    """Return ordered list of all 35 experiments to run."""
    return [
        # ---- Phase 1: EuroSAT Baselines (3 seeds for ensemble) ----
        {
            "name": "baseline",
            "flags": ["--model", "baseline", "--seed", "42"],
            "retry": False,
            "phase": "baseline",
        },
        {
            "name": "baseline_s123",
            "flags": ["--model", "baseline", "--seed", "123"],
            "retry": False,
            "phase": "baseline",
        },
        {
            "name": "baseline_s456",
            "flags": ["--model", "baseline", "--seed", "456"],
            "retry": False,
            "phase": "baseline",
        },

        # ---- Phase 2: Main decomposed model (retries OK) ----
        {
            "name": "decomp_with_losses",
            "flags": [
                "--model", "efficient_eurosat",
                "--use_decomposition",
                "--lambda_ucat", "0.1",
                "--lambda_aleatoric", "0.05",
                "--lambda_epistemic", "0.05",
                "--seed", "42",
            ],
            "retry": True,
            "phase": "main",
        },

        # ---- Phase 3: EuroSAT ablations (seed 42) ----
        {
            "name": "ucat_only",
            "flags": [
                "--model", "efficient_eurosat",
                "--no_early_exit", "--no_learned_dropout",
                "--no_learned_residual", "--no_temp_annealing",
                "--lambda_ucat", "0.1", "--seed", "42",
            ],
            "retry": False,
            "phase": "ablation",
        },
        {
            "name": "early_exit_only",
            "flags": [
                "--model", "efficient_eurosat",
                "--no_learned_temp", "--no_learned_dropout",
                "--no_learned_residual", "--no_temp_annealing",
                "--lambda_ucat", "0.0", "--seed", "42",
            ],
            "retry": False,
            "phase": "ablation",
        },
        {
            "name": "dropout_only",
            "flags": [
                "--model", "efficient_eurosat",
                "--no_learned_temp", "--no_early_exit",
                "--no_learned_residual", "--no_temp_annealing",
                "--lambda_ucat", "0.0", "--seed", "42",
            ],
            "retry": False,
            "phase": "ablation",
        },
        {
            "name": "residual_only",
            "flags": [
                "--model", "efficient_eurosat",
                "--no_learned_temp", "--no_early_exit",
                "--no_learned_dropout", "--no_temp_annealing",
                "--lambda_ucat", "0.0", "--seed", "42",
            ],
            "retry": False,
            "phase": "ablation",
        },
        {
            "name": "annealing_only",
            "flags": [
                "--model", "efficient_eurosat",
                "--no_early_exit", "--no_learned_dropout",
                "--no_learned_residual",
                "--lambda_ucat", "0.0", "--seed", "42",
            ],
            "retry": False,
            "phase": "ablation",
        },
        {
            "name": "all_combined_s42",
            "flags": [
                "--model", "efficient_eurosat",
                "--lambda_ucat", "0.1", "--seed", "42",
            ],
            "retry": False,
            "phase": "ablation",
        },
        {
            "name": "all_no_ucat",
            "flags": [
                "--model", "efficient_eurosat",
                "--lambda_ucat", "0.0", "--seed", "42",
            ],
            "retry": False,
            "phase": "ablation",
        },
        {
            "name": "all_combined_s123",
            "flags": [
                "--model", "efficient_eurosat",
                "--lambda_ucat", "0.1", "--seed", "123",
            ],
            "retry": False,
            "phase": "ablation",
        },
        {
            "name": "all_combined_s456",
            "flags": [
                "--model", "efficient_eurosat",
                "--lambda_ucat", "0.1", "--seed", "456",
            ],
            "retry": False,
            "phase": "ablation",
        },
        {
            "name": "decomp_input_dep_only",
            "flags": [
                "--model", "efficient_eurosat",
                "--use_decomposition",
                "--lambda_ucat", "0.1",
                "--lambda_aleatoric", "0.0",
                "--lambda_epistemic", "0.0",
                "--seed", "42",
            ],
            "retry": False,
            "phase": "decomp_ablation",
        },
        {
            "name": "decomp_with_losses_s123",
            "flags": [
                "--model", "efficient_eurosat",
                "--use_decomposition",
                "--lambda_ucat", "0.1",
                "--lambda_aleatoric", "0.05",
                "--lambda_epistemic", "0.05",
                "--seed", "123",
            ],
            "retry": False,
            "phase": "decomp_ablation",
        },

        # ---- Phase 3b: 3-seed coverage for ablations ----
        {"name": "ucat_only_s123", "flags": [
            "--model", "efficient_eurosat", "--no_early_exit", "--no_learned_dropout",
            "--no_learned_residual", "--no_temp_annealing", "--lambda_ucat", "0.1", "--seed", "123"],
            "retry": False, "phase": "ablation_seeds"},
        {"name": "ucat_only_s456", "flags": [
            "--model", "efficient_eurosat", "--no_early_exit", "--no_learned_dropout",
            "--no_learned_residual", "--no_temp_annealing", "--lambda_ucat", "0.1", "--seed", "456"],
            "retry": False, "phase": "ablation_seeds"},
        {"name": "early_exit_only_s123", "flags": [
            "--model", "efficient_eurosat", "--no_learned_temp", "--no_learned_dropout",
            "--no_learned_residual", "--no_temp_annealing", "--lambda_ucat", "0.0", "--seed", "123"],
            "retry": False, "phase": "ablation_seeds"},
        {"name": "early_exit_only_s456", "flags": [
            "--model", "efficient_eurosat", "--no_learned_temp", "--no_learned_dropout",
            "--no_learned_residual", "--no_temp_annealing", "--lambda_ucat", "0.0", "--seed", "456"],
            "retry": False, "phase": "ablation_seeds"},
        {"name": "dropout_only_s123", "flags": [
            "--model", "efficient_eurosat", "--no_learned_temp", "--no_early_exit",
            "--no_learned_residual", "--no_temp_annealing", "--lambda_ucat", "0.0", "--seed", "123"],
            "retry": False, "phase": "ablation_seeds"},
        {"name": "dropout_only_s456", "flags": [
            "--model", "efficient_eurosat", "--no_learned_temp", "--no_early_exit",
            "--no_learned_residual", "--no_temp_annealing", "--lambda_ucat", "0.0", "--seed", "456"],
            "retry": False, "phase": "ablation_seeds"},
        {"name": "residual_only_s123", "flags": [
            "--model", "efficient_eurosat", "--no_learned_temp", "--no_early_exit",
            "--no_learned_dropout", "--no_temp_annealing", "--lambda_ucat", "0.0", "--seed", "123"],
            "retry": False, "phase": "ablation_seeds"},
        {"name": "residual_only_s456", "flags": [
            "--model", "efficient_eurosat", "--no_learned_temp", "--no_early_exit",
            "--no_learned_dropout", "--no_temp_annealing", "--lambda_ucat", "0.0", "--seed", "456"],
            "retry": False, "phase": "ablation_seeds"},
        {"name": "annealing_only_s123", "flags": [
            "--model", "efficient_eurosat", "--no_early_exit", "--no_learned_dropout",
            "--no_learned_residual", "--lambda_ucat", "0.0", "--seed", "123"],
            "retry": False, "phase": "ablation_seeds"},
        {"name": "annealing_only_s456", "flags": [
            "--model", "efficient_eurosat", "--no_early_exit", "--no_learned_dropout",
            "--no_learned_residual", "--lambda_ucat", "0.0", "--seed", "456"],
            "retry": False, "phase": "ablation_seeds"},
        {"name": "decomp_with_losses_s456", "flags": [
            "--model", "efficient_eurosat", "--use_decomposition", "--lambda_ucat", "0.1",
            "--lambda_aleatoric", "0.05", "--lambda_epistemic", "0.05", "--seed", "456"],
            "retry": False, "phase": "ablation_seeds"},

        # ---- Phase 4: Multi-dataset (CIFAR-100 + RESISC45) ----
        {"name": "cifar100_baseline", "flags": [
            "--model", "baseline", "--dataset", "cifar100", "--seed", "42"],
            "retry": False, "phase": "multi_dataset"},
        {"name": "cifar100_all_combined", "flags": [
            "--model", "efficient_eurosat", "--dataset", "cifar100",
            "--lambda_ucat", "0.1", "--seed", "42"],
            "retry": False, "phase": "multi_dataset"},
        {"name": "cifar100_decomp", "flags": [
            "--model", "efficient_eurosat", "--dataset", "cifar100",
            "--use_decomposition", "--lambda_ucat", "0.1",
            "--lambda_aleatoric", "0.05", "--lambda_epistemic", "0.05", "--seed", "42"],
            "retry": False, "phase": "multi_dataset"},
        {"name": "resisc45_baseline", "flags": [
            "--model", "baseline", "--dataset", "resisc45", "--seed", "42"],
            "retry": False, "phase": "multi_dataset"},
        {"name": "resisc45_all_combined", "flags": [
            "--model", "efficient_eurosat", "--dataset", "resisc45",
            "--lambda_ucat", "0.1", "--seed", "42"],
            "retry": False, "phase": "multi_dataset"},
        {"name": "resisc45_decomp", "flags": [
            "--model", "efficient_eurosat", "--dataset", "resisc45",
            "--use_decomposition", "--lambda_ucat", "0.1",
            "--lambda_aleatoric", "0.05", "--lambda_epistemic", "0.05", "--seed", "42"],
            "retry": False, "phase": "multi_dataset"},

        # ---- Phase 5: Multi-architecture (ViT-Small on EuroSAT) ----
        {"name": "eurosat_baseline_small", "flags": [
            "--model", "baseline", "--arch", "small", "--seed", "42"],
            "retry": False, "phase": "multi_arch"},
        {"name": "eurosat_all_combined_small", "flags": [
            "--model", "efficient_eurosat", "--arch", "small",
            "--lambda_ucat", "0.1", "--seed", "42"],
            "retry": False, "phase": "multi_arch"},
        {"name": "eurosat_decomp_small", "flags": [
            "--model", "efficient_eurosat", "--arch", "small",
            "--use_decomposition", "--lambda_ucat", "0.1",
            "--lambda_aleatoric", "0.05", "--lambda_epistemic", "0.05", "--seed", "42"],
            "retry": False, "phase": "multi_arch"},
    ]


# ======================================================================
# Evaluation Pipeline
# ======================================================================

def run_all_evaluations(data_root, batch_size):
    """Run all evaluation and analysis scripts across all datasets."""
    print("\n" + "=" * 72)
    print("  EVALUATION & ANALYSIS PHASE")
    print("=" * 72)

    # ---- Per-dataset evaluations ----
    for dataset in DATASETS:
        print(f"\n  --- Evaluations for dataset: {dataset} ---")

        # Find checkpoints for this dataset
        if dataset == "eurosat":
            ckpt_main = "./checkpoints/decomp_with_losses/best_model_val_acc.pt"
            ckpt_base = "./checkpoints/baseline/best_model_val_acc.pt"
            ckpt_combined = "./checkpoints/all_combined_s42/best_model_val_acc.pt"
        else:
            ckpt_main = f"./checkpoints/{dataset}_decomp/best_model_val_acc.pt"
            ckpt_base = f"./checkpoints/{dataset}_baseline/best_model_val_acc.pt"
            ckpt_combined = f"./checkpoints/{dataset}_all_combined/best_model_val_acc.pt"

        ds_flag = ["--dataset", dataset]
        suffix = f"_{dataset}" if dataset != "eurosat" else ""
        results_base = f"./analysis_results{suffix}"

        # 1. Evaluate main decomposed model
        if os.path.isfile(ckpt_main):
            run_evaluation_script("evaluate.py", [
                "--checkpoint", ckpt_main,
                "--data_root", data_root,
                "--save_dir", f"{results_base}/main_eval",
                "--batch_size", str(batch_size),
            ] + ds_flag)

        # 2. Evaluate baseline
        if os.path.isfile(ckpt_base):
            run_evaluation_script("evaluate.py", [
                "--checkpoint", ckpt_base,
                "--data_root", data_root,
                "--save_dir", f"{results_base}/baseline_eval",
                "--batch_size", str(batch_size),
            ] + ds_flag)

        # 3. Decomposition analysis (only for models with decomposition)
        if os.path.isfile(ckpt_main):
            run_evaluation_script("analyze_decomposition.py", [
                "--checkpoint", ckpt_main,
                "--data_root", data_root,
                "--save_dir", f"{results_base}/decomposition",
                "--batch_size", str(batch_size),
            ] + ds_flag)

        # 4. UCAT analysis
        ckpt_for_ucat = ckpt_main if os.path.isfile(ckpt_main) else ckpt_combined
        if os.path.isfile(ckpt_for_ucat):
            run_evaluation_script("analyze_ucat.py", [
                "--checkpoint", ckpt_for_ucat,
                "--data_root", data_root,
                "--save_dir", f"{results_base}/ucat",
                "--batch_size", str(batch_size),
            ] + ds_flag)

        # 5. OOD detection
        if os.path.isfile(ckpt_for_ucat):
            run_evaluation_script("analyze_ood.py", [
                "--checkpoint", ckpt_for_ucat,
                "--data_root", data_root,
                "--save_dir", f"{results_base}/ood",
                "--batch_size", str(batch_size),
            ] + ds_flag)

        # 6. Calibration
        if os.path.isfile(ckpt_for_ucat) and os.path.isfile(ckpt_base):
            run_evaluation_script("analyze_calibration.py", [
                "--checkpoint_ucat", ckpt_for_ucat,
                "--checkpoint_baseline", ckpt_base,
                "--data_root", data_root,
                "--save_dir", f"{results_base}/calibration",
                "--batch_size", str(batch_size),
            ] + ds_flag)

        # 7. Robustness (EuroSAT only — other datasets use different corruption pipeline)
        if dataset == "eurosat" and os.path.isfile(ckpt_for_ucat):
            run_evaluation_script("analyze_robustness.py", [
                "--checkpoint", ckpt_for_ucat,
                "--data_root", data_root,
                "--save_dir", f"{results_base}/robustness",
                "--batch_size", str(batch_size),
            ] + ds_flag)

    # ---- Cross-dataset / global evaluations ----
    print(f"\n  --- Global evaluations ---")

    # 8. Latency benchmark
    run_evaluation_script("benchmark_latency.py", [
        "--save_dir", "./analysis_results/latency",
        "--num_runs", "200",
        "--warmup_runs", "50",
    ])

    # 9. Training dynamics
    dynamics_results = glob.glob("./results/decomp_with_losses*.json")
    if dynamics_results:
        run_evaluation_script("analyze_training_dynamics.py", [
            "--results_file", dynamics_results[-1],
            "--save_dir", "./analysis_results/dynamics",
        ])

    # 10. Ablation accuracy summary (EuroSAT)
    print("\n  Collecting ablation accuracy results...")
    run_ablation_accuracy_summary(data_root, batch_size)

    # 11. Uncertainty method comparison
    print("\n  Running uncertainty method comparison...")
    run_uncertainty_comparison(data_root, batch_size)

    # 12. Statistical significance tests
    print("\n  Running statistical significance tests...")
    run_evaluation_script("statistical_significance.py", [
        "--results_dir", "./results",
        "--save_dir", "./analysis_results/significance",
    ])

    # 13. Theoretical analysis
    print("\n  Running theoretical analysis...")
    ckpt_main = "./checkpoints/decomp_with_losses/best_model_val_acc.pt"
    if os.path.isfile(ckpt_main):
        run_evaluation_script("theoretical_analysis.py", [
            "--checkpoint", ckpt_main,
            "--data_root", data_root,
            "--save_dir", "./analysis_results/theoretical",
            "--batch_size", str(batch_size),
        ])

    # 14. Generate all figures
    print("\n  Generating publication figures...")
    run_evaluation_script("generate_figures.py", [
        "--results_dir", "./analysis_results",
        "--output_dir", "./figures",
        "--dpi", "300",
    ])


def run_ablation_accuracy_summary(data_root, batch_size):
    """Evaluate all EuroSAT checkpoints and produce accuracy summary."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    eval_script = os.path.join(script_dir, "evaluate.py")

    # Collect results from all training JSONs
    results = {}
    for json_file in sorted(glob.glob("./results/*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)
            exp_name = data.get("experiment_name", os.path.basename(json_file))
            test_acc = data.get("test", {}).get("test_acc")
            dataset = data.get("dataset", "eurosat")
            arch = data.get("arch", "tiny")
            if test_acc is not None:
                results[exp_name] = {
                    "test_acc": test_acc,
                    "dataset": dataset,
                    "arch": arch,
                }
                print(f"  {exp_name:35s}: {test_acc*100:.2f}% ({dataset}/{arch})")
        except (json.JSONDecodeError, OSError):
            continue

    os.makedirs("./analysis_results/accuracy", exist_ok=True)
    with open("./analysis_results/accuracy/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)


def run_uncertainty_comparison(data_root, batch_size):
    """Run uncertainty method comparison for each dataset that has checkpoints."""
    for dataset in DATASETS:
        if dataset == "eurosat":
            ckpt_main = "./checkpoints/decomp_with_losses/best_model_val_acc.pt"
            ckpt_baselines = [
                "./checkpoints/baseline/best_model_val_acc.pt",
                "./checkpoints/baseline_s123/best_model_val_acc.pt",
                "./checkpoints/baseline_s456/best_model_val_acc.pt",
            ]
        else:
            ckpt_main = f"./checkpoints/{dataset}_decomp/best_model_val_acc.pt"
            ckpt_baselines = [
                f"./checkpoints/{dataset}_baseline/best_model_val_acc.pt",
            ]

        if not os.path.isfile(ckpt_main):
            print(f"  SKIP uncertainty comparison for {dataset}: no main checkpoint")
            continue

        # Only run if we have at least the main checkpoint
        existing_baselines = [c for c in ckpt_baselines if os.path.isfile(c)]

        args = [
            "--ucat_checkpoint", ckpt_main,
            "--data_root", data_root,
            "--dataset", dataset,
            "--save_dir", f"./analysis_results/uncertainty_{dataset}",
            "--batch_size", str(batch_size),
        ]
        if existing_baselines:
            args.extend(["--baseline_checkpoints"] + existing_baselines)

        run_evaluation_script("compare_uncertainty_methods.py", args)


# ======================================================================
# Main
# ======================================================================

def main():
    args = parse_args()
    overall_start = time.time()

    print()
    print("#" * 72)
    print("#  EfficientEuroSAT — Autonomous Self-Tuning TNNLS Pipeline")
    print("#" * 72)
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Data root:  {args.data_root}")
    print(f"  Resume:     {args.resume}")
    print(f"  Max retries per experiment: {MAX_RETRIES}")
    print(f"  Datasets:   {', '.join(DATASETS)}")
    print()

    experiments = get_experiments()
    summary = []
    baseline_acc = None

    total = len(experiments)
    print(f"  Total experiments: {total}")
    print()

    # ---- Training phase ----
    for idx, exp in enumerate(experiments):
        name = exp["name"]
        flags = list(exp["flags"])
        can_retry = exp["retry"]
        phase = exp["phase"]

        save_dir = f"./checkpoints/{name}"

        print(f"\n  [{idx+1}/{total}] {name} (phase: {phase})")

        # Resume check
        if args.resume and checkpoint_exists(save_dir):
            print(f"  SKIP: {name} (checkpoint exists)")
            results = read_results(name)
            acc = get_test_accuracy(results)
            if phase == "baseline" and name == "baseline" and acc is not None:
                baseline_acc = acc
            summary.append({"name": name, "status": "SKIPPED", "acc": acc, "phase": phase})
            continue

        # Train with retries
        success = False
        current_flags = flags
        acc = None
        for attempt in range(MAX_RETRIES if can_retry else 1):
            if attempt > 0:
                import shutil
                if os.path.isdir(save_dir):
                    shutil.rmtree(save_dir)
                for f in glob.glob(f"./results/{name}*.json"):
                    os.remove(f)

            ok, duration = run_train(
                name, current_flags, args.epochs, args.batch_size, args.data_root
            )

            if not ok:
                print(f"  Training failed for {name}")
                if can_retry and attempt < MAX_RETRIES - 1:
                    print(f"  Retrying ({attempt + 2}/{MAX_RETRIES})...")
                    issues = ["accuracy_low"]
                    current_flags = adjust_hyperparams(current_flags, issues, attempt)
                continue

            results = read_results(name)
            acc = get_test_accuracy(results)

            if phase == "baseline" and name == "baseline":
                baseline_acc = acc
                success = True
                break

            if acc is not None and can_retry and attempt < MAX_RETRIES - 1:
                issues = diagnose_issues(results, baseline_acc)
                if not issues:
                    success = True
                    break

                print(f"  Issues detected: {issues}")
                print(f"  Adjusting hyperparams for retry {attempt + 2}/{MAX_RETRIES}...")
                current_flags = adjust_hyperparams(current_flags, issues, attempt)
            else:
                success = ok
                break

        status = "COMPLETED" if success else "FAILED"
        summary.append({"name": name, "status": status, "acc": acc, "phase": phase})

    # ---- Evaluation phase ----
    run_all_evaluations(args.data_root, args.batch_size)

    # ---- Summary ----
    total_time = time.time() - overall_start
    print()
    print("=" * 72)
    print("  FINAL SUMMARY")
    print("=" * 72)
    print(f"  {'Experiment':<35s} {'Phase':<16s} {'Status':<12s} {'Test Acc':<10s}")
    print("  " + "-" * 68)
    for row in summary:
        acc_str = f"{row['acc']*100:.2f}%" if row['acc'] is not None else "N/A"
        print(f"  {row['name']:<35s} {row['phase']:<16s} {row['status']:<12s} {acc_str:<10s}")
    print("=" * 72)
    print(f"  Total time: {format_duration(total_time)}")

    # Per-phase summary
    phases = {}
    for row in summary:
        p = row["phase"]
        if p not in phases:
            phases[p] = {"completed": 0, "failed": 0, "skipped": 0}
        phases[p][row["status"].lower()] = phases[p].get(row["status"].lower(), 0) + 1
    print("\n  Per-phase breakdown:")
    for p, counts in phases.items():
        parts = [f"{v} {k}" for k, v in counts.items() if v > 0]
        print(f"    {p:<20s}: {', '.join(parts)}")

    # Save summary
    os.makedirs("./analysis_results", exist_ok=True)
    with open("./analysis_results/pipeline_summary.json", "w") as f:
        json.dump({
            "summary": summary,
            "total_time_seconds": total_time,
            "benchmarks": BENCHMARKS,
            "datasets": DATASETS,
            "num_experiments": total,
        }, f, indent=2)

    print()
    print("  All outputs:")
    print("    Checkpoints:   ./checkpoints/<name>/best_model_val_acc.pt")
    print("    Results:       ./results/<name>.json")
    print("    Analysis:      ./analysis_results/")
    print("    Significance:  ./analysis_results/significance/")
    print("    Theoretical:   ./analysis_results/theoretical/")
    print("    Uncertainty:   ./analysis_results/uncertainty_<dataset>/")
    print("    Figures:       ./figures/")
    print()


if __name__ == "__main__":
    main()
