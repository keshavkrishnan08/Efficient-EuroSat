# Running EfficientEuroSAT Autonomously on GPU

## Quick Start

```bash
# 1. Clone the repo
git clone git@github.com:keshavkrishnan08/Efficient-EuroSat.git
cd Efficient-EuroSat

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run everything autonomously
bash run.sh
```

That's it. `run.sh` handles everything from data download to final figures.

## What Happens When You Run `run.sh`

### Phase 0: Setup & Smoke Tests
- Installs dependencies
- Auto-detects GPU (CUDA or MPS)
- Verifies all models build correctly (baseline, efficient, decomposed)
- Verifies loss functions compute without errors

### Phase 1: Training (via Autonomous Orchestrator)
Trains all experiments in order:
1. **Baseline** ViT-Tiny (no modifications)
2. **Full model** with all 5 UCAT attention modifications
3. **Ablation experiments** (each modification individually, leave-one-out)
4. **Decomposition experiments** (aleatoric/epistemic temperature decomposition)

### Phase 2: Self-Tuning
If benchmark targets aren't met, the orchestrator automatically:
- Diagnoses the issue (low accuracy, overfitting, high tau correlation, etc.)
- Adjusts hyperparameters (learning rate, dropout, lambda weights, etc.)
- Retrains (up to 3 retries per experiment)

### Phase 3: Evaluation & Figures
Runs all analysis scripts automatically:
- Accuracy evaluation and per-class breakdown
- UCAT temperature-entropy correlation analysis
- OOD detection via temperature (EuroSAT vs DTD)
- Calibration comparison (baseline vs UCAT)
- Robustness under corruptions
- Decomposition validation (tau_a vs tau_e independence)
- Blur response analysis
- Training dynamics
- Latency benchmarks
- Publication-quality figure generation (IEEE TNNLS format)

## GPU Requirements

| Setup | VRAM Needed | Estimated Time |
|-------|-------------|----------------|
| Single GPU (A100/H100) | ~4 GB | ~2-3 hours |
| Single GPU (RTX 3090/4090) | ~4 GB | ~3-5 hours |
| Single GPU (T4/V100) | ~4 GB | ~4-6 hours |

The model is ViT-Tiny (~5.7M params), so GPU memory is not a concern. Training time depends on the number of retry cycles triggered by the self-tuning orchestrator.

## Running on Common GPU Platforms

### SLURM Cluster
```bash
#!/bin/bash
#SBATCH --job-name=eurosat
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=eurosat_%j.log

module load cuda python
pip install -r requirements.txt
bash run.sh
```

### Google Colab
```python
!git clone git@github.com:keshavkrishnan08/Efficient-EuroSat.git
%cd Efficient-EuroSat
!pip install -r requirements.txt
!bash run.sh
```

### Lambda / Vast.ai / RunPod
```bash
git clone git@github.com:keshavkrishnan08/Efficient-EuroSat.git
cd Efficient-EuroSat
pip install -r requirements.txt
nohup bash run.sh > run.log 2>&1 &
tail -f run.log
```

## Output Structure

After completion, results are organized as:

```
checkpoints/          # Trained model weights
  baseline/
  full_model/
  decomp_with_losses/
  ...
results/              # Training metrics JSON per experiment
analysis_results/     # Evaluation outputs
  accuracy/
  ucat/
  ood/
  calibration/
  robustness/
  decomposition/
  dynamics/
  latency/
figures/              # Publication-ready PNG figures (300 DPI)
pipeline_summary.json # Full pipeline report
```

## Troubleshooting

- **Out of memory**: Reduce batch size in `scripts/autonomous_orchestrator.py` (default: 64)
- **Data download fails**: Manually download EuroSAT to `./data/eurosat/` and DTD to `./data/dtd/`
- **Script hangs on data loading**: Set `num_workers=0` in the orchestrator if running in a container
- **Want to skip training and just evaluate**: Place checkpoint files in `./checkpoints/<experiment_name>/best_model_val_acc.pt` and run evaluation scripts directly
