#!/bin/bash

# Full experiment pipeline for Hybrid Transformer-SSM

echo "==================================="
echo "Hybrid Transformer-SSM Experiments"
echo "==================================="

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=online

# Create directories
mkdir -p data logs checkpoints figures

# Download LRA data if needed
if [ ! -d "data/lra_release" ]; then
    echo "Downloading LRA dataset..."
    # Add download commands here
fi

# 1. Synthetic data experiments
echo "Running synthetic data experiments..."
python experiments/run_experiment.py \
    --config configs/base.yaml \
    --epochs 20

# 2. LRA Text classification
echo "Running LRA Text classification..."
python experiments/run_experiment.py \
    --config configs/lra_text.yaml \
    --epochs 50

# 3. LRA ListOps
echo "Running LRA ListOps..."
python experiments/run_experiment.py \
    --config configs/lra_listops.yaml \
    --epochs 50

# 4. LRA Pathfinder
echo "Running LRA Pathfinder..."
python experiments/run_experiment.py \
    --config configs/lra_pathfinder.yaml \
    --epochs 100

# 5. Run ablations
echo "Running ablation studies..."
python experiments/ablations.py

# 6. Analyze routing behavior
echo "Analyzing routing behavior..."
python experiments/analyze_routing.py \
    --checkpoint checkpoints/best_model.pt \
    --output_dir figures/routing_analysis

# 7. Generate paper figures
echo "Generating paper figures..."
python evaluation/visualization.py \
    --results_dir logs \
    --output_dir figures/paper

echo "==================================="
echo "Experiments complete!"
echo "Results saved to ./logs and ./figures"
echo "==================================="