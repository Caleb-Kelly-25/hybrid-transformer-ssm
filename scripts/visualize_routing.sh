#!/bin/bash

################################################################################
# Routing Visualization Script
# Generates comprehensive visualizations of routing behavior
################################################################################

set -e

# Configuration
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Paths
CHECKPOINT_DIR="./checkpoints"
OUTPUT_DIR="./figures/routing_analysis"
CONFIG_DIR="./configs"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Routing Visualization Suite${NC}"
echo -e "${BLUE}========================================${NC}"

# Function to visualize single checkpoint
visualize_checkpoint() {
    local checkpoint=$1
    local config=$2
    local name=$3
    
    echo -e "\n${GREEN}Visualizing: ${name}${NC}"
    
    local output_subdir="${OUTPUT_DIR}/${name}"
    mkdir -p "$output_subdir"
    
    # Run routing analysis
    python experiments/analyze_routing.py \
        --checkpoint "$checkpoint" \
        --config "$config" \
        --output_dir "$output_subdir" \
        --num_batches 20
    
    echo -e "${GREEN}✓ Visualizations saved to ${output_subdir}${NC}"
}

# Find all checkpoints
echo -e "\n${YELLOW}Searching for checkpoints...${NC}"

# Visualize best model for each task
if [ -f "${CHECKPOINT_DIR}/lra/listops/best_model.pt" ]; then
    visualize_checkpoint \
        "${CHECKPOINT_DIR}/lra/listops/best_model.pt" \
        "${CONFIG_DIR}/lra_listops.yaml" \
        "listops"
fi

if [ -f "${CHECKPOINT_DIR}/lra/text/best_model.pt" ]; then
    visualize_checkpoint \
        "${CHECKPOINT_DIR}/lra/text/best_model.pt" \
        "${CONFIG_DIR}/lra_text.yaml" \
        "text"
fi

if [ -f "${CHECKPOINT_DIR}/lra/pathfinder/best_model.pt" ]; then
    visualize_checkpoint \
        "${CHECKPOINT_DIR}/lra/pathfinder/best_model.pt" \
        "${CONFIG_DIR}/lra_pathfinder.yaml" \
        "pathfinder"
fi

if [ -f "${CHECKPOINT_DIR}/lra/pathx/best_model.pt" ]; then
    visualize_checkpoint \
        "${CHECKPOINT_DIR}/lra/pathx/best_model.pt" \
        "${CONFIG_DIR}/lra_pathx.yaml" \
        "pathx"
fi

# Create comparison plots
echo -e "\n${BLUE}Creating comparison plots...${NC}"

python - <<EOF
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

output_dir = Path("${OUTPUT_DIR}")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

tasks = ['listops', 'text', 'pathfinder', 'pathx']
colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']

for ax, task, color in zip(axes.flat, tasks, colors):
    report_path = output_dir / task / 'comprehensive_report.json'
    if report_path.exists():
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        # Plot layer specialization
        if 'layer_specialization' in report:
            layers = report['layer_specialization']['layer_transformer_ratio_mean']
            ax.plot(range(len(layers)), layers, 'o-', 
                   color=color, linewidth=2, markersize=8, label=task)
            
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Transformer Ratio')
        ax.set_title(f'{task.capitalize()} Layer Specialization')
        ax.grid(True, alpha=0.3)
        ax.legend()

plt.suptitle('Routing Behavior Across LRA Tasks', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / 'task_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Comparison plot saved to {output_dir}/task_comparison.png")
EOF

echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}Visualization Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Results saved to: ${OUTPUT_DIR}"