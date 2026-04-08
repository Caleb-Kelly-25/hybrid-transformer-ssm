#!/bin/bash

################################################################################
# LRA Benchmark Training Script
# Trains Hybrid Transformer-SSM on all Long Range Arena tasks
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# Configuration
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export WANDB_MODE=${WANDB_MODE:-online}
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Paths
DATA_DIR="./data/lra_release"
CHECKPOINT_DIR="./checkpoints/lra"
LOG_DIR="./logs/lra"
CONFIG_DIR="./configs"

# Create directories
mkdir -p "$DATA_DIR" "$CHECKPOINT_DIR" "$LOG_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  LRA Benchmark Training Suite${NC}"
echo -e "${BLUE}========================================${NC}"

# Function to run experiment
run_experiment() {
    local task=$1
    local config=$2
    local epochs=$3
    local batch_size=$4
    
    echo -e "\n${GREEN}[TASK] ${task}${NC}"
    echo -e "${YELLOW}Config: ${config}${NC}"
    echo -e "${YELLOW}Epochs: ${epochs}${NC}"
    echo -e "${YELLOW}Batch Size: ${batch_size}${NC}"
    
    local log_file="${LOG_DIR}/${task}_$(date +%Y%m%d_%H%M%S).log"
    
    python experiments/run_experiment.py \
        --config "${CONFIG_DIR}/${config}" \
        --epochs "${epochs}" \
        --batch_size "${batch_size}" \
        2>&1 | tee "$log_file"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo -e "${GREEN}✓ ${task} completed successfully${NC}"
    else
        echo -e "${RED}✗ ${task} failed${NC}"
        return 1
    fi
}

# Download LRA data if not present
if [ ! -d "$DATA_DIR" ]; then
    echo -e "\n${YELLOW}Downloading LRA dataset...${NC}"
    # Note: You need to implement the actual download logic
    # git clone https://github.com/google-research/long-range-arena.git
    # mv long-range-arena/lra_release "$DATA_DIR"
    echo -e "${RED}Please download LRA data manually to ${DATA_DIR}${NC}"
    exit 1
fi

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo -e "\n${BLUE}GPU Information:${NC}"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
fi

# Train on each LRA task
echo -e "\n${BLUE}Starting LRA Training...${NC}"

# 1. ListOps (2K length, 10 classes)
run_experiment "listops" "lra_listops.yaml" 50 32

# 2. Text Classification (4K length, 2 classes)
run_experiment "text" "lra_text.yaml" 50 16

# 3. Retrieval (4K length, 2 classes)
run_experiment "retrieval" "lra_retrieval.yaml" 50 16

# 4. Pathfinder (1K length, 2 classes)
run_experiment "pathfinder" "lra_pathfinder.yaml" 100 64

# 5. Path-X (16K length, 2 classes) - Ultra-long
run_experiment "pathx" "lra_pathx.yaml" 100 8

echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}LRA Training Complete!${NC}"
echo -e "${BLUE}========================================${NC}"

# Generate summary
echo -e "\n${BLUE}Generating summary...${NC}"
python - <<EOF
import json
import glob
import os

results = {}
for log_file in glob.glob("${LOG_DIR}/*.log"):
    task = os.path.basename(log_file).split('_')[0]
    # Parse accuracy from log (simplified - you may need to adjust parsing)
    with open(log_file, 'r') as f:
        content = f.read()
        if 'Test Results:' in content:
            # Extract accuracy
            for line in content.split('\n'):
                if 'accuracy' in line:
                    acc = float(line.split(':')[1].strip())
                    results[task] = acc

print("\nLRA Results Summary:")
print("-" * 40)
for task, acc in sorted(results.items()):
    print(f"{task:15s}: {acc:.4f}")
    
with open("${LOG_DIR}/summary.json", 'w') as f:
    json.dump(results, f, indent=2)
EOF

echo -e "\n${GREEN}Summary saved to ${LOG_DIR}/summary.json${NC}"