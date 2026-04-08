FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Mamba
RUN pip install mamba-ssm causal-conv1d

# Copy project
COPY . .

CMD ["python", "experiments/run_experiment.py", "--config", "configs/base.yaml"]