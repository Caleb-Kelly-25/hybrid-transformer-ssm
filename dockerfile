FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

WORKDIR /workspace

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install core deps FIRST (critical order)
RUN pip install numpy
RUN pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
RUN pip install ninja

RUN python -c "import torch; print(torch.cuda.is_available())"
RUN nvcc --version

# Install Mamba (must come before requirements)
RUN pip install --no-build-isolation mamba-ssm causal-conv1d

# Now install remaining requirements (WITHOUT mamba inside!)
COPY requirements.txt .
RUN grep -v "mamba-ssm" requirements.txt | grep -v "causal-conv1d" > clean_requirements.txt
RUN pip install --no-cache-dir -r clean_requirements.txt



# Copy project
COPY . .

#CMD ["python", "experiments/run_experiment.py", "--config", "configs/base.yaml"]