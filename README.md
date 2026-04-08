# Hybrid Transformer-SSM with Adaptive Layer Scheduling

[![Paper](https://img.shields.io/badge/paper-arXiv-red)](https://arxiv.org/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Official implementation of "Hybrid Transformer-SSM with Adaptive Layer Scheduling for Ultra-Long Sequence Modeling".

## 🚀 Key Features

- **Adaptive Token-Level Routing**: Dynamically routes each token between Transformer and Mamba SSM blocks
- **State-of-the-Art Performance**: Competitive results on Long Range Arena (LRA) benchmark
- **Memory Efficient**: Linear scaling for ultra-long sequences (up to 16K+ tokens)
- **Interpretable**: Built-in routing visualization and analysis tools
- **Research Ready**: Complete ablation studies and benchmarking suite

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/hybrid-transformer-ssm.git
cd hybrid-transformer-ssm

# Install dependencies
pip install -r requirements.txt

# Install Mamba (optional but recommended)
pip install mamba-ssm causal-conv1d