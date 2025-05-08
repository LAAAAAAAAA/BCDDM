# BCDDM: Branch-Corrected Denoising Diffusion Model for Black Hole Image Generation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5+-EE4C2C.svg)](https://pytorch.org/)

Official implementation of the BCDDM framework for black hole image generation and parameter inversion, as described in our preprint:  
**"BCDDM: Branch-Corrected Denoising Diffusion Model for Black Hole Image Generation"** ([arXiv:2502.08528](https://arxiv.org/abs/2502.08528)).

## Key Features
- 🌀 **Branch-Corrected Architecture**: Novel neural network design for stable black hole image generation
- ⚡ **Efficient Training**: Optimized for GRMHD simulation datasets (2000+ samples)
- 🔄 **Parameter Inversion**: Recover black hole spin (a*), inclination (θ) from synthetic images
- 📊 **EHT-Resolution Ready**: Tested at 20 μas angular resolution

## Installation
To get started with the BCDDM project, follow these steps:
```bash
git clone https://github.com/yourusername/BCDDM.git
cd BCDDM
```

For a complete environment setup, I recommend creating a conda environment:
```bash
conda create -n bcddm python=3.9
conda activate bcddm
pip install -r requirements.txt
