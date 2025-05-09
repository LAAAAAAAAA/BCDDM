# BCDDM: Branch-Corrected Denoising Diffusion Model for Black Hole Image Generation

[![Python 3.11+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5+-EE4C2C.svg)](https://pytorch.org/)

The BCDDM framework for black hole image generation as described in:  
**"BCDDM: Branch-Corrected Denoising Diffusion Model for Black Hole Image Generation"** ([arXiv:2502.08528](https://arxiv.org/abs/2502.08528)).

## Overview
This project implements a Branch-Corrected Denoising Diffusion Model (BCDDM) for generating black hole images. 

## Key Features
- Generates black hole images conditioned on **seven physical parameters** of Radiatively Inefficient Accretion Flow (RIAF) models
- Introduces **branch correction mechanisms** and **weighted hybrid loss functions**
- Demonstrates strong correlation between generated images and their physical parameters
- Achieves significant improvement in parameter prediction performance when augmenting GRRT datasets with BCDDM-generated images (validated via ResNet50 regression)
- Reduces computational costs for black hole image generation while enabling:
  - Dataset expansion
  - Parameter estimation
  - Model fitting


## Installation

### Clone Repository
```bash
git clone https://github.com/LAAAAAAAAA/BCDDM.git
cd BCDDM
```

### Create Conda Environment
```bash
conda create -n bcddm python=3.11.9
conda activate bcddm
pip install -r requirements.txt
```

## Prerequisites
The RIAF image dataset used for training is available at:  
[https://doi.org/10.5281/zenodo.15354648](https://doi.org/10.5281/zenodo.15354648)  
Download and extract the dataset into the project directory.

## Training and Sampling with BCDDM

### Single-GPU Training
```bash
python train.py --config config/plateau_config.json
```

### Multi-GPU Training (Optional)
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun \
    --nproc_per_node=2 \
    --master_port=29500 \
    train.py \
    --distributed \
    --config config/plateau_config.json
```

### Image Generation
Generate comparison images using parameters from GRRT simulations:
```bash
python sample.py --config config/sampling_config_from-grrt.json
```

Generate images with random parameters in parameter space:
```bash
python sample.py --config config/sampling_config_random.json
```

## ResNet Training and Evaluation

### For Original GRRT Dataset
1. Split the dataset into train/val/test sets
2. Place them in corresponding subdirectories under `ResNet_input/`:
   ```
   ResNet_input/
   ├── train/
   ├── val/
   └── test/
   ```

### For BCDDM-Generated Images
1. First convert the format:
   ```bash
   python modify_format.py path/to/data_*.npz
   ```
2. Then split and organize into the same directory structure as above

### Training and Testing
```bash
# Train ResNet
python ResNet_train.py

# Evaluate performance
python ResNet_test.py
```

## Citation
If you find this project useful, please cite:

Associated paper: 
[BCDDM: Branch-Corrected Denoising Diffusion Model for Black Hole Image Generation](https://arxiv.org/abs/2502.08528)

Dataset: [10.5281/zenodo.15354648](https://doi.org/10.5281/zenodo.15354648)

## License
This project is licensed under the **BSD 3-Clause License** - see the [LICENSE](LICENSE) file for details.