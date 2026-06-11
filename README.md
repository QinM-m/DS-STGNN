# DS-STGNN: Density-Aware Sparse Spatio-Temporal Graph Neural Networks for Cross-Camera Data Association

[![Pytorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e=1&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> This repository contains the official PyTorch implementation for our paper: **DS-STGNN: Density-Aware Sparse Spatio-Temporal Graph Neural Networks for Cross-Camera Data Association**.

## 📝 Abstract
This repository provides the code for our proposed **Density-Aware Sparse Spatio-Temporal Graph Neural Network (DS-STGNN)** for cross-camera data association in MTMCT. 

Traditional GNN-based association methods often rely on computationally expensive fully connected graphs and require complex heuristic post-processing. To address these limitations, DS-STGNN introduces two main components:
1. **Density-aware adaptive sparsification:** Dynamically adjusts graph connectivity based on local crowd density, significantly reducing computational overhead and mitigating trajectory fragmentation.
2. **Dual-Branch Message Passing:** Explicitly decouples spatial graph convolutions from temporal propagation, enabling a robust, end-to-end association pipeline without the strict dependency on artificial post-processing.

Extensive experiments show that our method significantly reduces inference time while achieving strong zero-shot generalization and state-of-the-art accuracy in complex scenarios.

## 🎬 Visualization Demo (EPFL-Laboratory)
<video src="laboratory_4view_demo1.mp4" width="800" controls="controls" muted="muted"></video>
## ⚙️ Environment Setup

We recommend using [Anaconda](https://www.anaconda.com/) to manage the environment. 

```bash
# 1. Create and activate conda environment
conda create -n ds_stgnn python=3.8
conda activate ds_stgnn

# 2. Install PyTorch (Please adapt the CUDA version to your hardware)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 3. Install PyTorch Geometric (Crucial)
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f [https://data.pyg.org/whl/torch-2.0.0+cu118.html](https://data.pyg.org/whl/torch-2.0.0+cu118.html)

# 4. Install other dependencies
pip install -r requirements.txt
```
## 📂  Data Preparation

We evaluate our model on the challenging EPFL dataset.

1. Download the EPFL dataset from: https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/
2. Organize the dataset structure as follows:
```bash
./datasets/
  └── EPFL/
      ├── EPFL-Basketball/
      ├── EPFL-Terrace/
      └── EPFL-Laboratory/
```
## 🚀 Training & Evaluation

Modify the configurations in config/config_training.yaml and config/config_inference.yaml, then run the scripts:
```bash
# For training
python main_training.py --config config/config_training.yaml

# For evaluation
python main.py --config config/config_inference.yaml

```