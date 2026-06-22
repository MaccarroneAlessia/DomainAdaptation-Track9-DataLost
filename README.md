# Multi-source Domain Adaptation for Action Recognition

[![Report](https://img.shields.io/badge/Paper-REPORT.md-blue)](docs/REPORT.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 👥 Group and Project Information

- **Group ID**: DataLost
- **Project ID**: Track 9

## 📝 Project Description

We tackle **Multi-Source Domain Adaptation (MSDA)** for action recognition.
Two labeled source datasets (HMDB-51, UCF-101) are combined to classify actions in an unlabeled target domain (a Kinetics-400 subset). 

To ensure a rigorous evaluation and avoid data leakage (since Kinetics is the target), we extract features using an **ImageNet-1K pretrained ResNet-50 inflated to 3D (I3D)**, completely freezing the backbone. A shared encoder is trained over these pre-extracted clip features with per-source classifiers and an adversarial domain discriminator (Gradient Reversal Layer). Finally, a similarity-based weighted ensemble dynamically balances the two sources at inference on the target.

> 📖 **Official Report**: theoretical details, analysis, and contributions are available in Italian at [docs/REPORT.md](docs/REPORT.md).

## 🛠 Technical Reproducibility

### 1. Data and Environment Setup

```bash
git clone https://github.com/MaccarroneAlessia/DomainAdaptation-Track9-DataLost.git
cd DomainAdaptation-Track9-DataLost
conda env create -f environment.yml
conda activate dl-project
```

> ⚠️ **HPC / Cluster Execution:** Real training and feature extraction are designed to be executed via SLURM and Apptainer on the cluster. Check `run_pipeline.sbatch` for the exact offline deployment commands.

**Datasets.** Download the raw videos into `data/raw/`:
- HMDB-51: https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/
- UCF-101: https://www.crcv.ucf.edu/data/UCF101.php
- Kinetics (subset): https://github.com/cvdfoundation/kinetics-dataset

We use a **closed-set of 11 classes shared across all three datasets** (based on the UCF-HMDB benchmark, intentionally excluding 'fencing' to avoid forced mappings). The mapping lives in `src/data/class_mapping.py`.

**Feature extraction (run once).** DA runs on cached features from the ImageNet-pretrained I3D backbone:

```bash
python -m src.data.extract_features --dataset hmdb51   --video-root data/raw/hmdb51   --backbone inflated_resnet50 --out-root features_imagenet
python -m src.data.extract_features --dataset ucf101   --video-root data/raw/UCF-101  --backbone inflated_resnet50 --out-root features_imagenet
python -m src.data.extract_features --dataset kinetics --video-root data/raw/kinetics --backbone inflated_resnet50 --out-root features_imagenet
```

This produces `features_imagenet/<dataset>/<class>/<id>.npy` and an `index.csv` per dataset.

### 2. Network Training

**Multi-source DA model (Adversarial Training + GRL):**

```bash
python -m src.training.train --config experiments/configs/model_v1_in.yaml
```

### 3. Interactive Notebooks (Analysis & Plots)

The project includes two interactive Jupyter Notebooks to explore the mathematical dynamics and visualize the results:

- `notebooks/1_dati_backbone_feature.ipynb`: Explores the datasets, class distribution, and the I3D feature extraction mechanics.
- `notebooks/2_training_da_classifiers.ipynb`: Demonstrates the GRL math via an interactive simulation (Smoke Test) and plots the **real PCA alignments** and **Zero-Shot accuracies** by loading the trained weights from `experiments/checkpoints/msda_model_v1_in.pt`.
- `notebooks/3_validazione_risultati.ipynb`: Extra validation and metric analyses.

## 📁 Repository structure (Project Hierarchy)

```text
DomainAdaptation-Track9-DataLost/
├── docs/                               # Official Documentation
│   ├── REPORT.md                       # report (theory, results, contributions)
│   └── OFFLINE_SETUP.md                # Setup guide 
├── experiments/                        # Experiment configurations and outputs
│   ├── checkpoints/                    # Saved model weights (.pt files)
│   └── configs/                        
│       ├── baseline.yaml               # Training config for Source-Only baseline (no DA)
│       └── model_v1_in.yaml            # Config for Adversarial Multi-Source DA
├── notebooks/                          # Interactive analysis and visualization
│   ├── 1_dati_backbone_feature.ipynb   # Dataset exploration, label mapping, and I3D feature extraction mechanics
│   ├── 2_training_da_classifiers.ipynb # Interactive Smoke Test simulation, PCA generation, and Zero-Shot eval
│   └── 3_validazione_risultati.ipynb   # Extra validation and metric analyses
├── src/                                # Core Python source code
│   ├── data/ (or datasets/)            # Data loading and preprocessing pipelines
│   │   ├── class_mapping.py            # Definition of the 11 shared action classes
│   │   ├── datasets.py                 # PyTorch DataLoaders and InfiniteSampler logic
│   │   ├── extract_features.py         # Offline massive I3D feature extraction script
│   │   └── verify_classes.py           # Verification utility for raw video folders
│   ├── evaluation/                     # Post-training evaluation scripts
│   │   ├── evaluate.py                 # Main CLI script to test trained models
│   │   └── confusion.py                # Confusion matrix generators
│   ├── models/                         # Neural network architectures
│   │   ├── inflated_resnet.py          # 2D ResNet-50 inflated to 3D video backbone
│   │   ├── multisource_da.py           # Main MSDA architecture (Shared Encoder, Heads, Discriminator, Ensemble)
│   │   └── grl.py                      # Gradient Reversal Layer (Autograd logic)
│   ├── training/                       
│   │   └── train.py                    # Main training loop (Adversarial GRL & Source-Drop mechanism)
│   └── utils/                          
│       └── common.py                   # Shared utilities and helpers
├── figures/                            # Output directory for plots generated by the notebooks
├── run_pipeline.sbatch                 # SLURM script for automated cluster job submission
├── environment.yml                     # Conda environment dependencies
└── README.md                           # This file
```
