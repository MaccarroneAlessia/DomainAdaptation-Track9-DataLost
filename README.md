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

## 📁 Repository structure

```
src/
  data/      class_mapping, feature extraction, datasets
  models/    GRL, Inflated ResNet, shared encoder, discriminator, ensemble
  training/  train.py (adversarial msda loop)
notebooks/
  1_dati_backbone_feature.ipynb
  2_training_da_classifiers.ipynb
experiments/
  configs/     model_v1_in.yaml
  checkpoints/ (saved models)
docs/
  REPORT.md
run_pipeline.sbatch
```
