# Multi-source Domain Adaptation for Action Recognition

[![Report](https://img.shields.io/badge/Paper-REPORT.md-blue)](docs/REPORT.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 👥 Group and Project Information

- **Group ID**: DataLost
- **Project ID**: Track 9

## 📝 Project Description

We tackle **Multi-Source Domain Adaptation (MSDA)** for action recognition.
Two labeled source datasets (HMDB-51, UCF-101) are combined to classify actions
in an unlabeled target (a Kinetics subset). A shared encoder is trained over
pre-extracted clip features with per-source classifiers and an adversarial
domain discriminator (Gradient Reversal Layer), and a similarity-based weighted
ensemble dynamically balances the two sources at inference on the target.

> 📖 **Official Report**: theoretical details, analysis, and contributions are in
> [docs/REPORT.md](docs/REPORT.md).

## 🛠 Technical Reproducibility

### 1. Data and Environment Setup

```bash
git clone https://github.com/MaccarroneAlessia/DomainAdaptation-Track9-DataLost.git
cd DomainAdaptation-Track9-DataLost
conda env create -f environment.yml
conda activate dl-project
```

> ⚠️ **Running on an offline / air-gapped cluster?** Internet-dependent steps
> (backbone weights, environment, datasets) must be prepared from home and
> transferred. Follow [docs/OFFLINE_SETUP.md](docs/OFFLINE_SETUP.md) instead of
> the steps below, then come back here for the training commands.

**Datasets.** Download the raw videos into `data/raw/`:
- HMDB-51: https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/
- UCF-101: https://www.crcv.ucf.edu/data/UCF101.php
- Kinetics (subset): https://github.com/cvdfoundation/kinetics-dataset

We use a **closed-set of 12 classes shared across all three datasets** (based on
the UCF-HMDB DA benchmark). The mapping lives in `src/data/class_mapping.py`.
After downloading, verify the class names match the folder layout:

```bash
python -m src.data.verify_classes --dataset ucf101 --video-root data/raw/UCF-101
python -m src.data.verify_classes --dataset hmdb51 --video-root data/raw/hmdb51
```

**Feature extraction (run once).** DA runs on cached features from a frozen
Kinetics-pretrained 3D backbone, not on raw video:

```bash
python -m src.data.extract_features --dataset hmdb51   --video-root data/raw/hmdb51   --out-root features
python -m src.data.extract_features --dataset ucf101   --video-root data/raw/UCF-101  --out-root features
python -m src.data.extract_features --dataset kinetics --video-root data/raw/kinetics --out-root features
```

This produces `features/<dataset>/<class>/<id>.npy` and an `index.csv` per dataset.

### 2. Network Training

**Baseline (objective 1 — no adaptation, zero-shot on target):**

```bash
python -m src.training.train --config experiments/configs/baseline.yaml
```

**Multi-source DA model (objectives 2–3):**

```bash
python -m src.training.train --config experiments/configs/model_v1.yaml
```

### 3. Evaluation

```bash
python -m src.evaluation.evaluate --config experiments/configs/model_v1.yaml
```

Prints target accuracy and the **Source-1 vs Source-2 influence ratio**
(objective 4).

## Repository structure

```
src/
  data/      class_mapping, feature extraction, datasets, class verification
  models/    GRL, shared encoder + per-source heads + discriminator + ensemble
  training/  train.py (baseline + msda modes)
  evaluation/evaluate.py
experiments/configs/   baseline.yaml, model_v1.yaml
docs/REPORT.md
```

*For the declaration of individual tasks and the use of AI, see `docs/REPORT.md`.*
