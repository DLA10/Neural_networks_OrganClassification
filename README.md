# Deep Learning: OrganSMNIST Multi-Class Classification

A deep-learning pipeline that classifies 2D abdominal CT slices from the **OrganSMNIST** dataset into 11 organ classes. The project progresses from a minimal fully-connected baseline all the way to ImageNet-pretrained transfer learning and class-imbalance-aware augmentation, with a final head-to-head comparative study on a held-out test set.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Repository Layout](#repository-layout)
4. [Environment & Setup](#environment--setup)
5. [How to Run](#how-to-run)
6. [Pipeline Summary](#pipeline-summary)
   - [Initial Visualisation](#initial-visualisation)
   - [Q1: Baseline Neural Network](#q1-baseline-neural-network)
   - [Q2: Improved Neural Network](#q2-improved-neural-network)
   - [Q3: Baseline CNN](#q3-baseline-cnn)
   - [Q4: Improved CNN](#q4-improved-cnn)
   - [Q5: Transfer Learning](#q5-transfer-learning)
   - [Q6: Test Set Evaluation](#q6-test-set-evaluation-comparative-study)
   - [Q7: Data Augmentation & Class Imbalance](#q7-data-augmentation--class-imbalance)
7. [Final Results Snapshot](#final-results-snapshot)
8. [Key Findings](#key-findings)
9. [Limitations & Future Work](#limitations--future-work)
10. [References](#references)

---

## Project Overview

The task is an 11-class image classification problem on grayscale abdominal CT slices (OrganSMNIST, 128×128). The coursework walks through a deliberate progression:

| Stage | Model | Purpose |
|---|---|---|
| Q1 | 1-hidden-layer NN (100 neurons) | Capacity baseline |
| Q2 | Deeper regularised NN (5 trials) | Beat Q1 with MLPs only |
| Q3 | 1-block CNN | Minimal conv baseline |
| Q4 | Deeper CNN (5 trials) | Beat Q3 with a well-tuned CNN |
| Q5 | DenseNet121 & VGG16 (frozen, ImageNet) | Transfer learning |
| Q6 | All 6 models on held-out test | Comparative evaluation |
| Q7 | Best model + augmentation + soft class weights | Tackle class imbalance |

Everything is implemented in TensorFlow/Keras in a single notebook (`coursework_1.ipynb`). Trained weights for each model are saved to disk so evaluation in Q6 and retraining in Q7 can be done without re-running earlier cells.

---

## Dataset

**OrganSMNIST** is part of the **MedMNIST v2** benchmark (Yang et al., 2023). It is derived from the Liver Tumor Segmentation Benchmark (LiTS), with 2D sagittal slices of abdominal CT volumes re-sampled to 128×128 grayscale. There are **11 classes**:

```
Bladder, Femur-L, Femur-R, Heart, Kidney-L, Kidney-R,
Liver, Lung-L, Lung-R, Pancreas, Spleen
```

**Splits used in this project** (from `organsmnist_128.npz`):

| Split | Images | Shape |
|---|---|---|
| Train | 13,932 | (N, 128, 128, 1) |
| Validation | 2,452 | (N, 128, 128, 1) |
| Test | 8,827 | (N, 128, 128, 1) |

**Class imbalance is significant.** Liver contributes ~24.9 % of training images (3,464), while Femur-R contributes only ~4.4 % (614), a **5.6× skew** between the largest and smallest class. This is the main driver behind the design choices in Q7.

**Preprocessing applied:** pixel normalisation to `[0.0, 1.0]` and expansion of the channel axis to `(128, 128, 1)` so Keras `Conv2D` layers accept the input.

---

## Repository Layout

```
.
├── coursework_1.ipynb     # Main submission notebook (all 7 questions)
└── README.md              # This file
```

The following files are kept locally (not committed). Re-running the notebook regenerates them:

```
organsmnist_128.npz   # Dataset (~305 MB)
model_q1.keras        # Saved models (one is > 100 MB)
model_q2.keras
model_q3.keras
model_q4.keras
model_q5a.keras       # DenseNet121 (frozen)
model_q5b.keras       # VGG16 (frozen)
model_q7.keras        # Augmented DenseNet121
```

> **Note on large files.** The dataset `.npz` (~305 MB) and the trained `.keras` model files (one of them ~102 MB) exceed GitHub's 100 MB single-file limit and together would bloat the repo unnecessarily, so they are not committed.

---

## Environment & Setup

- **Python:** 3.11
- **Core libraries:** `tensorflow` (2.x, Keras 3), `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`
- **Platform:** Windows 11. TensorFlow on native Windows does **not** use the GPU for TF ≥ 2.11, so training was run on CPU. Use WSL2 or TensorFlow-DirectML if you need GPU.

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
```

Place `organsmnist_128.npz` in the same folder as the notebook before running.

---

## How to Run

1. Download `organsmnist_128.npz` from MedMNIST v2 (or request it from the authors) and drop it in this folder.
2. Open `coursework_1.ipynb` in Jupyter / VS Code.
3. Run the cells **in order**. Q1–Q5 will each save a `.keras` file; Q6 loads all six for test-set evaluation.
4. Q7 retrains the best model (DenseNet121) with augmentation and soft class weights. It is the longest cell and was run separately.

> Training is slow on CPU. Q5 and Q7 each take ~15–30 min per model depending on hardware. Because training was already run once, the notebook is shipped with **all outputs preserved**, so you can read the metrics without executing anything.

---

## Pipeline Summary

### Initial Visualisation

The notebook opens with three diagnostic plots: class distribution across splits, labelled 5-samples-per-class grid, left/right mirror pairs, and a pixel-intensity histogram. Key observations:

- **Class imbalance is 5.6×** between Liver and Femur-R.
- **Left/right pairs** (Kidney-L vs Kidney-R, Femur-L vs Femur-R, Lung-L vs Lung-R) are near-mirror images, which becomes the hardest failure mode downstream.
- **Pixel intensities are heavily skewed toward 0** (typical CT appearance), so models mostly discriminate in the low-intensity range.

### Q1: Baseline Neural Network

`Flatten(128×128×1) → Dense(100, relu) → Dense(11, softmax)`

Adam, sparse categorical cross-entropy, batch size 128, 20 epochs with `EarlyStopping(patience=5)`.

**Result:** best validation accuracy **64.48 %**. Train/val curves stay tightly coupled around 60–62 %, confirming **under-fitting** (capacity limited), not over-fitting. A single dense layer on raw pixels has no notion of spatial structure.

### Q2: Improved Neural Network

Five MLP trials varying depth, width, dropout, batch-normalisation, and learning rate:

| Trial | Layers | Dropout | BN | LR | Val Acc |
|---|---|---|---|---|---|
| 1 | [256, 128] | 0.0 | ✗ | 0.001 | 0.6831 |
| **2** | **[512, 256, 128]** | **0.3** | **✓** | **0.0005** | **0.7537** |
| 3 | [512, 256, 128] | 0.4 | ✓ | 0.0005 | 0.7296 |
| 4 | [256, 128, 64] | 0.3 | ✓ | 0.001 | 0.7011 |
| 5 | [1024, 512, 256] | 0.4 | ✓ | 0.0003 | 0.6778 |

**Best: Trial 2.** Depth 3, moderate dropout, BN, slightly lower LR. Pushing beyond (Trial 3 heavier dropout, Trial 5 much larger with very low LR) under-converges or over-regularises.

### Q3: Baseline CNN

`Conv2D(32, 3×3) → MaxPool(2) → GAP → Dense(64) → Dense(11)` (only **3,147 trainable params**).

**Result:** ~**61.9 %** validation accuracy after the full 20 epochs (early stopping never triggered). One conv block detects edges/gradients but can't compose higher-level organ shapes. Interestingly, the Q1 MLP outperforms this tiny CNN because sheer parameter count (1.6 M vs 3 K) dominates when features are under-developed.

### Q4: Improved CNN

Five CNN trials stacking progressively deeper conv blocks:

| Trial | Filters | Dropout | BN | LR | Val Acc |
|---|---|---|---|---|---|
| 1 | [32, 64, 128] | 0.20 | ✗ | 0.001 | 0.8532 |
| 2 | [32, 64, 128] | 0.25 | ✗ | 0.001 | 0.8654 |
| 3 | [32, 64, 128] | 0.40 | ✓ | 0.0005 | 0.8556 |
| **4** | **[64, 128, 256]** | **0.30** | **✓** | **0.001** | **0.8740** |
| 5 | [32, 64, 128, 256] | 0.30 | ✓ | 0.0005 | 0.8719 |

**Best: Trial 4.** Wider filters (64→128→256), BN + moderate dropout. Going deeper (Trial 5, four blocks) gives no further gain, suggesting depth alone is not the bottleneck; capacity per level matters more on 128×128 inputs.

### Q5: Transfer Learning

Two ImageNet-pretrained backbones, **frozen** (feature extraction, not fine-tuning), with a lightweight head:

```
grayscale → replicate to 3-channels → pretrained base (frozen)
→ GlobalAveragePooling → Dense(256, relu) → Dropout(0.3) → Dense(11, softmax)
```

| Backbone | Val Acc | Val Loss | Epochs trained (best) |
|---|---|---|---|
| **DenseNet121** | **0.8793** | 0.3009 | 9 (best at epoch 6) |
| VGG16 | 0.8597 | 0.3432 | 9 (best at epoch 6) |

DenseNet's dense connectivity encourages feature reuse across depths, which pays off here even with a frozen base.

### Q6: Test Set Evaluation (Comparative Study)

All six best models evaluated on the held-out **test set** (8,827 images):

| Model | Accuracy | F1 (macro) | Precision | Recall |
|---|---|---|---|---|
| Q1: Baseline NN | 0.4830 | 0.4312 | 0.4688 | 0.4389 |
| Q2: Improved NN | 0.5329 | 0.5027 | 0.5157 | 0.5228 |
| Q3: Baseline CNN | 0.5532 | 0.4286 | 0.4953 | 0.4430 |
| Q4: Improved CNN | **0.7554** | 0.6864 | 0.7160 | 0.7053 |
| **Q5a: DenseNet121** | 0.7529 | **0.7037** | 0.7112 | 0.7061 |
| Q5b: VGG16 | 0.7358 | 0.6727 | 0.7083 | 0.6772 |

**Selection criterion: macro F1** (treats all 11 classes equally, the right metric for a 5.6× imbalanced dataset). **Q5a (DenseNet121) is the winner** at macro F1 = **0.7037**, beating Q4 on balanced performance even though Q4 edges it on raw accuracy by 0.25 pp.

**Per-class breakdown of DenseNet121 (test set):**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Bladder | 0.7111 | 0.9075 | 0.7974 | 811 |
| Femur-L | 0.4758 | 0.6036 | 0.5321 | 439 |
| Femur-R | 0.4985 | 0.3708 | **0.4253** | 445 |
| Heart | 0.9074 | 0.7882 | 0.8437 | 510 |
| Kidney-L | 0.5183 | 0.5440 | 0.5308 | 704 |
| Kidney-R | 0.5245 | 0.4012 | 0.4546 | 693 |
| Liver | 0.9627 | 0.9687 | **0.9657** | 2078 |
| Lung-L | 0.8271 | 0.8917 | 0.8582 | 397 |
| Lung-R | 0.8784 | 0.8724 | 0.8754 | 439 |
| Pancreas | 0.7135 | 0.7826 | 0.7464 | 1343 |
| Spleen | 0.8063 | 0.6364 | 0.7113 | 968 |

- **Best:** Liver (F1 0.97), Lung-L/R (~0.86), Heart (0.84).
- **Worst:** Femur-R (F1 0.42), Kidney-R (0.45), Femur-L (0.53), Kidney-L (0.53). **Left/right pairs dominate the error mass**, exactly as the initial visualisation predicted.

### Q7: Data Augmentation & Class Imbalance

Starting from Q5a (DenseNet121), two techniques are added:

1. **Conservative geometric augmentation:** rotation ±5°, zoom ±5 %, shift ±2 %. *No horizontal flip* (would turn Kidney-L into something resembling Kidney-R, worsening the exact confusion we want to fix).
2. **Soft class weights** (sqrt-dampened). Instead of full inverse-frequency weighting, which can destabilise training, weights are `sqrt(n_max / n_i)` so minority classes get moderate (not extreme) up-weighting:

| Class | Weight | | Class | Weight |
|---|---|---|---|---|
| Liver (majority) | 0.544 | | Lung-L | 1.177 |
| Pancreas | 0.716 | | Heart | 1.193 |
| Spleen | 0.812 | | Femur-L | 1.277 |
| Bladder | 0.946 | | Femur-R | 1.293 |
| Kidney-L | 0.952 | | Lung-R | 1.131 |
| Kidney-R | 0.958 | | | |

**Overall before/after (test set, DenseNet121):**

| Metric | Before | After | Δ |
|---|---|---|---|
| Accuracy | 0.7529 | 0.7529 | 0.0000 |
| F1 (macro) | 0.7037 | 0.7042 | +0.0005 |
| Precision (macro) | 0.7112 | 0.7092 | −0.0021 |
| Recall (macro) | 0.7061 | 0.7078 | +0.0017 |

**Per-class F1 change (where the real story is):**

| Class | Before | After | Δ |
|---|---|---|---|
| **Femur-R** | 0.4253 | 0.4882 | **+0.0629** |
| Heart | 0.8437 | 0.8667 | +0.0230 |
| Spleen | 0.7113 | 0.7203 | +0.0090 |
| Pancreas | 0.7464 | 0.7527 | +0.0062 |
| Lung-R | 0.8754 | 0.8802 | +0.0048 |
| Lung-L | 0.8582 | 0.8603 | +0.0021 |
| Bladder | 0.7974 | 0.7951 | −0.0023 |
| Kidney-R | 0.4546 | 0.4518 | −0.0028 |
| Liver | 0.9657 | 0.9604 | −0.0053 |
| Kidney-L | 0.5308 | 0.5214 | −0.0095 |
| **Femur-L** | 0.5321 | 0.4490 | **−0.0831** |

**The intended rebalancing worked on Femur-R (+6.3 pp).** The hardest minority class improved substantially. But part of the gain came from redistributing mass away from Femur-L (−8.3 pp), the sibling the model was previously over-predicting. Net macro F1 barely moves (+0.0005), but the class-wise picture is more **balanced**, which is the right outcome for a downstream medical application where rare-class recall matters.

---

## Final Results Snapshot

| Metric | Q1 | Q2 | Q3 | Q4 | Q5a (DenseNet) | Q5b (VGG) | Q7 (Aug) |
|---|---|---|---|---|---|---|---|
| Val Acc (best) | 0.6448 | 0.7537 | 0.619 | 0.8740 | 0.8793 | 0.8597 | ≈0.88 |
| Test Acc | 0.4830 | 0.5329 | 0.5532 | 0.7554 | **0.7529** | 0.7358 | 0.7529 |
| Test F1 (macro) | 0.4312 | 0.5027 | 0.4286 | 0.6864 | **0.7037** | 0.6727 | 0.7042 |
| Test Recall (macro) | 0.4389 | 0.5228 | 0.4430 | 0.7053 | 0.7061 | 0.6772 | **0.7078** |
| Params (trainable) | 1.64 M | 8.56 M | 3.1 K | ~405 K | ~265 K (head only) | ~134 K (head only) | ~265 K (head only) |

**Selected best model:** **Q5a DenseNet121 + augmentation + soft class weights (Q7).** Highest balanced performance across 11 classes.

---

## Key Findings

1. **Spatial structure matters more than raw capacity.** A 3.1 K-param conv baseline (Q3) nearly matches a 1.6 M-param MLP (Q1), and a well-tuned 3-block CNN (Q4) crushes the deepest MLP by ~22 pp on test accuracy.
2. **Transfer learning is competitive without fine-tuning.** Frozen DenseNet121 with only a small head trained reaches best balanced performance, using ~65 % the trainable parameters of Q4 (~265 K vs ~405 K).
3. **Accuracy is the wrong headline metric here.** Q4 leads on accuracy but DenseNet121 leads on macro F1, the right call on a 5.6× imbalanced dataset.
4. **Left/right organ pairs are the dominant failure mode.** In sagittal CT, Kidney-L vs Kidney-R and Femur-L vs Femur-R differ mainly by position in the scan, which a frozen ImageNet backbone has no built-in prior for.
5. **Augmentation + soft class weights trade off rather than improve globally.** Femur-R improves +6.3 pp F1, but Femur-L drops −8.3 pp. The total macro F1 change is marginal (+0.0005); gains come from **rebalancing**, not from new information.

---

## Limitations & Future Work

- **Frozen backbones leave performance on the table.** Progressive unfreezing (e.g. top block of DenseNet) with a very low LR would likely close the gap on the hardest classes.
- **Left/right disambiguation needs a positional prior**, e.g. a coordinate channel, or explicitly cropping left/right halves of the slice before classification.
- **A specialised loss** (focal loss, class-balanced loss, or per-class threshold tuning) would probably move the needle further than geometric augmentation on imbalance.
- **Input resolution.** MedMNIST also ships a 224×224 version; re-running the best model at higher resolution would give the pretrained features more signal to work with.
- **Ensembling Q4 and Q5a:** they disagree in different ways (Q4 stronger on accuracy, Q5a on balance), and a simple probability average would probably beat either alone.

---

## References

[1] J. Yang, R. Shi, D. Wei, Z. Liu, L. Zhao, B. Ke, H. Pfister, and B. Ni. **"MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification."** *Scientific Data*, vol. 10, no. 1, p. 41, 2023.

[2] M. Abadi *et al.* **"TensorFlow: A System for Large-Scale Machine Learning."** *Proc. OSDI*, 2016.

[3] K. Simonyan and A. Zisserman. **"Very Deep Convolutional Networks for Large-Scale Image Recognition."** *Proc. ICLR*, 2015.

[4] G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger. **"Densely Connected Convolutional Networks."** *Proc. CVPR*, 2017, pp. 4700–4708.

[5] F. Pedregosa *et al.* **"Scikit-learn: Machine Learning in Python."** *JMLR*, vol. 12, pp. 2825–2830, 2011.

[6] F. Chollet. **"Keras."** https://keras.io, 2015.

[7] P. Bilic *et al.* **"The Liver Tumor Segmentation Benchmark (LiTS)."** *Medical Image Analysis*, vol. 84, p. 102680, 2023.

---

*Coursework submitted for CI7521 (Machine Learning & Deep Learning), MSc Data Science, Kingston University.*
