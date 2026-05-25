# OUHAWR Holistic Approach

Offline Unconstrained Handwritten Arabic Word Recognition (OUHAWR) — Holistic Pipeline

Based on the paper:
> *Offline Handwritten Arabic Word Recognition using Multi-Stream HMM with Explicit State Duration*
> Benouareth, Kermi, Kessentini (2026)

---

## Overview

This repository implements the **holistic pre-filtering stage** of the two-stage OUHAWR pipeline. It produces N-best word hypotheses via texture feature extraction, dimensionality reduction, feature fusion, and SVM classification.

### Pipeline

```
Word Image
    │
    ▼
Preprocessing (resize → smooth → entropy filter)
    │
    ├──► LPQ+  descriptor  ──► LSDR/PCA/LDA reduction ──┐
    ├──► WLD   descriptor  ──► LSDR/PCA/LDA reduction ──┼──► DCA Fusion ──► SVM ──► N-Best
    └──► LDNP  descriptor  ──► LSDR/PCA/LDA reduction ──┘
```

---

## Project Structure

```
ouhawr_holistic/
├── ouhawr/
│   ├── preprocessing/
│   │   └── image_preprocessing.py   # Resize, smooth, entropy filter
│   ├── descriptors/
│   │   ├── lpq_plus.py              # LPQ+ (Algorithm 3 from paper)
│   │   ├── wld.py                   # Weber Local Descriptor
│   │   └── ldnp.py                  # Local Directional Number Pattern
│   ├── dimensionality_reduction/
│   │   ├── pca_reduction.py         # PCA
│   │   ├── lda_reduction.py         # LDA
│   │   ├── lsdr_reduction.py        # LSDR (SPFC model, Cook & Forzani 2008)
│   │   ├── combined_reductions.py   # PCA+LDA, LSDR+LDA pipelines
│   │   └── reducer_factory.py       # Factory for all reduction methods
│   ├── fusion/
│   │   └── dca_fusion.py            # Discriminant Correlation Analysis (MDCA)
│   ├── classification/
│   │   └── svm_classifier.py        # One-vs-rest SVM with N-best output
│   ├── data/
│   │   ├── ifn_enit_loader.py       # IFN/ENIT dataset loader
│   │   └── lexicon_sampler.py       # 100-word / 40-sample lexicon builder
│   ├── utils/
│   │   ├── cross_validation.py      # K-fold CV for hyperparameter search
│   │   └── metrics.py               # Recognition rate, top-N accuracy
│   └── pipeline.py                  # End-to-end holistic pipeline
├── scripts/
│   ├── train_and_evaluate.py        # Train on 5 sets, test on 2
│   ├── cross_validate_params.py     # Run K-fold CV for all parameters
│   └── extract_features.py         # Standalone feature extraction
├── configs/
│   └── default_config.yaml          # All hyperparameters
├── tests/
│   ├── test_descriptors.py
│   ├── test_reduction.py
│   ├── test_fusion.py
│   └── test_pipeline.py
├── requirements.txt
└── setup.py
```

---

## IFN/ENIT Dataset

The IFN/ENIT database contains handwritten Arabic town names across 6 sets:

| Set      | Role (default split)  |
|----------|-----------------------|
| DataSetA | Training              |
| DataSetB | Training              |
| DataSetC | Training              |
| DataSetD | Training              |
| DataSetE | Training              |
| DataSetF | Test                  |
| DataSetS | Test                  |

Configure paths in `configs/default_config.yaml`.

Expected directory layout:
```
/path/to/IFN_ENIT/
├── DataSetA/
│   ├── <word_class>/
│   │   ├── image1.tif
│   │   └── ...
├── DataSetB/
│   └── ...
...
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Cross-validate parameters on 100-word sub-lexicon
python scripts/cross_validate_params.py \
    --data_root /path/to/IFN_ENIT \
    --config configs/default_config.yaml \
    --n_words 100 --n_samples 40 --k_folds 5

# 2. Train on DataSets A-E, test on F+S
python scripts/train_and_evaluate.py \
    --data_root /path/to/IFN_ENIT \
    --config configs/default_config.yaml

# 3. Extract features only (for downstream HMM stage)
python scripts/extract_features.py \
    --data_root /path/to/IFN_ENIT \
    --output_dir features/
```

---

## Parameters (configured via YAML or K-fold CV)

| Parameter           | Default | Search Space          |
|---------------------|---------|-----------------------|
| LPQ+ window M       | 5       | {3,5,7,9}             |
| LPQ+ bins B         | 8       | {4,8,16}              |
| LPQ+ patches N      | 25      | {16,25,36}            |
| WLD T (excitation)  | 8       | {8,10,12}             |
| WLD M (orientation) | 6       | {6,8}                 |
| WLD patches N       | 21      | {9,15,21}             |
| LDNP scales L       | 3       | {1,2,3}               |
| LDNP patches N      | 25      | {16,25,36}            |
| Reduction dim d     | 64      | {32,64,128,256}       |
| SVM C               | 1.0     | {0.01,0.1,1,10,100}   |
| Reduction method    | LSDR    | PCA,LDA,PCA+LDA,LSDR,LSDR+LDA |
| N-best hypotheses   | 10      | {5,10,20,50}          |

---

## References

- Benouareth et al. (2026) — *OUHAWR using Multi-Stream HMM with Explicit State Duration*
- Cook & Forzani (2008) — *Principal Fitted Components for Dimension Reduction in Regression*
- Haghighat et al. (2016) — *Discriminant Correlation Analysis*
- Chen et al. (2010) — *WLD: A Robust Local Image Descriptor*
- Rivera et al. (2013) — *Local Directional Number Pattern*
- Xiao et al. (2017) — *Local Phase Quantization Plus*
