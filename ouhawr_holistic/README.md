# OUHAWR Holistic Approach

Offline Unconstrained Handwritten Arabic Word Recognition (OUHAWR) — Holistic Pipeline

Based on the paper:
> *Offline Handwritten Arabic Word Recognition using Multi-Stream HMM with Explicit State Duration*
> 
> A. Benouareth, A. Kermi, submitted to International Journal of Computer Vision, (2026)

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
| set_a    | Training              |
| set_b    | Training              |
| set_c    | Training              |
|set_d     | Training              |
| set_e    | Test                  |
| set_f    | Test                  |
|set_s     | Test                  |

Configure paths in `configs/default_config.yaml`.

Expected directory layout:
```
/path/to/IFN_ENIT/
├── set_a/
│   ├── <word_class>/
│   │   ├── image1.tif
│   │   └── ...
├── set_b/
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

# 2. Train on DataSets A-D, test on E+F+S
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

| Parameter           | Default | Search Space                                |
|---------------------|---------|---------------------------------------------|
| LPQ+ window M       | 5       | {3,5,7,9}                                   |
| LPQ+ bins B         | 8       | {4,8,16}                                    |
| LPQ+ patches N      | 32      | {2, 4, 8, 16, 32, 64,128,256, 512,1024,2048}|
| WLD T (excitation)  | 8       | {8,10,12,14,16}                             |
| WLD M (orientation) | 6       | {6,8}                                       |                     
| WLD patches N       | 128     | {2, 4, 8, 16, 32, 64,128,256, 512,1024,2048}|
| LDNP scales L       | 3       | {1,2,3}                                     |
| LDNP patches N      | 128     | {2, 4, 8, 16, 32, 64,128,256, 512,1024,2048}|
| Reduction dim d     | 64      |{32,40,50,60,64,70,80,90,100,110,120,128,130,140,150,160,170,180,190,200,210,220,230,240 250,256}                                                                                              |
| SVM C               | 1.0     | {0.01, 0.1, 1.0, 10.0, 100.0}               |
| Reduction method    | LSDR    | PCA,LDA,PCA+LDA,LSDR,LSDR+LDA               |
| N-best hypotheses   | 10      | {1,5, 10, 20, 30,40,50,60, 70,80,90,100}    |

---

## References

- Benouareth et al. (2026) — *OUHAWR using Multi-Stream HMM with Explicit State Duration*
1. Cook, R. D., & Forzani, L. (2008).
   **Principal Fitted Components for Dimension Reduction in Regression.**
   *Statistical Science, 23(4), 485–501.*
   https://doi.org/10.1214/08-STS275

2. Haghighat, M., Abdel-Mottaleb, M., & Alhalabi, W. (2016).
   **Discriminant Correlation Analysis: Real-Time Feature Level Fusion for Multimodal Biometric Recognition.**
   *IEEE Transactions on Information Forensics and Security, 11(9), 1984–1996.*
   https://doi.org/10.1109/TIFS.2016.2569061

3. Chen, J., Shan, S., He, C., Zhao, G., Pietikainen, M., Chen, X., & Gao, W. (2010).
   **WLD: A Robust Local Image Descriptor.**
   *IEEE Transactions on Pattern Analysis and Machine Intelligence, 32(9), 1705–1720.*
   https://doi.org/10.1109/TPAMI.2009.155

4. Rivera, A. R., Castillo, J. R., & Chae, O. (2013).
   **Local Directional Number Pattern for Face Analysis: Face and Expression Recognition.**
   *IEEE Transactions on Image Processing, 22(5), 1740–1752.*
   https://doi.org/10.1109/TIP.2012.2235848

5. Xiao, Y., Cao, Z., Wang, L., & Li, T. (2017).
   **Local Phase Quantization Plus: A Principled Method for Embedding Local Phase
   Quantization into Fisher Vector for Blurred Image Recognition.**
   *Information Sciences, Volume 420, December 2017, Pages 77-95.*
   https://doi.org/10.1016/j.ins.2017.08.059
  
