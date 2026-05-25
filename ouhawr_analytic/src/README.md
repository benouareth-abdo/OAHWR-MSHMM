# OUHAWR – Offline Unconstrained Handwritten Arabic Word Recognition
## Multi-Stream HMM with Explicit State Duration (MSDHMM)

> Implementation of the analytic approach described in:
>
> **Benouareth A., Kermi A., Kessentini Y.**  
> *"Offline Handwritten Arabic Word Recognition using Multi-Stream HMM with Explicit State Duration"*  
> Information Fusion, 2026.

---

## Overview

This project implements the **analytic stage** of the OUHAWR system: a Synchronous Multi-Stream Hidden Markov Model with Explicit State Duration (MSDHMM) for offline Arabic handwritten word recognition.

### Key Features

| Feature | Details |
|---|---|
| **Duration distributions** | Gamma, Gaussian, Laplace (§3.3.1), Poisson, **Mixture of Gamma + Laplace** (§3.3.2) |
| **EM re-estimation** | Closed-form for Laplace (Eq. 17–18) and Mixture coefficients (Eq. 21–22) |
| **Feature streams (L=4)** | Upper contour (15-dim), Lower contour (15-dim), Statistical (26-dim), Structural skeleton (21-dim) |
| **Decoding** | Fast two-level algorithm (Algorithms 1 & 2, §3.4) — O(T³N²CL + T²KV) |
| **Stream weights** | Mutual-information-based (Eq. 25–26) |
| **Training** | Embedded Baum-Welch on word-level samples (§5.3) |
| **Benchmark** | IFN/ENIT database loader + evaluation utilities |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        MSDHMM                           │
│                                                         │
│  ┌──────────────┐    ┌──────────────────────────────┐  │
│  │ Preprocessor │───▶│     FeatureExtractor          │  │
│  │  §5.1        │    │  Sliding window (R→L)  §5.2  │  │
│  │ • Smooth     │    │  Stream 0: Upper Contour 15d  │  │
│  │ • ChainCode  │    │  Stream 1: Lower Contour 15d  │  │
│  │ • Baselines  │    │  Stream 2: Statistical   26d  │  │
│  │ • Thinning   │    │  Stream 3: Structural    24d  │  │
│  └──────────────┘    └──────────────────────────────┘  │
│                                    │                    │
│                    ┌───────────────▼───────────────┐   │
│                    │    EmbeddedTrainer (§5.3)      │   │
│                    │  Baum-Welch + Duration EM      │   │
│                    │  → CharacterModel × 160+       │   │
│                    └───────────────┬───────────────┘   │
│                                    │                    │
│                    ┌───────────────▼───────────────┐   │
│                    │  TwoLevelDecoder (§3.4)        │   │
│                    │  Alg.1: χ,ε precomputation     │   │
│                    │  Alg.2: Word-level DP          │   │
│                    │  → DecodeResult                │   │
│                    └───────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## File Structure

```
ouhawr/
├── src/
│   ├── MSDHMM.hpp           # Main header: all classes & interfaces
│   ├── Preprocessing.hpp    # Image types, Preprocessor, FeatureExtractor
│   ├── Database.hpp         # IFN/ENIT loader, Evaluator
│   ├── Preprocessing.cpp    # §5.1 pipeline + §5.2 feature extraction
│   ├── StreamHMM.cpp        # Single-stream CHSMM (forward-backward, Viterbi, Alg.1)
│   ├── Distributions.cpp    # Gamma, Gaussian, Laplace, Poisson, Mixture re-estimation
│   ├── Decoder.cpp          # TwoLevelDecoder (Alg.1+2), StreamWeightOptimiser
│   ├── EmbeddedTrainer.cpp  # EmbeddedTrainer, MSDHMM top-level
│   ├── Database.cpp         # IFN/ENIT loader & evaluation
│   └── main.cpp             # Test driver
├── CMakeLists.txt
└── README.md
```

---

## Building

### Prerequisites

- **C++20** compiler (GCC ≥ 10, Clang ≥ 12, MSVC 2019+)
- **CMake ≥ 3.16**
- Optional: **OpenMP** for multi-threaded training

### Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Debug build (with AddressSanitizer)

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
```

---

## Running

### Quick test (synthetic data, no database needed)

```bash
./ouhawr_test
```

### With options

```bash
# Mixture duration, 10 N-best hypotheses, 15 classes, verbose
./ouhawr_test --dur Mixture --nbest 10 --classes 15 --verbose

# IFN/ENIT benchmark (standard sets a-d train, e test)
./ouhawr_test --db /path/to/ifnenit --dur Mixture --nbest 20

# Compare duration distributions
for DUR in Gamma Laplace Mixture Poisson; do
    echo "=== $DUR ==="; ./ouhawr_test --dur $DUR --classes 10 --no-unit
done
```

### All options

| Flag | Default | Description |
|---|---|---|
| `--db <path>` | – | IFN/ENIT root directory |
| `--dur <type>` | `Mixture` | Duration: `Gamma` / `Gaussian` / `Laplace` / `Poisson` / `Mixture` |
| `--iter <N>` | 20 | Max EM iterations |
| `--nbest <N>` | 5 | N-best hypotheses from holistic stage |
| `--classes <N>` | 8 | Number of synthetic word classes |
| `--train <N>` | 10 | Training samples per class (synthetic) |
| `--test <N>` | 5 | Test samples per class (synthetic) |
| `--verbose` | off | Enable per-iteration log-likelihood output |
| `--no-unit` | off | Skip unit tests |

---

## IFN/ENIT Database

The standard **IFN/ENIT** benchmark for offline handwritten Arabic word recognition.  
Download: [ifnenit.com](http://www.ifnenit.com/)

Expected directory structure:

```
ifnenit_root/
├── sets/
│   ├── a/  b/  c/  d/   ← training sets (sets a–d)
│   │   └── *.tif / *.pgm
│   └── e/               ← test set
│       └── *.tif / *.pgm
└── truth/
    ├── a.xml  b.xml  c.xml  d.xml  e.xml
    └── *.txt  (fallback plain-text format)
```

---

## Model Details

### Duration Distributions

| Distribution | Formula | Re-estimation |
|---|---|---|
| **Gamma** | p(d) = β^α/Γ(α) · d^(α-1) · e^(-βd) | Newton-Raphson MLE (Levinson [8]) |
| **Gaussian** | p(d) = N(d; μ, σ²) | Weighted mean & variance |
| **Laplace** | p(d) = (1/2ν) exp(-\|d-μ\|/ν) | Eq. (17): ν̄ = Σ\|d-μ\|w_d/Σw_d; Eq. (18): μ̄ = weighted median |
| **Poisson** | p(d) = e^(-λ) λ^d / d! | λ̄ = weighted mean |
| **Mixture** | p(d) = c₁p¹(d) + c₂p²(d) | Eq. (21)–(22): EM responsibilities |

### Stream Weights (Eq. 25–26)

```
w_{lc} = I(O^l_c, Q^l_c) / Σ_l I(O^l_c, Q^l_c)

I(O,Q) = H(Q) - H(Q|O)   [mutual information]
```

### Two-Level Decoding Complexity

| Level | Complexity |
|---|---|
| Level 1 (per-character, per-stream) | O(T³ N² L C) |
| Level 2 (word-level DP) | O(T² K V) |
| **Total** | **O(T³N²CL + T²KV)** |

---

## References

1. Benouareth et al. (2008). *Arabic handwritten word recognition using HMMs with explicit state duration.* EURASIP JASP.
2. Benouareth et al. (2008). *Semi-continuous HMMs with explicit state duration.* Pattern Recognition Letters.
3. Kessentini et al. (2010). *Offline handwritten word recognition using multi-stream HMMs.* Pattern Recognition Letters.
4. Levinson (1986). *Continuously variable duration HMMs.* Computer Speech & Language.
5. Koerich et al. (2004). *Fast two-level HMM decoding algorithm.* IWFHR.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
