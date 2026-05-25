#!/usr/bin/env python3
"""
scripts/cross_validate_params.py

Run K-fold cross-validation on the 100-word / 40-sample sub-lexicon
to determine optimal hyper-parameter settings for the holistic pipeline.

Usage
-----
python scripts/cross_validate_params.py \\
    --data_root /path/to/IFN_ENIT \\
    --config configs/default_config.yaml \\
    --n_words 100 --n_samples 40 --k_folds 5 \\
    --output results/cv_results.csv
"""

import argparse
import sys
import os
import json
import time
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ouhawr.preprocessing.image_preprocessing import ImagePreprocessor
from ouhawr.data.ifn_enit_loader import IFNENITLoader, DEFAULT_TRAIN
from ouhawr.data.lexicon_sampler import build_cv_lexicon
from ouhawr.utils.cross_validation import CrossValidator


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="K-fold cross-validation for OUHAWR holistic pipeline."
    )
    p.add_argument("--data_root",  required=True,
                   help="Root directory of IFN/ENIT database.")
    p.add_argument("--config",     default="configs/default_config.yaml",
                   help="YAML configuration file.")
    p.add_argument("--n_words",    type=int, default=100,
                   help="Number of word classes for sub-lexicon.")
    p.add_argument("--n_samples",  type=int, default=40,
                   help="Number of image samples per class.")
    p.add_argument("--k_folds",    type=int, default=5,
                   help="Number of K-fold splits.")
    p.add_argument("--output",     default="results/cv_results.csv",
                   help="Path to save CV results CSV.")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--fast",       action="store_true",
                   help="Use a reduced search space for quick testing.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # -- Load config --
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # -- Preprocessor --
    pre_cfg = cfg.get("preprocessing", {})
    preprocessor = ImagePreprocessor(
        target_width=pre_cfg.get("target_width", 256),
        target_height=pre_cfg.get("target_height", 128),
        entropy_window=pre_cfg.get("entropy_window", 7),
    )

    # -- Load IFN/ENIT (training sets only) --
    print(f"[CV] Loading IFN/ENIT from: {args.data_root}")
    loader = IFNENITLoader(
        root=args.data_root,
        datasets=DEFAULT_TRAIN,
        preprocessor=preprocessor,
    )
    print(loader.summary())

    # -- Build 100-word sub-lexicon --
    print(f"\n[CV] Building {args.n_words}-word sub-lexicon "
          f"({args.n_samples} samples/class) …")
    images, labels = build_cv_lexicon(
        loader,
        datasets=DEFAULT_TRAIN,
        n_words=args.n_words,
        n_samples=args.n_samples,
        random_seed=args.seed,
    )
    print(f"[CV] Sub-lexicon: {len(images)} images, "
          f"{len(np.unique(labels))} classes.")

    # -- Search space --
    if args.fast:
        # Reduced for quick validation runs
        search_space = {
            "lpq_window_size":       [5],
            "lpq_bins_B":            [8],
            "lpq_n_patches":         [32],
            "wld_excitation_bins_T": [8],
            "wld_orientation_bins_M":[6],
            "wld_n_patches":         [128],
            "ldnp_n_patches":        [128],
            "ldnp_n_scales_L":       [3],
            "reduction_method":      ["LSDR", "PCA+LDA"],
            "reduction_dim":         [64],
            "svm_C":                 [0.1, 1.0],
            "n_best":                [10,100],
        }
    else:
        # Full search space from config
        cv_cfg = cfg.get("cross_validation", {})
        lpq_cv   = cfg.get("lpq_plus", {}).get("cv_search", {})
        wld_cv   = cfg.get("wld",      {}).get("cv_search", {})
        ldnp_cv  = cfg.get("ldnp",     {}).get("cv_search", {})
        red_cv   = cfg.get("reduction",{}).get("cv_search", {})
        svm_cv   = cfg.get("svm",      {}).get("cv_search", {})

        search_space = {
            "lpq_window_size":        lpq_cv.get("window_size",       [3, 5, 7, 9]),
            "lpq_bins_B":             lpq_cv.get("bins_B",            [4, 8, 16]),
            "lpq_n_patches":          lpq_cv.get("n_patches",         [2, 4, 8, 16, 32, 64,128,256, 512,1024,2048]),
            "wld_excitation_bins_T":  wld_cv.get("excitation_bins_T", [8, 10, 12, 14,16]),
            "wld_orientation_bins_M": wld_cv.get("orientation_bins_M",[6, 8]),
            "wld_n_patches":          wld_cv.get("n_patches",         [2, 4, 8, 16, 32, 64,128,256, 512,1024,2048]),
            "ldnp_n_patches":         ldnp_cv.get("n_patches",        [2, 4, 8, 16, 32, 64,128,256, 512,1024,2048]),
            "ldnp_n_scales_L":        ldnp_cv.get("n_scales_L",       [2, 3]),
            "reduction_method":       red_cv.get("method",
                                        ["PCA", "LDA", "LSDR",
                                         "PCA+LDA", "LSDR+LDA"]),
            "reduction_dim":          red_cv.get("target_dim",        [32,40,50,60,64,70,80,90,100,110,120,128,130,140,150,160,170,180,190,200,210,220,230,240,250,256]),
            "svm_C":                  svm_cv.get("C",                 [0.01, 0.1, 1.0, 10.0, 100.0]),
            "n_best":                 svm_cv.get("n_best",            [1,5, 10, 20, 30,40,50,60, 70,80,90,100]),
        }

    # -- Run CV --
    cv = CrossValidator(
        images=images,
        labels=labels,
        search_space=search_space,
        k_folds=args.k_folds,
        random_seed=args.seed,
        verbose=True,
    )

    t0 = time.time()
    best = cv.run()
    elapsed = time.time() - t0

    print(f"\n[CV] Total time: {elapsed/60:.1f} min")
    print(f"[CV] Best params: {best.params}")
    print(f"[CV] Best top-1 : {best.mean_top1*100:.2f}% ± "
          f"{best.std_top1*100:.2f}%")

    # -- Save results --
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    rows = []
    for r in cv.results_:
        row = dict(r.params)
        row["mean_top1"] = r.mean_top1
        row["std_top1"]  = r.std_top1
        row["mean_topn"] = r.mean_topn
        row["std_topn"]  = r.std_topn
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("mean_top1", ascending=False)
    df.to_csv(args.output, index=False)
    print(f"[CV] Results saved to: {args.output}")

    # Save best params as JSON
    best_json = args.output.replace(".csv", "_best_params.json")
    with open(best_json, "w") as f:
        json.dump(best.params, f, indent=2)
    print(f"[CV] Best params saved to: {best_json}")


if __name__ == "__main__":
    main()
