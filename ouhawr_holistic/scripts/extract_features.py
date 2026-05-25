#!/usr/bin/env python3
"""
scripts/extract_features.py

Standalone feature extraction script.

Extracts LPQ+, WLD, and LDNP features (before and after reduction / fusion)
from IFN/ENIT images and saves them as .npz files for use by downstream
stages (e.g. the HMM-based analytic approach).

Usage
-----
python scripts/extract_features.py \\
    --data_root /path/to/IFN_ENIT \\
    --config    configs/default_config.yaml \\
    [--datasets DataSetA DataSetB DataSetC DataSetD DataSetE] \\
    [--output_dir features/] \\
    [--stage raw|reduced|fused]   # which representation to save
"""

import argparse
import sys
import os
import numpy as np
import yaml
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ouhawr.preprocessing.image_preprocessing import ImagePreprocessor
from ouhawr.data.ifn_enit_loader import IFNENITLoader, DEFAULT_TRAIN
from ouhawr.descriptors import LPQPlus, WLD, LDNP
from ouhawr.dimensionality_reduction import make_reducer, is_supervised
from ouhawr.fusion.dca_fusion import MDCA


# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract and save OUHAWR holistic features."
    )
    p.add_argument("--data_root",  required=True)
    p.add_argument("--config",     default="configs/default_config.yaml")
    p.add_argument("--datasets",   nargs="+", default=DEFAULT_TRAIN)
    p.add_argument("--output_dir", default="features/")
    p.add_argument("--stage",      choices=["raw", "reduced", "fused"],
                   default="fused",
                   help="Which representation to save.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    # -- Preprocessor --
    pre_cfg = cfg.get("preprocessing", {})
    preprocessor = ImagePreprocessor(
        target_width=pre_cfg.get("target_width", 256),
        target_height=pre_cfg.get("target_height", 128),
        entropy_window=pre_cfg.get("entropy_window", 7),
    )

    # -- Load data --
    print(f"[Extract] Loading {args.datasets} from {args.data_root} …")
    loader = IFNENITLoader(
        root=args.data_root,
        datasets=args.datasets,
        preprocessor=preprocessor,
    )
    images, labels = loader.load_dataset(
        datasets=args.datasets, random_seed=args.seed
    )
    print(f"[Extract] Loaded {len(images)} images, "
          f"{len(np.unique(labels))} classes.")

    # -- Descriptors --
    lpq_cfg  = cfg.get("lpq_plus", {})
    wld_cfg  = cfg.get("wld",      {})
    ldnp_cfg = cfg.get("ldnp",     {})

    lpq = LPQPlus(
        window_size=lpq_cfg.get("window_size", 5),
        bins_B=lpq_cfg.get("bins_B", 8),
        n_patches=lpq_cfg.get("n_patches", 32),
    )
    wld  = WLD(
        excitation_bins_T=wld_cfg.get("excitation_bins_T", 8),
        orientation_bins_M=wld_cfg.get("orientation_bins_M", 6),
        n_patches=wld_cfg.get("n_patches", 128),
        n_scales_L=wld_cfg.get("n_scales_L", 2),
    )
    ldnp = LDNP(
        n_patches=ldnp_cfg.get("n_patches", 128),
        n_scales_L=ldnp_cfg.get("n_scales_L", 3),
    )

    # Fit LPQ+ decorrelation matrix
    print("[Extract] Fitting LPQ+ decorrelation matrix …")
    lpq.fit(images)

    # Extract raw features
    print("[Extract] Extracting raw features …")
    t0 = time.time()
    F_lpq  = np.vstack([lpq.transform(img)  for img in images])
    F_wld  = np.vstack([wld.transform(img)  for img in images])
    F_ldnp = np.vstack([ldnp.transform(img) for img in images])
    print(f"  LPQ+  : {F_lpq.shape}   ({time.time()-t0:.1f}s)")
    t0 = time.time()
    print(f"  WLD   : {F_wld.shape}")
    print(f"  LDNP  : {F_ldnp.shape}")

    if args.stage == "raw":
        out = os.path.join(args.output_dir, "features_raw.npz")
        np.savez_compressed(
            out,
            F_lpq=F_lpq, F_wld=F_wld, F_ldnp=F_ldnp,
            labels=labels,
        )
        print(f"[Extract] Raw features saved to: {out}")
        return

    # Dimensionality reduction
    red_cfg = cfg.get("reduction", {})
    method  = red_cfg.get("method", "LSDR")
    dim     = red_cfg.get("target_dim", 64)
    supervised = is_supervised(method)

    print(f"[Extract] Reducing with {method} → {dim} dims …")
    r_lpq  = make_reducer(method, dim)
    r_wld  = make_reducer(method, dim)
    r_ldnp = make_reducer(method, dim)

    if supervised:
        R_lpq  = r_lpq.fit_transform(F_lpq,  labels)
        R_wld  = r_wld.fit_transform(F_wld,  labels)
        R_ldnp = r_ldnp.fit_transform(F_ldnp, labels)
    else:
        R_lpq  = r_lpq.fit_transform(F_lpq)
        R_wld  = r_wld.fit_transform(F_wld)
        R_ldnp = r_ldnp.fit_transform(F_ldnp)

    print(f"  R_lpq : {R_lpq.shape}")
    print(f"  R_wld : {R_wld.shape}")
    print(f"  R_ldnp: {R_ldnp.shape}")

    if args.stage == "reduced":
        out = os.path.join(args.output_dir, "features_reduced.npz")
        np.savez_compressed(
            out,
            R_lpq=R_lpq, R_wld=R_wld, R_ldnp=R_ldnp,
            labels=labels,
        )
        print(f"[Extract] Reduced features saved to: {out}")
        return

    # DCA fusion
    fus_cfg = cfg.get("fusion", {})
    print("[Extract] Fitting MDCA fusion …")
    mdca = MDCA(n_components=fus_cfg.get("n_components", None))
    FFV  = mdca.fit_transform(R_lpq, R_wld, R_ldnp, labels)
    print(f"  FFV (fused): {FFV.shape}")

    out = os.path.join(args.output_dir, "features_fused.npz")
    np.savez_compressed(out, FFV=FFV, labels=labels)
    print(f"[Extract] Fused features saved to: {out}")


if __name__ == "__main__":
    main()
