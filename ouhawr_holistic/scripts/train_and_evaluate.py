#!/usr/bin/env python3
"""
scripts/train_and_evaluate.py

Train the holistic OUHAWR pipeline on IFN/ENIT DataSets A–E and
evaluate on DataSets F and S.

Usage
-----
python scripts/train_and_evaluate.py \\
    --data_root /path/to/IFN_ENIT \\
    --config    configs/default_config.yaml \\
    [--best_params results/cv_results_best_params.json] \\
    [--output_dir  results/] \\
    [--save_model  models/pipeline.joblib]
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ouhawr.preprocessing.image_preprocessing import ImagePreprocessor
from ouhawr.data.ifn_enit_loader import (
    IFNENITLoader, DEFAULT_TRAIN, DEFAULT_TEST
)
from ouhawr.pipeline import HolisticPipeline
from ouhawr.utils.metrics import (
    top1_accuracy, topn_accuracy, per_class_accuracy, recognition_report
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train holistic OUHAWR on DataSets A-E, test on F+S."
    )
    p.add_argument("--data_root",    required=True,
                   help="Root directory of IFN/ENIT database.")
    p.add_argument("--config",       default="configs/default_config.yaml")
    p.add_argument("--best_params",  default=None,
                   help="JSON file with best CV params (optional).")
    p.add_argument("--output_dir",   default="results/",
                   help="Directory to write evaluation results.")
    p.add_argument("--save_model",   default=None,
                   help="Path to save the fitted pipeline (joblib).")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--train_sets",   nargs="+", default=None,
                   help="Override training set names.")
    p.add_argument("--test_sets",    nargs="+", default=None,
                   help="Override test set names.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # -- Load config --
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get("data", {})
    train_sets = args.train_sets or data_cfg.get("train_sets", DEFAULT_TRAIN)
    test_sets  = args.test_sets  or data_cfg.get("test_sets",  DEFAULT_TEST)

    print("=" * 60)
    print("  OUHAWR Holistic Pipeline — Training & Evaluation")
    print("=" * 60)
    print(f"  Train sets : {train_sets}")
    print(f"  Test  sets : {test_sets}")
    print(f"  Data root  : {args.data_root}")
    print()

    # -- Preprocessor --
    pre_cfg = cfg.get("preprocessing", {})
    preprocessor = ImagePreprocessor(
        target_width=pre_cfg.get("target_width", 256),
        target_height=pre_cfg.get("target_height", 128),
        entropy_window=pre_cfg.get("entropy_window", 7),
    )

    # -- Load training data --
    print(f"[Data] Loading training sets: {train_sets} …")
    t0 = time.time()
    train_loader = IFNENITLoader(
        root=args.data_root,
        datasets=train_sets,
        preprocessor=preprocessor,
    )
    train_images, train_labels = train_loader.load_dataset(
        datasets=train_sets, random_seed=args.seed
    )
    print(f"[Data] Training: {len(train_images)} images, "
          f"{len(np.unique(train_labels))} classes  "
          f"({time.time()-t0:.1f}s)")

    # -- Load test data --
    print(f"[Data] Loading test sets: {test_sets} …")
    t0 = time.time()
    test_loader = IFNENITLoader(
        root=args.data_root,
        datasets=test_sets,
        preprocessor=preprocessor,
    )
    test_images, test_labels = test_loader.load_dataset(
        datasets=test_sets, random_seed=args.seed
    )
    print(f"[Data] Test    : {len(test_images)} images, "
          f"{len(np.unique(test_labels))} classes  "
          f"({time.time()-t0:.1f}s)\n")

    # -- Build pipeline --
    pipeline = HolisticPipeline(config=cfg)

    # Apply best CV params if provided
    if args.best_params:
        with open(args.best_params, "r") as f:
            best = json.load(f)
        print(f"[Pipeline] Applying best CV params from {args.best_params}:")
        for k, v in best.items():
            print(f"    {k} = {v}")
        print()
        pipeline.update_params(best)

    # -- Train --
    print("[Pipeline] Training …")
    t0 = time.time()
    pipeline.fit(train_images, train_labels, preprocess=False)
    train_time = time.time() - t0
    print(f"[Pipeline] Training complete in {train_time:.1f}s\n")

    # -- Evaluate on each test set individually --
    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    for ds in test_sets:
        print(f"[Eval] Evaluating on {ds} …")
        t0 = time.time()
        ds_loader = IFNENITLoader(
            root=args.data_root,
            datasets=[ds],
            preprocessor=preprocessor,
        )
        ds_images, ds_labels = ds_loader.load_dataset(
            datasets=[ds], random_seed=args.seed
        )
        if len(ds_images) == 0:
            print(f"  (no images found — skipping)")
            continue

        nbest  = pipeline.predict_nbest(ds_images, preprocess=False)
        y_pred = np.array([h[0][0] for h in nbest])

        t1 = top1_accuracy(ds_labels, y_pred)
        tn = topn_accuracy(ds_labels, nbest)
        elapsed = time.time() - t0

        print(recognition_report(ds_labels, y_pred, nbest))
        print(f"  Inference time: {elapsed:.2f}s  "
              f"({elapsed/len(ds_images)*1000:.1f} ms/image)\n")

        all_results[ds] = {
            "n_images":  len(ds_images),
            "n_classes": int(len(np.unique(ds_labels))),
            "top1":      round(t1 * 100, 2),
            "top_n":     round(tn * 100, 2),
            "n_best":    pipeline.clf.n_best,
        }

        # Per-class breakdown
        per_cls = per_class_accuracy(ds_labels, y_pred)
        cls_df  = pd.DataFrame(
            {"class": list(per_cls.keys()),
             "accuracy": list(per_cls.values())}
        ).sort_values("accuracy")
        cls_csv = os.path.join(args.output_dir, f"per_class_{ds}.csv")
        cls_df.to_csv(cls_csv, index=False)
        print(f"  Per-class accuracy saved to: {cls_csv}")

    # -- Combined test evaluation (F + S together) --
    if len(test_sets) > 1:
        print(f"\n[Eval] Combined evaluation on {test_sets} …")
        nbest_all  = pipeline.predict_nbest(test_images, preprocess=False)
        y_pred_all = np.array([h[0][0] for h in nbest_all])
        t1_all = top1_accuracy(test_labels, y_pred_all)
        tn_all = topn_accuracy(test_labels, nbest_all)
        print(recognition_report(test_labels, y_pred_all, nbest_all))
        all_results["COMBINED"] = {
            "n_images":  len(test_images),
            "n_classes": int(len(np.unique(test_labels))),
            "top1":      round(t1_all * 100, 2),
            "top_n":     round(tn_all * 100, 2),
            "n_best":    pipeline.clf.n_best,
        }

    # -- Save summary --
    summary_path = os.path.join(args.output_dir, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Eval] Summary saved to: {summary_path}")

    # Print table
    print("\n" + "=" * 60)
    print(f"  {'Dataset':<12}  {'Images':>7}  {'Top-1':>7}  "
          f"{'Top-N':>7}  {'N':>4}")
    print("-" * 60)
    for ds, r in all_results.items():
        print(f"  {ds:<12}  {r['n_images']:>7}  {r['top1']:>6.2f}%  "
              f"{r['top_n']:>6.2f}%  {r['n_best']:>4}")
    print("=" * 60)

    # -- Save model --
    if args.save_model:
        os.makedirs(os.path.dirname(args.save_model) or ".", exist_ok=True)
        pipeline.save(args.save_model)


if __name__ == "__main__":
    main()
