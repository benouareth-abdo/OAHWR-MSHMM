"""
ouhawr/utils/cross_validation.py

K-fold cross-validation to determine optimal values for all
hyper-parameters of the holistic pipeline.

The CV is run on the 100-word / 40-sample sub-lexicon built from the
IFN/ENIT training sets (DataSets A–E).

Parameters searched (grid search):
  LPQ+  : window_size, bins_B, n_patches
  WLD   : excitation_bins_T, orientation_bins_M, n_patches
  LDNP  : n_scales_L, n_patches
  Reduction: method, target_dim
  SVM   : C, n_best
"""

import numpy as np
import itertools
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..preprocessing.image_preprocessing import ImagePreprocessor
from ..descriptors import LPQPlus, WLD, LDNP
from ..dimensionality_reduction import make_reducer, is_supervised
from ..fusion.dca_fusion import MDCA
from ..classification.svm_classifier import SVMClassifier
from ..utils.metrics import top1_accuracy, topn_accuracy
from ..data.lexicon_sampler import stratified_kfold_split


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CVResult:
    params: Dict[str, Any]
    mean_top1: float
    std_top1: float
    mean_topn: float
    std_topn: float
    fold_top1: List[float] = field(default_factory=list)
    elapsed_sec: float = 0.0

    def __repr__(self) -> str:
        return (
            f"CVResult(top1={self.mean_top1*100:.2f}±{self.std_top1*100:.2f}%, "
            f"top-n={self.mean_topn*100:.2f}%, "
            f"params={self.params})"
        )


# ---------------------------------------------------------------------------
# Feature extraction helper
# ---------------------------------------------------------------------------

def extract_features(
    images: List[np.ndarray],
    lpq_params: dict,
    wld_params: dict,
    ldnp_params: dict,
    lpq_W: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Extract LPQ+, WLD, and LDNP features from a list of images.

    Returns (F_lpq, F_wld, F_ldnp, lpq_W_out)
    where lpq_W_out is the decorrelation matrix (fitted on train images).
    """
    lpq = LPQPlus(**lpq_params)
    wld = WLD(**wld_params)
    ldnp = LDNP(**ldnp_params)

    if lpq_W is not None:
        lpq.W_ = lpq_W
    else:
        lpq.fit(images)

    F_lpq = np.vstack([lpq.transform(img) for img in images])
    F_wld = np.vstack([wld.transform(img) for img in images])
    F_ldnp = np.vstack([ldnp.transform(img) for img in images])

    return F_lpq, F_wld, F_ldnp, lpq.W_


# ---------------------------------------------------------------------------
# Single fold evaluation
# ---------------------------------------------------------------------------

def evaluate_fold(
    train_imgs: List[np.ndarray],
    train_labels: np.ndarray,
    val_imgs: List[np.ndarray],
    val_labels: np.ndarray,
    params: Dict[str, Any],
) -> Tuple[float, float]:
    """
    Train and evaluate the holistic pipeline on one fold.

    Returns
    -------
    (top1_acc, topn_acc)
    """
    # -- Build descriptor configs from params --
    lpq_params = dict(
        window_size=params.get("lpq_window_size", 5),
        bins_B=params.get("lpq_bins_B", 8),
        n_patches=params.get("lpq_n_patches", 32),
        freq_scalar=params.get("lpq_freq_scalar", 1.0),
        quant_alpha=params.get("lpq_quant_alpha", 1.0),
    )
    wld_params = dict(
        excitation_bins_T=params.get("wld_excitation_bins_T", 8),
        orientation_bins_M=params.get("wld_orientation_bins_M", 6),
        n_patches=params.get("wld_n_patches", 128),
        n_scales_L=params.get("wld_n_scales_L", 2),
    )
    ldnp_params = dict(
        n_patches=params.get("ldnp_n_patches", 128),
        n_scales_L=params.get("ldnp_n_scales_L", 3),
    )
    red_method = params.get("reduction_method", "LSDR")
    red_dim    = params.get("reduction_dim", 64)
    svm_C      = params.get("svm_C", 1.0)
    n_best     = params.get("n_best", 10)

    # -- Extract train features --
    F_lpq_tr, F_wld_tr, F_ldnp_tr, W_lpq = extract_features(
        train_imgs, lpq_params, wld_params, ldnp_params
    )

    # -- Fit reducers --
    r_lpq  = make_reducer(red_method, red_dim)
    r_wld  = make_reducer(red_method, red_dim)
    r_ldnp = make_reducer(red_method, red_dim)

    supervised = is_supervised(red_method)
    if supervised:
        R_lpq_tr  = r_lpq.fit_transform(F_lpq_tr, train_labels)
        R_wld_tr  = r_wld.fit_transform(F_wld_tr, train_labels)
        R_ldnp_tr = r_ldnp.fit_transform(F_ldnp_tr, train_labels)
    else:
        R_lpq_tr  = r_lpq.fit_transform(F_lpq_tr)
        R_wld_tr  = r_wld.fit_transform(F_wld_tr)
        R_ldnp_tr = r_ldnp.fit_transform(F_ldnp_tr)

    # -- Fit MDCA --
    mdca = MDCA(n_components=None)
    FFV_tr = mdca.fit_transform(R_lpq_tr, R_wld_tr, R_ldnp_tr, train_labels)

    # -- Train SVM --
    clf = SVMClassifier(C=svm_C, n_best=n_best)
    clf.fit(FFV_tr, train_labels)

    # -- Extract val features (reuse fitted W and reducers) --
    F_lpq_v, F_wld_v, F_ldnp_v, _ = extract_features(
        val_imgs, lpq_params, wld_params, ldnp_params, lpq_W=W_lpq
    )
    R_lpq_v  = r_lpq.transform(F_lpq_v)
    R_wld_v  = r_wld.transform(F_wld_v)
    R_ldnp_v = r_ldnp.transform(F_ldnp_v)

    FFV_v = mdca.transform(R_lpq_v, R_wld_v, R_ldnp_v)

    # -- Evaluate --
    y_pred = clf.predict(FFV_v)
    nbest  = clf.predict_nbest(FFV_v)
    t1 = top1_accuracy(val_labels, y_pred)
    tn = topn_accuracy(val_labels, nbest)
    return t1, tn


# ---------------------------------------------------------------------------
# Cross-validation runner
# ---------------------------------------------------------------------------

class CrossValidator:
    """
    Grid-search cross-validator for the holistic pipeline.

    Parameters
    ----------
    images      : preprocessed images (100-word sub-lexicon)
    labels      : class labels
    search_space: dict of {param_name: [values_to_try]}
    k_folds     : number of CV folds
    random_seed : RNG seed
    verbose     : print progress
    """

    def __init__(
        self,
        images: List[np.ndarray],
        labels: np.ndarray,
        search_space: Optional[Dict[str, List[Any]]] = None,
        k_folds: int = 5,
        random_seed: int = 42,
        verbose: bool = True,
    ):
        self.images = images
        self.labels = labels
        self.k_folds = k_folds
        self.random_seed = random_seed
        self.verbose = verbose
        self.search_space = search_space or self._default_search_space()

        self.results_: List[CVResult] = []
        self.best_params_: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------

    @staticmethod
    def _default_search_space() -> Dict[str, List[Any]]:
        return {
            "lpq_window_size":       [3, 5, 7, 9],
            "lpq_bins_B":            [4, 8, 16],
            "lpq_n_patches":         [2, 4, 8, 16, 32, 64,128,256, 512,1024,2048],
            "wld_excitation_bins_T": [8, 10],
            "wld_orientation_bins_M":[6, 8],
            "wld_n_patches":         [2, 4, 8, 16, 32, 64,128,256, 512,1024,2048],
            "ldnp_n_patches":        [2, 4, 8, 16, 32, 64,128,256, 512,1024,2048],
            "ldnp_n_scales_L":       [2, 3],
            "reduction_method":      ["LSDR", "PCA+LDA"],
            "reduction_dim":         [32,40,50,60,64,70,80,90,100,110,120,128,130,140,150,160,170,180,190,200,210,220,230,240,250,256, 260, 270, 280, 290, 300],
            "svm_C":                 [0.001,0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            "n_best":                [1,5, 10, 20, 30,40,50,60, 70,80,90,100],
        }

    # ------------------------------------------------------------------

    def run(self) -> CVResult:
        """
        Execute grid-search K-fold CV.

        Returns
        -------
        Best CVResult (highest mean top-1 accuracy).
        """
        folds = stratified_kfold_split(self.labels, self.k_folds,
                                       self.random_seed)
        param_names = sorted(self.search_space.keys())
        param_lists = [self.search_space[k] for k in param_names]
        all_combos  = list(itertools.product(*param_lists))

        total = len(all_combos)
        if self.verbose:
            print(f"[CV] Grid search over {total} parameter combinations "
                  f"with {self.k_folds} folds each.")

        for combo_idx, combo in enumerate(all_combos):
            params = dict(zip(param_names, combo))
            t0 = time.time()

            fold_t1, fold_tn = [], []
            for fold_idx, (tr_idx, vl_idx) in enumerate(folds):
                tr_imgs = [self.images[i] for i in tr_idx]
                vl_imgs = [self.images[i] for i in vl_idx]
                tr_lbl  = self.labels[tr_idx]
                vl_lbl  = self.labels[vl_idx]
                try:
                    t1, tn = evaluate_fold(
                        tr_imgs, tr_lbl, vl_imgs, vl_lbl, params
                    )
                except Exception as e:
                    if self.verbose:
                        print(f"  Fold {fold_idx} failed: {e}")
                    t1, tn = 0.0, 0.0
                fold_t1.append(t1)
                fold_tn.append(tn)

            elapsed = time.time() - t0
            result = CVResult(
                params=params,
                mean_top1=float(np.mean(fold_t1)),
                std_top1=float(np.std(fold_t1)),
                mean_topn=float(np.mean(fold_tn)),
                std_topn=float(np.std(fold_tn)),
                fold_top1=fold_t1,
                elapsed_sec=elapsed,
            )
            self.results_.append(result)

            if self.verbose:
                print(
                    f"  [{combo_idx+1:4d}/{total}] "
                    f"top1={result.mean_top1*100:.2f}% "
                    f"({elapsed:.1f}s) | {params}"
                )

        # Sort by mean top-1 accuracy
        self.results_.sort(key=lambda r: -r.mean_top1)
        self.best_params_ = self.results_[0].params

        if self.verbose:
            print("\n[CV] Best result:", self.results_[0])

        return self.results_[0]

    def top_k_results(self, k: int = 10) -> List[CVResult]:
        """Return the top-k CV results sorted by mean top-1 accuracy."""
        return self.results_[:k]
