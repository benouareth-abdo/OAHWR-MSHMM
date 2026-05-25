"""
ouhawr/pipeline.py

End-to-end holistic OUHAWR pipeline.

Encapsulates:
  preprocessing → LPQ+/WLD/LDNP extraction → LSDR/PCA/LDA reduction
  → MDCA fusion → SVM classification → N-best output

Usage
-----
  pipeline = HolisticPipeline.from_config("configs/default_config.yaml")
  pipeline.fit(train_images, train_labels)
  nbest = pipeline.predict_nbest(test_images)
  top1  = pipeline.predict(test_images)
"""

import numpy as np
import yaml
import joblib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .preprocessing.image_preprocessing import ImagePreprocessor
from .descriptors import LPQPlus, WLD, LDNP
from .dimensionality_reduction import make_reducer, is_supervised
from .fusion.dca_fusion import MDCA
from .classification.svm_classifier import SVMClassifier
from .utils.metrics import top1_accuracy, topn_accuracy, recognition_report


class HolisticPipeline:
    """
    Full holistic OUHAWR pipeline.

    Parameters
    ----------
    config : dict of hyper-parameters (see configs/default_config.yaml)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._build_components()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_components(self) -> None:
        cfg = self.config

        # Preprocessing
        pre_cfg = cfg.get("preprocessing", {})
        self.preprocessor = ImagePreprocessor(
            target_width=pre_cfg.get("target_width", 256),
            target_height=pre_cfg.get("target_height", 128),
            entropy_window=pre_cfg.get("entropy_window", 7),
        )

        # LPQ+
        lpq_cfg = cfg.get("lpq_plus", {})
        self.lpq = LPQPlus(
            window_size=lpq_cfg.get("window_size", 5),
            freq_scalar=lpq_cfg.get("freq_scalar", 1.0),
            quant_alpha=lpq_cfg.get("quant_alpha", 1.0),
            bins_B=lpq_cfg.get("bins_B", 8),
            n_patches=lpq_cfg.get("n_patches", 32),
            cell_grid=tuple(lpq_cfg.get("cell_grid", [2, 2])),
        )

        # WLD
        wld_cfg = cfg.get("wld", {})
        self.wld = WLD(
            neighbourhood_size=wld_cfg.get("neighbourhood_size", 5),
            excitation_bins_T=wld_cfg.get("excitation_bins_T", 8),
            orientation_bins_M=wld_cfg.get("orientation_bins_M", 6),
            dominant_orient_K=wld_cfg.get("dominant_orient_K", 2),
            n_patches=wld_cfg.get("n_patches", 128),
            n_scales_L=wld_cfg.get("n_scales_L", 2),
        )

        # LDNP
        ldnp_cfg = cfg.get("ldnp", {})
        self.ldnp = LDNP(
            window_size=ldnp_cfg.get("window_size", 3),
            n_patches=ldnp_cfg.get("n_patches", 128),
            n_scales_L=ldnp_cfg.get("n_scales_L", 3),
        )

        # Dimensionality reduction (one per descriptor)
        red_cfg = cfg.get("reduction", {})
        method  = red_cfg.get("method", "LSDR")
        dim     = red_cfg.get("target_dim", 64)
        self.reducer_lpq  = make_reducer(method, dim)
        self.reducer_wld  = make_reducer(method, dim)
        self.reducer_ldnp = make_reducer(method, dim)
        self._reduction_supervised = is_supervised(method)

        # MDCA fusion
        fus_cfg = cfg.get("fusion", {})
        self.mdca = MDCA(n_components=fus_cfg.get("n_components", None))

        # SVM
        svm_cfg = cfg.get("svm", {})
        self.clf = SVMClassifier(
            C=svm_cfg.get("C", 1.0),
            n_best=svm_cfg.get("n_best", 100),
        )

        self._fitted = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        images: List[np.ndarray],
        labels: np.ndarray,
        preprocess: bool = False,
    ) -> "HolisticPipeline":
        """
        Fit all pipeline components on training data.

        Parameters
        ----------
        images     : list of word images (entropy-filtered if preprocess=False,
                     otherwise raw images)
        labels     : string class labels
        preprocess : if True, run the preprocessing step first
        """
        if preprocess:
            images = [self.preprocessor.preprocess(img) for img in images]

        print(f"[Pipeline] Fitting LPQ+ decorrelation matrix on "
              f"{len(images)} images …")
        self.lpq.fit(images)

        print("[Pipeline] Extracting features …")
        F_lpq, F_wld, F_ldnp = self._extract_all(images)

        print("[Pipeline] Fitting dimensionality reducers …")
        if self._reduction_supervised:
            R_lpq  = self.reducer_lpq.fit_transform(F_lpq, labels)
            R_wld  = self.reducer_wld.fit_transform(F_wld, labels)
            R_ldnp = self.reducer_ldnp.fit_transform(F_ldnp, labels)
        else:
            R_lpq  = self.reducer_lpq.fit_transform(F_lpq)
            R_wld  = self.reducer_wld.fit_transform(F_wld)
            R_ldnp = self.reducer_ldnp.fit_transform(F_ldnp)

        print("[Pipeline] Fitting MDCA fusion …")
        FFV = self.mdca.fit_transform(R_lpq, R_wld, R_ldnp, labels)

        print(f"[Pipeline] Training SVM (C={self.clf.C}) …")
        self.clf.fit(FFV, labels)

        self._fitted = True
        print("[Pipeline] Training complete.")
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(
        self,
        images: List[np.ndarray],
        preprocess: bool = False,
    ) -> np.ndarray:
        """Return top-1 predicted class for each image."""
        nbest = self.predict_nbest(images, preprocess=preprocess)
        return np.array([hyps[0][0] for hyps in nbest])

    def predict_nbest(
        self,
        images: List[np.ndarray],
        preprocess: bool = False,
    ) -> list:
        """Return N-best hypotheses for each image."""
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
        if preprocess:
            images = [self.preprocessor.preprocess(img) for img in images]
        FFV = self._transform(images)
        return self.clf.predict_nbest(FFV)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        images: List[np.ndarray],
        labels: np.ndarray,
        preprocess: bool = False,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate on a test set.

        Returns dict with keys: top1, top_n.
        """
        nbest  = self.predict_nbest(images, preprocess=preprocess)
        y_pred = np.array([h[0][0] for h in nbest])
        t1 = top1_accuracy(labels, y_pred)
        tn = topn_accuracy(labels, nbest)
        if verbose:
            print(recognition_report(labels, y_pred, nbest))
        return {"top1": t1, "top_n": tn}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_all(
        self, images: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        F_lpq  = np.vstack([self.lpq.transform(img)  for img in images])
        F_wld  = np.vstack([self.wld.transform(img)  for img in images])
        F_ldnp = np.vstack([self.ldnp.transform(img) for img in images])
        return F_lpq, F_wld, F_ldnp

    def _transform(self, images: List[np.ndarray]) -> np.ndarray:
        F_lpq, F_wld, F_ldnp = self._extract_all(images)
        R_lpq  = self.reducer_lpq.transform(F_lpq)
        R_wld  = self.reducer_wld.transform(F_wld)
        R_ldnp = self.reducer_ldnp.transform(F_ldnp)
        return self.mdca.transform(R_lpq, R_wld, R_ldnp)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the fitted pipeline to disk."""
        joblib.dump(self, path)
        print(f"[Pipeline] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "HolisticPipeline":
        """Load a fitted pipeline from disk."""
        pipeline = joblib.load(path)
        print(f"[Pipeline] Loaded from {path}")
        return pipeline

    @classmethod
    def from_config(cls, config_path: str) -> "HolisticPipeline":
        """Instantiate a pipeline from a YAML configuration file."""
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        return cls(config=cfg)

    def update_params(self, params: Dict[str, Any]) -> "HolisticPipeline":
        """
        Update hyper-parameters from a flat dict (e.g. best CV params)
        and rebuild components.
        """
        # Map flat param keys to config structure
        mapping = {
            "lpq_window_size":       ("lpq_plus", "window_size"),
            "lpq_bins_B":            ("lpq_plus", "bins_B"),
            "lpq_n_patches":         ("lpq_plus", "n_patches"),
            "wld_excitation_bins_T": ("wld",      "excitation_bins_T"),
            "wld_orientation_bins_M":("wld",      "orientation_bins_M"),
            "wld_n_patches":         ("wld",      "n_patches"),
            "wld_n_scales_L":        ("wld",      "n_scales_L"),
            "ldnp_n_patches":        ("ldnp",     "n_patches"),
            "ldnp_n_scales_L":       ("ldnp",     "n_scales_L"),
            "reduction_method":      ("reduction","method"),
            "reduction_dim":         ("reduction","target_dim"),
            "svm_C":                 ("svm",      "C"),
            "n_best":                ("svm",      "n_best"),
        }
        for flat_key, val in params.items():
            if flat_key in mapping:
                section, key = mapping[flat_key]
                if section not in self.config:
                    self.config[section] = {}
                self.config[section][key] = val

        self._build_components()
        return self
