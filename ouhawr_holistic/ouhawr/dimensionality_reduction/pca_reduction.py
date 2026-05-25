"""
ouhawr/dimensionality_reduction/pca_reduction.py

PCA-based dimensionality reduction wrapper.
"""

import numpy as np
from sklearn.decomposition import PCA as SklearnPCA
from typing import Optional


class PCAReducer:
    """
    Wrapper around scikit-learn PCA.

    Parameters
    ----------
    n_components : target dimensionality d
    whiten       : whether to whiten the components
    """

    def __init__(self, n_components: int = 64, whiten: bool = False):
        self.n_components = n_components
        self.whiten = whiten
        self._pca: Optional[SklearnPCA] = None

    def fit(self, X: np.ndarray) -> "PCAReducer":
        n_comp = min(self.n_components, X.shape[0], X.shape[1])
        self._pca = SklearnPCA(n_components=n_comp, whiten=self.whiten)
        self._pca.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("Call fit() first.")
        return self._pca.transform(X).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    @property
    def output_dim(self) -> int:
        return self._pca.n_components_ if self._pca else self.n_components

    def get_params(self) -> dict:
        return dict(n_components=self.n_components, whiten=self.whiten)
