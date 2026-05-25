"""
ouhawr/dimensionality_reduction/combined_reductions.py

Two-stage reduction pipelines:
  • PCA + LDA  : first reduce variance with PCA, then maximise
                 class separability with LDA.
  • LSDR + LDA : first project onto the sufficient subspace via LSDR,
                 then refine with LDA for maximum discriminance.

Both pipelines accept the same fit(X, y) / transform(X) API as the
single-stage reducers.
"""

import numpy as np
from .pca_reduction import PCAReducer
from .lda_reduction import LDAReducer
from .lsdr_reduction import LSRDReducer


class PCALDAReducer:
    """
    Sequential PCA → LDA pipeline.

    Parameters
    ----------
    pca_dim : intermediate dimensionality after PCA (should satisfy
              pca_dim >= n_classes - 1 to avoid LDA information loss)
    lda_dim : final dimensionality after LDA (≤ n_classes - 1)
    whiten  : whether to whiten PCA components before LDA
    """

    def __init__(
        self,
        pca_dim: int = 128,
        lda_dim: int = 64,
        whiten: bool = False,
    ):
        self.pca_dim = pca_dim
        self.lda_dim = lda_dim
        self.whiten = whiten
        self._pca = PCAReducer(n_components=pca_dim, whiten=whiten)
        self._lda = LDAReducer(n_components=lda_dim)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PCALDAReducer":
        X_pca = self._pca.fit_transform(X)
        self._lda.fit(X_pca, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._lda.transform(self._pca.transform(X))

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)

    @property
    def output_dim(self) -> int:
        return self._lda.output_dim

    def get_params(self) -> dict:
        return dict(pca_dim=self.pca_dim, lda_dim=self.lda_dim,
                    whiten=self.whiten)


class LSRDLDAReducer:
    """
    Sequential LSDR → LDA pipeline.

    Parameters
    ----------
    lsdr_dim : dimensionality of the sufficient subspace (central subspace)
    lda_dim  : final dimensionality after LDA (≤ n_classes - 1)
    ridge    : regularisation for LSDR within-class scatter
    """

    def __init__(
        self,
        lsdr_dim: int = 128,
        lda_dim: int = 64,
        ridge: float = 1e-4,
    ):
        self.lsdr_dim = lsdr_dim
        self.lda_dim = lda_dim
        self.ridge = ridge
        self._lsdr = LSRDReducer(n_components=lsdr_dim, ridge=ridge)
        self._lda = LDAReducer(n_components=lda_dim)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LSRDLDAReducer":
        X_lsdr = self._lsdr.fit_transform(X, y)
        self._lda.fit(X_lsdr, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._lda.transform(self._lsdr.transform(X))

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)

    @property
    def output_dim(self) -> int:
        return self._lda.output_dim

    def get_params(self) -> dict:
        return dict(lsdr_dim=self.lsdr_dim, lda_dim=self.lda_dim,
                    ridge=self.ridge)
