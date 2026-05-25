"""
ouhawr/dimensionality_reduction/lda_reduction.py

LDA-based dimensionality reduction wrapper.

Note: LDA can produce at most (n_classes - 1) components.
If n_components > n_classes - 1 the output dim is capped automatically.
"""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from typing import Optional


class LDAReducer:
    """
    Wrapper around scikit-learn LinearDiscriminantAnalysis.

    Parameters
    ----------
    n_components : target dimensionality d  (capped at n_classes - 1)
    solver       : LDA solver ('svd', 'lsqr', 'eigen')
    """

    def __init__(self, n_components: int = 64, solver: str = "svd"):
        self.n_components = n_components
        self.solver = solver
        self._lda: Optional[LinearDiscriminantAnalysis] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LDAReducer":
        n_classes = len(np.unique(y))
        n_comp = min(self.n_components, n_classes - 1, X.shape[1])
        self._lda = LinearDiscriminantAnalysis(
            n_components=n_comp, solver=self.solver
        )
        self._lda.fit(X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._lda is None:
            raise RuntimeError("Call fit() first.")
        return self._lda.transform(X).astype(np.float32)

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)

    @property
    def output_dim(self) -> int:
        return self._lda.scalings_.shape[1] if self._lda else self.n_components

    def get_params(self) -> dict:
        return dict(n_components=self.n_components, solver=self.solver)
