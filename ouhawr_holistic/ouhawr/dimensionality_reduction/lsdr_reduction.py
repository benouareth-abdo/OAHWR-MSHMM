"""
ouhawr/dimensionality_reduction/lsdr_reduction.py

Likelihood-Based Sufficient Dimension Reduction (LSDR) via the
Structured Principal Fitted Components (SPFC) model.

Reference:
  Cook & Forzani (2008). "Principal Fitted Components for Dimension
  Reduction in Regression." Statistical Science, 23(4), 485-501.

The SPFC model estimates a p×d projection matrix Γ whose columns span
the central subspace S, such that Y ⊥ X | Γᵀ X (sufficiency).

Model (inverse regression form):
    X | Y=y ~ N(Γ η(y), Δ)

where:
  - η(y) is the inverse mean function   (class-conditional mean of ΓᵀX)
  - Δ is the common conditional covariance (independent of y)

MLE of Γ — implemented algorithm (Cook & Forzani, Sect. 3):
  1. Compute class-conditional means μ_c = E[X|Y=c]
  2. Compute within-class covariance Σ_w
  3. Solve the generalised eigenproblem:
        Σ_b v = λ Σ_w v
     where Σ_b is the between-class covariance
  4. Take the top-d eigenvectors as columns of Γ

This is equivalent to standard Fisher's LDA in the p < n regime,
but the derivation via inverse regression provides the sufficiency
guarantee: R(X) = ΓᵀX preserves all Y-relevant information in X.

In the p >> n regime the within-class scatter is regularised with a
small ridge before solving the eigenproblem.
"""

import numpy as np
from typing import Optional


class LSRDReducer:
    """
    LSDR / SPFC dimensionality reducer.

    Parameters
    ----------
    n_components : d — dimension of the central subspace
    max_iter     : maximum iterations (for iterative refinement, if used)
    tol          : convergence tolerance
    ridge        : regularisation added to within-class scatter (for p >> n)
    """

    def __init__(
        self,
        n_components: int = 64,
        max_iter: int = 200,
        tol: float = 1e-6,
        ridge: float = 1e-4,
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.ridge = ridge

        self.Gamma_: Optional[np.ndarray] = None   # shape (p, d)
        self._mean_: Optional[np.ndarray] = None   # global mean for centering

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LSRDReducer":
        """
        Estimate Γ via the SPFC maximum-likelihood procedure.

        Parameters
        ----------
        X : (n, p) feature matrix
        y : (n,)   integer class labels
        """
        X = X.astype(np.float64)
        n, p = X.shape
        classes = np.unique(y)
        c = len(classes)
        d = min(self.n_components, c - 1, p)

        # Global mean
        mu = X.mean(axis=0)
        self._mean_ = mu.astype(np.float32)
        Xc = X - mu

        # Class-conditional means and within-class scatter
        Sigma_b = np.zeros((p, p), dtype=np.float64)
        Sigma_w = np.zeros((p, p), dtype=np.float64)

        for cls in classes:
            idx = y == cls
            n_c = idx.sum()
            mu_c = Xc[idx].mean(axis=0)
            diff = Xc[idx] - mu_c                          # (n_c, p)
            Sigma_w += diff.T @ diff                        # within-class
            Sigma_b += n_c * np.outer(mu_c, mu_c)          # between-class

        Sigma_w /= n
        Sigma_b /= n

        # Regularise within-class scatter for p >> n stability
        Sigma_w += self.ridge * np.eye(p)

        # Generalised eigenproblem: Σ_b v = λ Σ_w v
        # → Σ_w^{-1} Σ_b v = λ v
        try:
            # Cholesky of Σ_w for numerical stability
            L = np.linalg.cholesky(Sigma_w)
            L_inv = np.linalg.inv(L)
            M = L_inv @ Sigma_b @ L_inv.T
            eigenvalues, eigenvectors = np.linalg.eigh(M)
            # eigenvectors of the transformed problem → back-transform
            V = L_inv.T @ eigenvectors
        except np.linalg.LinAlgError:
            # Fallback: direct eigenproblem via eig
            Sw_inv = np.linalg.pinv(Sigma_w)
            M = Sw_inv @ Sigma_b
            eigenvalues, V = np.linalg.eig(M)
            eigenvalues = eigenvalues.real
            V = V.real

        # Sort by descending eigenvalue and keep top d
        order = np.argsort(-eigenvalues)
        V = V[:, order[:d]]

        # Orthonormalise via QR
        Gamma, _ = np.linalg.qr(V)
        self.Gamma_ = Gamma[:, :d].astype(np.float32)   # (p, d)
        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project X onto the estimated central subspace.
        R(X) = Γᵀ (X − μ)   shape (n, d)
        """
        if self.Gamma_ is None:
            raise RuntimeError("Call fit() first.")
        Xc = X.astype(np.float64) - self._mean_.astype(np.float64)
        return (Xc @ self.Gamma_).astype(np.float32)

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)

    @property
    def output_dim(self) -> int:
        return self.Gamma_.shape[1] if self.Gamma_ is not None else self.n_components

    def get_params(self) -> dict:
        return dict(
            n_components=self.n_components,
            max_iter=self.max_iter,
            tol=self.tol,
            ridge=self.ridge,
        )
