"""
ouhawr/fusion/dca_fusion.py

Discriminant Correlation Analysis (DCA) and its multi-set extension
(MDCA) for feature-level fusion.

Reference:
  Haghighat, Abdel-Mottaleb & Alhalabi (2016). "Discriminant Correlation
  Analysis: Real-Time Feature Level Fusion for Multimodal Biometric
  Recognition." IEEE TIFIS, 11(9), 1984-1996.

DCA fuses two feature matrices by:
  1. Unitising the between-class scatter of each set (removes between-class
     correlations and ensures class structure is preserved).
  2. Unitising the between-set covariance (maximises within-class
     correlation across the two sets).
  3. Concatenating the two transformed feature vectors → FFV.

MDCA generalises DCA to three (or more) sets by applying DCA
sequentially on pairs until a single fused vector is obtained.

This module implements the sequential pairwise MDCA strategy used in
the paper (Section 4.4, Fig. 6).
"""

import numpy as np
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Single DCA fusion of two matrices
# ---------------------------------------------------------------------------

class DCA:
    """
    Discriminant Correlation Analysis for two feature sets.

    After fit(X, Y, y), calling transform(X, Y) returns the concatenated
    fused descriptor [X*ᵀ ; Y*ᵀ]ᵀ for each sample.

    Parameters
    ----------
    n_components : output dimensionality s per branch (None → maximum feasible)
    """

    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components
        self.Wx_: Optional[np.ndarray] = None   # (p, s) transform for X
        self.Wy_: Optional[np.ndarray] = None   # (q, s) transform for Y

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        y: np.ndarray,
    ) -> "DCA":
        """
        Estimate DCA transformation matrices.

        Parameters
        ----------
        X : (n, p) first feature matrix
        Y : (n, q) second feature matrix
        y : (n,)   integer class labels (same for both sets)
        """
        X = X.astype(np.float64)
        Y = Y.astype(np.float64)
        n = X.shape[0]

        # --- Step 1: compute between-class scatter matrices ---
        Phi_bx = self._between_class_sqrt(X, y)  # (p, c)
        Phi_by = self._between_class_sqrt(Y, y)  # (q, c)

        # --- Step 2: unitise Sbx and Sby ---
        Wbx = self._unitise(Phi_bx)  # (p, s1)
        Wby = self._unitise(Phi_by)  # (q, s2)

        # Determine s = min(s1, s2, n_components)
        s = min(Wbx.shape[1], Wby.shape[1])
        if self.n_components is not None:
            s = min(s, self.n_components)

        Wbx = Wbx[:, :s]
        Wby = Wby[:, :s]

        # --- Step 3: project to between-class space ---
        Xp = (Wbx.T @ X.T).T   # (n, s)
        Yp = (Wby.T @ Y.T).T   # (n, s)

        # --- Step 4: unitise between-set covariance ---
        S_xy = (1 / n) * Xp.T @ Yp    # (s, s)
        Wcx, Wcy = self._unitise_cross(S_xy)  # each (s, s)

        # --- Final transforms ---
        self.Wx_ = (Wbx @ Wcx).astype(np.float32)  # (p, s)
        self.Wy_ = (Wby @ Wcy).astype(np.float32)  # (q, s)
        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(
        self, X: np.ndarray, Y: np.ndarray
    ) -> np.ndarray:
        """
        Apply DCA transformations and concatenate.

        Returns
        -------
        FFV : (n, 2s) fused feature matrix
        """
        if self.Wx_ is None:
            raise RuntimeError("Call fit() first.")
        X_star = X.astype(np.float32) @ self.Wx_  # (n, s)
        Y_star = Y.astype(np.float32) @ self.Wy_  # (n, s)
        return np.concatenate([X_star, Y_star], axis=1)

    def fit_transform(
        self, X: np.ndarray, Y: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        return self.fit(X, Y, y).transform(X, Y)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _between_class_sqrt(
        X: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """
        Φ_bx = [√n₁(μ₁-μ), ..., √nc(μc-μ)]   shape (p, c)
        """
        classes = np.unique(y)
        n = X.shape[0]
        mu = X.mean(axis=0)
        cols = []
        for cls in classes:
            idx = y == cls
            n_c = idx.sum()
            mu_c = X[idx].mean(axis=0)
            cols.append(np.sqrt(n_c) * (mu_c - mu))
        return np.column_stack(cols)  # (p, c)

    @staticmethod
    def _unitise(Phi: np.ndarray) -> np.ndarray:
        """
        Compute Wb such that Wb.T @ (Phi @ Phi.T) @ Wb = I.

        When p > n we work in the (c×c) kernel space:
          eigendecompose Phi.T @ Phi → V, D
          Wb = Phi @ V @ D^{-1/2}
        """
        p, c = Phi.shape
        if p >= c:
            # Standard: eigendecompose c×c matrix
            S = Phi.T @ Phi     # (c, c)
        else:
            # p < c: work in p-dimensional space
            S = Phi @ Phi.T     # (p, p)

        eigenvalues, eigenvectors = np.linalg.eigh(S)

        # Keep only positive eigenvalues
        pos = eigenvalues > 1e-10
        eigenvalues = eigenvalues[pos]
        eigenvectors = eigenvectors[:, pos]

        D_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues))

        if p >= c:
            # Wb shape (p, rank)
            Wb = Phi @ eigenvectors @ D_inv_sqrt
        else:
            Wb = eigenvectors @ D_inv_sqrt  # (p, rank)

        return Wb

    @staticmethod
    def _unitise_cross(
        S_xy: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Wcx, Wcy such that Wcx.T @ S_xy @ Wcy = I.
        Uses SVD: S_xy = U Σ V.T  →  Wcx = U Σ^{-1/2}, Wcy = V Σ^{-1/2}
        """
        U, sigma, Vt = np.linalg.svd(S_xy, full_matrices=False)
        pos = sigma > 1e-10
        sigma = sigma[pos]
        U = U[:, pos]
        Vt = Vt[pos, :]

        S_inv_sqrt = np.diag(1.0 / np.sqrt(sigma))
        Wcx = U @ S_inv_sqrt    # (s, rank)
        Wcy = Vt.T @ S_inv_sqrt  # (s, rank)
        return Wcx, Wcy


# ---------------------------------------------------------------------------
# MDCA — sequential pairwise DCA for 3 feature sets
# ---------------------------------------------------------------------------

class MDCA:
    """
    Multiset Discriminant Correlation Analysis for three feature sets,
    implemented as two sequential DCA applications (Fig. 6 in paper):

      Round 1: DCA(F_i1, F_i2)  →  FFV1   (2s₁-dim)
      Round 2: DCA(FFV1, F_i3)  →  FFV    (2s₂-dim)

    where i1, i2, i3 are ordered by descending rank (as in the paper).

    Parameters
    ----------
    n_components : target dimension s per DCA branch (None → max feasible)
    """

    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components
        self._dca1 = DCA(n_components=n_components)
        self._dca2 = DCA(n_components=n_components)
        self._order: Optional[Tuple[int, int, int]] = None

    def fit(
        self,
        F1: np.ndarray,
        F2: np.ndarray,
        F3: np.ndarray,
        y: np.ndarray,
    ) -> "MDCA":
        """
        Parameters
        ----------
        F1, F2, F3 : (n, p_i) reduced feature matrices for the three descriptors
        y          : (n,) integer class labels
        """
        # Order by descending rank (paper: i1 = argmax rank, etc.)
        ranks = [np.linalg.matrix_rank(F) for F in [F1, F2, F3]]
        order = sorted(range(3), key=lambda i: -ranks[i])
        self._order = tuple(order)
        sets = [F1, F2, F3]
        A = sets[order[0]]
        B = sets[order[1]]
        C = sets[order[2]]

        # Round 1: fuse A and B
        FFV1 = self._dca1.fit_transform(A, B, y)

        # Round 2: fuse FFV1 with C
        self._dca2.fit(FFV1, C, y)
        return self

    def transform(
        self,
        F1: np.ndarray,
        F2: np.ndarray,
        F3: np.ndarray,
    ) -> np.ndarray:
        """
        Apply fitted MDCA transforms.

        Returns
        -------
        FFV : (n, 2s₂) fused feature matrix
        """
        if self._order is None:
            raise RuntimeError("Call fit() first.")
        sets = [F1, F2, F3]
        A = sets[self._order[0]].astype(np.float32)
        B = sets[self._order[1]].astype(np.float32)
        C = sets[self._order[2]].astype(np.float32)

        FFV1 = self._dca1.transform(A, B)
        FFV = self._dca2.transform(FFV1, C)
        return FFV

    def fit_transform(
        self,
        F1: np.ndarray,
        F2: np.ndarray,
        F3: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        return self.fit(F1, F2, F3, y).transform(F1, F2, F3)
