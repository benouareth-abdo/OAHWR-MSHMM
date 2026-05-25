"""
ouhawr/dimensionality_reduction/reducer_factory.py

Factory that instantiates any supported reduction method by name.

Supported methods
-----------------
  "PCA"      — Principal Component Analysis
  "LDA"      — Linear Discriminant Analysis
  "LSDR"     — Likelihood-Based Sufficient Dimension Reduction (SPFC)
  "PCA+LDA"  — PCA followed by LDA
  "LSDR+LDA" — LSDR followed by LDA
"""

from .pca_reduction import PCAReducer
from .lda_reduction import LDAReducer
from .lsdr_reduction import LSRDReducer
from .combined_reductions import PCALDAReducer, LSRDLDAReducer


_SUPERVISED = {"LDA", "LSDR", "PCA+LDA", "LSDR+LDA"}


def make_reducer(method: str, n_components: int = 64, **kwargs):
    """
    Instantiate a dimensionality reducer.

    Parameters
    ----------
    method       : one of {"PCA", "LDA", "LSDR", "PCA+LDA", "LSDR+LDA"}
    n_components : target output dimension d
    **kwargs     : extra arguments forwarded to the constructor

    Returns
    -------
    reducer object with fit(X[, y]) and transform(X) methods.
    """
    method = method.upper().replace(" ", "")
    if method == "PCA":
        return PCAReducer(n_components=n_components,
                          whiten=kwargs.get("whiten", False))
    elif method == "LDA":
        return LDAReducer(n_components=n_components,
                          solver=kwargs.get("solver", "svd"))
    elif method == "LSDR":
        return LSRDReducer(n_components=n_components,
                           ridge=kwargs.get("ridge", 1e-4))
    elif method == "PCA+LDA":
        pca_dim = kwargs.get("pca_dim", max(n_components * 2, 128))
        return PCALDAReducer(pca_dim=pca_dim, lda_dim=n_components)
    elif method == "LSDR+LDA":
        lsdr_dim = kwargs.get("lsdr_dim", max(n_components * 2, 128))
        return LSRDLDAReducer(lsdr_dim=lsdr_dim, lda_dim=n_components)
    else:
        raise ValueError(
            f"Unknown reduction method '{method}'. "
            f"Choose from: PCA, LDA, LSDR, PCA+LDA, LSDR+LDA"
        )


def is_supervised(method: str) -> bool:
    """Return True if the method requires class labels y during fit."""
    return method.upper().replace(" ", "") in _SUPERVISED
