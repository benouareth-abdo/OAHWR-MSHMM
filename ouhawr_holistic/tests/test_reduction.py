"""
tests/test_reduction.py

Unit tests for all dimensionality reduction methods:
  PCA, LDA, LSDR, PCA+LDA, LSDR+LDA

Run with:  python -m pytest tests/test_reduction.py -v
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ouhawr.dimensionality_reduction import (
    PCAReducer, LDAReducer, LSRDReducer,
    PCALDAReducer, LSRDLDAReducer,
    make_reducer, is_supervised,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_data():
    """50 samples, 200-dim features, 5 classes."""
    rng = np.random.default_rng(42)
    n, p, c = 50, 200, 5
    X = rng.standard_normal((n, p)).astype(np.float32)
    # Make classes separable by adding class-specific offset
    y = np.repeat(np.arange(c), n // c)
    for i in range(c):
        X[y == i] += rng.standard_normal(p) * 2
    return X, y


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

class TestPCA:

    def test_output_shape(self, synthetic_data):
        X, _ = synthetic_data
        r = PCAReducer(n_components=32)
        R = r.fit_transform(X)
        assert R.shape == (X.shape[0], 32)

    def test_output_dtype(self, synthetic_data):
        X, _ = synthetic_data
        r = PCAReducer(n_components=16)
        R = r.fit_transform(X)
        assert R.dtype == np.float32

    def test_capped_at_input_dim(self, synthetic_data):
        X, _ = synthetic_data
        r = PCAReducer(n_components=9999)  # More than available
        R = r.fit_transform(X)
        assert R.shape[1] <= min(X.shape)

    def test_transform_before_fit_raises(self, synthetic_data):
        X, _ = synthetic_data
        r = PCAReducer(n_components=8)
        with pytest.raises(RuntimeError):
            r.transform(X)

    def test_fit_then_transform_consistent(self, synthetic_data):
        X, _ = synthetic_data
        r = PCAReducer(n_components=16)
        R1 = r.fit_transform(X)
        R2 = r.transform(X)
        np.testing.assert_allclose(R1, R2, atol=1e-5)

    def test_output_dim_property(self, synthetic_data):
        X, _ = synthetic_data
        r = PCAReducer(n_components=20)
        r.fit(X)
        assert r.output_dim == 20


# ---------------------------------------------------------------------------
# LDA
# ---------------------------------------------------------------------------

class TestLDA:

    def test_output_shape_capped(self, synthetic_data):
        X, y = synthetic_data
        # LDA: at most n_classes - 1 = 4 dimensions
        r = LDAReducer(n_components=100)
        R = r.fit_transform(X, y)
        n_classes = len(np.unique(y))
        assert R.shape == (X.shape[0], min(100, n_classes - 1))

    def test_output_dtype(self, synthetic_data):
        X, y = synthetic_data
        r = LDAReducer(n_components=4)
        R = r.fit_transform(X, y)
        assert R.dtype == np.float32

    def test_transform_before_fit_raises(self, synthetic_data):
        X, y = synthetic_data
        r = LDAReducer(n_components=4)
        with pytest.raises(RuntimeError):
            r.transform(X)

    def test_classes_separable_in_reduced_space(self, synthetic_data):
        """LDA should produce some class separation."""
        X, y = synthetic_data
        r = LDAReducer(n_components=4)
        R = r.fit_transform(X, y)
        # Check that class means differ in reduced space
        means = [R[y == c].mean(axis=0) for c in np.unique(y)]
        diffs = [np.linalg.norm(means[i] - means[j])
                 for i in range(len(means))
                 for j in range(i + 1, len(means))]
        assert max(diffs) > 0.1


# ---------------------------------------------------------------------------
# LSDR
# ---------------------------------------------------------------------------

class TestLSDR:

    def test_output_shape(self, synthetic_data):
        X, y = synthetic_data
        r = LSRDReducer(n_components=4)
        R = r.fit_transform(X, y)
        assert R.shape == (X.shape[0], 4)

    def test_output_dtype(self, synthetic_data):
        X, y = synthetic_data
        r = LSRDReducer(n_components=4)
        R = r.fit_transform(X, y)
        assert R.dtype == np.float32

    def test_transform_before_fit_raises(self, synthetic_data):
        X, _ = synthetic_data
        r = LSRDReducer(n_components=4)
        with pytest.raises(RuntimeError):
            r.transform(X)

    def test_gamma_shape(self, synthetic_data):
        X, y = synthetic_data
        r = LSRDReducer(n_components=4)
        r.fit(X, y)
        assert r.Gamma_.shape == (X.shape[1], 4)

    def test_projection_orthogonal(self, synthetic_data):
        """Columns of Gamma should be orthonormal."""
        X, y = synthetic_data
        r = LSRDReducer(n_components=4)
        r.fit(X, y)
        GtG = r.Gamma_.T.astype(np.float64) @ r.Gamma_.astype(np.float64)
        np.testing.assert_allclose(GtG, np.eye(4), atol=1e-4)

    def test_centering_applied(self, synthetic_data):
        """transform(X).mean should be near zero."""
        X, y = synthetic_data
        r = LSRDReducer(n_components=4)
        R = r.fit_transform(X, y)
        assert np.abs(R.mean()) < 1.0  # After centering

    def test_consistent_transform(self, synthetic_data):
        X, y = synthetic_data
        r = LSRDReducer(n_components=4)
        R1 = r.fit_transform(X, y)
        R2 = r.transform(X)
        np.testing.assert_allclose(R1, R2, atol=1e-5)


# ---------------------------------------------------------------------------
# PCA + LDA
# ---------------------------------------------------------------------------

class TestPCALDA:

    def test_output_shape(self, synthetic_data):
        X, y = synthetic_data
        r = PCALDAReducer(pca_dim=20, lda_dim=4)
        R = r.fit_transform(X, y)
        n_classes = len(np.unique(y))
        assert R.shape[0] == X.shape[0]
        assert R.shape[1] == min(4, n_classes - 1)

    def test_values_finite(self, synthetic_data):
        X, y = synthetic_data
        r = PCALDAReducer(pca_dim=20, lda_dim=4)
        R = r.fit_transform(X, y)
        assert np.all(np.isfinite(R))


# ---------------------------------------------------------------------------
# LSDR + LDA
# ---------------------------------------------------------------------------

class TestLSRDLDA:

    def test_output_shape(self, synthetic_data):
        X, y = synthetic_data
        r = LSRDLDAReducer(lsdr_dim=8, lda_dim=4)
        R = r.fit_transform(X, y)
        n_classes = len(np.unique(y))
        assert R.shape[0] == X.shape[0]
        assert R.shape[1] == min(4, n_classes - 1)

    def test_values_finite(self, synthetic_data):
        X, y = synthetic_data
        r = LSRDLDAReducer(lsdr_dim=8, lda_dim=4)
        R = r.fit_transform(X, y)
        assert np.all(np.isfinite(R))


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------

class TestFactory:

    @pytest.mark.parametrize("method", ["PCA", "LDA", "LSDR",
                                         "PCA+LDA", "LSDR+LDA"])
    def test_make_reducer_runs(self, synthetic_data, method):
        X, y = synthetic_data
        r = make_reducer(method, n_components=4)
        supervised = is_supervised(method)
        if supervised:
            R = r.fit_transform(X, y)
        else:
            R = r.fit_transform(X)
        assert R.shape[0] == X.shape[0]
        assert R.shape[1] >= 1

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            make_reducer("UNKNOWN", n_components=8)

    def test_is_supervised_pca_false(self):
        assert is_supervised("PCA") is False

    def test_is_supervised_lda_true(self):
        assert is_supervised("LDA") is True

    def test_is_supervised_lsdr_true(self):
        assert is_supervised("LSDR") is True
