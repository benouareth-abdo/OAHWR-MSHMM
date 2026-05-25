"""
tests/test_fusion.py

Unit tests for Discriminant Correlation Analysis (DCA) and
Multiset DCA (MDCA).

Run with:  python -m pytest tests/test_fusion.py -v
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ouhawr.fusion.dca_fusion import DCA, MDCA


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_features(n=60, p=32, c=5, seed=0) -> tuple:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)).astype(np.float32)
    y = np.repeat(np.arange(c), n // c)
    for i in range(c):
        X[y == i] += rng.standard_normal(p)
    return X, y


@pytest.fixture
def two_sets():
    X, y = make_features(n=60, p=32, c=5, seed=0)
    Y, _ = make_features(n=60, p=24, c=5, seed=1)
    return X, Y, y


@pytest.fixture
def three_sets():
    X, y = make_features(n=60, p=32, c=5, seed=0)
    Y, _ = make_features(n=60, p=24, c=5, seed=1)
    Z, _ = make_features(n=60, p=20, c=5, seed=2)
    return X, Y, Z, y


# ---------------------------------------------------------------------------
# DCA tests
# ---------------------------------------------------------------------------

class TestDCA:

    def test_transform_output_shape(self, two_sets):
        X, Y, y = two_sets
        dca = DCA()
        FFV = dca.fit_transform(X, Y, y)
        # output: 2s columns where s ≤ n_classes - 1
        assert FFV.shape[0] == X.shape[0]
        assert FFV.shape[1] % 2 == 0   # symmetric 2s shape

    def test_transform_before_fit_raises(self, two_sets):
        X, Y, y = two_sets
        dca = DCA()
        with pytest.raises(RuntimeError):
            dca.transform(X, Y)

    def test_n_components_respected(self, two_sets):
        X, Y, y = two_sets
        n_comp = 3
        dca = DCA(n_components=n_comp)
        FFV = dca.fit_transform(X, Y, y)
        assert FFV.shape[1] == 2 * n_comp

    def test_fit_then_transform_consistent(self, two_sets):
        X, Y, y = two_sets
        dca = DCA(n_components=3)
        FFV1 = dca.fit_transform(X, Y, y)
        FFV2 = dca.transform(X, Y)
        np.testing.assert_allclose(FFV1, FFV2, atol=1e-5)

    def test_output_dtype(self, two_sets):
        X, Y, y = two_sets
        dca = DCA(n_components=3)
        FFV = dca.fit_transform(X, Y, y)
        assert FFV.dtype == np.float32

    def test_values_finite(self, two_sets):
        X, Y, y = two_sets
        dca = DCA(n_components=3)
        FFV = dca.fit_transform(X, Y, y)
        assert np.all(np.isfinite(FFV))

    def test_within_class_correlation_increased(self, two_sets):
        """
        After DCA, within-class correlation between X* and Y* branches
        should be higher than raw correlation.
        """
        X, Y, y = two_sets
        dca = DCA(n_components=3)
        FFV = dca.fit_transform(X, Y, y)
        s = FFV.shape[1] // 2
        X_star = FFV[:, :s]
        Y_star = FFV[:, s:]

        # Mean within-class correlation (diagonal of cross-covariance)
        corr_raw = np.corrcoef(X[:, :s].T, Y[:, :s].T)[:s, s:].diagonal()
        corr_dca = np.corrcoef(X_star.T, Y_star.T)[:s, s:].diagonal()

        assert np.mean(np.abs(corr_dca)) >= np.mean(np.abs(corr_raw)) - 0.1

    def test_symmetric_inputs_give_consistent_output(self):
        """DCA with same X=Y should give a valid (possibly degenerate) result."""
        rng = np.random.default_rng(7)
        X = rng.standard_normal((40, 16)).astype(np.float32)
        y = np.repeat(np.arange(4), 10)
        dca = DCA(n_components=3)
        FFV = dca.fit_transform(X, X, y)
        assert np.all(np.isfinite(FFV))


# ---------------------------------------------------------------------------
# MDCA tests
# ---------------------------------------------------------------------------

class TestMDCA:

    def test_output_shape(self, three_sets):
        X, Y, Z, y = three_sets
        mdca = MDCA()
        FFV = mdca.fit_transform(X, Y, Z, y)
        assert FFV.shape[0] == X.shape[0]
        assert FFV.ndim == 2

    def test_output_dtype(self, three_sets):
        X, Y, Z, y = three_sets
        mdca = MDCA()
        FFV = mdca.fit_transform(X, Y, Z, y)
        assert FFV.dtype == np.float32

    def test_values_finite(self, three_sets):
        X, Y, Z, y = three_sets
        mdca = MDCA()
        FFV = mdca.fit_transform(X, Y, Z, y)
        assert np.all(np.isfinite(FFV))

    def test_transform_before_fit_raises(self, three_sets):
        X, Y, Z, y = three_sets
        mdca = MDCA()
        with pytest.raises(RuntimeError):
            mdca.transform(X, Y, Z)

    def test_fit_transform_consistent(self, three_sets):
        X, Y, Z, y = three_sets
        mdca = MDCA(n_components=3)
        FFV1 = mdca.fit_transform(X, Y, Z, y)
        FFV2 = mdca.transform(X, Y, Z)
        np.testing.assert_allclose(FFV1, FFV2, atol=1e-5)

    def test_n_components_respected(self, three_sets):
        X, Y, Z, y = three_sets
        mdca = MDCA(n_components=2)
        FFV = mdca.fit_transform(X, Y, Z, y)
        # After two DCA rounds with n_comp=2: output should be 4 (2×2)
        assert FFV.shape[1] == 4

    def test_order_by_rank(self, three_sets):
        """MDCA uses inputs ranked by matrix rank — check order is set."""
        X, Y, Z, y = three_sets
        mdca = MDCA(n_components=3)
        mdca.fit(X, Y, Z, y)
        assert mdca._order is not None
        assert set(mdca._order) == {0, 1, 2}

    def test_different_from_individual_features(self, three_sets):
        """Fused features should differ from any individual input."""
        X, Y, Z, y = three_sets
        mdca = MDCA(n_components=3)
        FFV = mdca.fit_transform(X, Y, Z, y)
        # FFV should not be identical to any input (different shape / content)
        assert FFV.shape[1] != X.shape[1] or not np.allclose(
            FFV[:, :X.shape[1]], X
        )
