"""
tests/test_descriptors.py

Unit tests for LPQ+, WLD, and LDNP descriptors.

Run with:  python -m pytest tests/test_descriptors.py -v
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ouhawr.descriptors import LPQPlus, WLD, LDNP


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_synthetic_image(h=128, w=256, seed=0) -> np.ndarray:
    """Create a synthetic entropy-filtered word image (float32)."""
    rng = np.random.default_rng(seed)
    img = rng.uniform(0, 3.0, (h, w)).astype(np.float32)
    return img


@pytest.fixture
def image():
    return make_synthetic_image()


@pytest.fixture
def image_batch():
    return [make_synthetic_image(seed=i) for i in range(5)]


# ---------------------------------------------------------------------------
# LPQ+ tests
# ---------------------------------------------------------------------------

class TestLPQPlus:

    def test_fit_sets_W(self, image_batch):
        lpq = LPQPlus(window_size=5, bins_B=8, n_patches=4)
        lpq.fit(image_batch)
        assert lpq.W_ is not None
        assert lpq.W_.shape == (8, 8)

    def test_transform_output_shape(self, image_batch, image):
        lpq = LPQPlus(window_size=5, bins_B=8, n_patches=32, cell_grid=(2, 2))
        lpq.fit(image_batch)
        feat = lpq.transform(image)
        # Expected: n_patches × (cell_rows × cell_cols × 8 × B) × 2 (mean+std)
        assert feat.ndim == 1
        assert feat.shape[0] == lpq.feature_dim
        assert feat.dtype == np.float32

    def test_transform_different_B(self, image_batch, image):
        for B in [4, 8, 16]:
            lpq = LPQPlus(bins_B=B, n_patches=32)
            lpq.fit(image_batch)
            feat = lpq.transform(image)
            assert feat.shape[0] == lpq.feature_dim

    def test_fit_transform(self, image_batch):
        lpq = LPQPlus(n_patches=32)
        feats = lpq.fit_transform(image_batch)
        assert feats.shape == (len(image_batch), lpq.feature_dim)

    def test_W_manual_set(self, image):
        lpq = LPQPlus(n_patches=32)
        W = np.eye(8, dtype=np.float32)
        lpq.set_W(W)
        feat = lpq.transform(image)
        assert feat.ndim == 1

    def test_W_wrong_shape_raises(self):
        lpq = LPQPlus(n_patches=32)
        with pytest.raises(AssertionError):
            lpq.set_W(np.eye(4, dtype=np.float32))

    def test_transform_before_fit_raises(self, image):
        lpq = LPQPlus(n_patches=32)
        with pytest.raises(RuntimeError):
            lpq.transform(image)

    def test_deterministic(self, image_batch, image):
        lpq1 = LPQPlus(n_patches=32)
        lpq1.fit(image_batch)
        f1 = lpq1.transform(image)
        f2 = lpq1.transform(image)
        np.testing.assert_array_equal(f1, f2)

    def test_decorrelation_matrix_is_orthogonal(self, image_batch):
        """W should be approximately orthogonal (from PCA)."""
        lpq = LPQPlus(n_patches=32)
        lpq.fit(image_batch)
        WWT = lpq.W_ @ lpq.W_.T
        np.testing.assert_allclose(WWT, np.eye(8), atol=1e-4)

    def test_get_params(self):
        lpq = LPQPlus(window_size=7, bins_B=4, n_patches=64)
        p = lpq.get_params()
        assert p["window_size"] == 7
        assert p["bins_B"] == 4
        assert p["n_patches"] == 64


# ---------------------------------------------------------------------------
# WLD tests
# ---------------------------------------------------------------------------

class TestWLD:

    def test_transform_shape(self, image):
        wld = WLD(excitation_bins_T=8, orientation_bins_M=6,
                  n_patches=9, n_scales_L=2)
        feat = wld.transform(image)
        assert feat.ndim == 1
        assert feat.shape[0] == wld.feature_dim
        assert feat.dtype == np.float32

    def test_feature_dim_formula(self):
        """feature_dim == T × M × N × L"""
        T, M, N, L = 8, 6, 32, 2
        wld = WLD(excitation_bins_T=T, orientation_bins_M=M,
                  n_patches=N, n_scales_L=L)
        assert wld.feature_dim == T * M * N * L

    def test_varying_scales(self, image):
        for L in [1, 2, 3]:
            wld = WLD(n_patches=32, n_scales_L=L)
            feat = wld.transform(image)
            assert feat.shape[0] == wld.feature_dim

    def test_values_finite(self, image):
        wld = WLD(n_patches=32, n_scales_L=1)
        feat = wld.transform(image)
        assert np.all(np.isfinite(feat))

    def test_deterministic(self, image):
        wld = WLD(n_patches=32, n_scales_L=1)
        f1 = wld.transform(image)
        f2 = wld.transform(image)
        np.testing.assert_array_equal(f1, f2)

    def test_different_images_give_different_features(self):
        wld = WLD(n_patches=32, n_scales_L=1)
        img1 = make_synthetic_image(seed=1)
        img2 = make_synthetic_image(seed=99)
        f1 = wld.transform(img1)
        f2 = wld.transform(img2)
        assert not np.allclose(f1, f2)

    def test_get_params(self):
        wld = WLD(excitation_bins_T=10, orientation_bins_M=8)
        p = wld.get_params()
        assert p["excitation_bins_T"] == 10
        assert p["orientation_bins_M"] == 8


# ---------------------------------------------------------------------------
# LDNP tests
# ---------------------------------------------------------------------------

class TestLDNP:

    def test_transform_shape(self, image):
        ldnp = LDNP(n_patches=32, n_scales_L=2)
        feat = ldnp.transform(image)
        assert feat.ndim == 1
        assert feat.shape[0] == ldnp.feature_dim
        assert feat.dtype == np.float32

    def test_feature_dim_formula(self):
        """feature_dim == N × L × N_CODES (64)"""
        N, L = 32, 3
        ldnp = LDNP(n_patches=N, n_scales_L=L)
        assert ldnp.feature_dim == N * L * ldnp.N_CODES

    def test_varying_scales(self, image):
        for L in [1, 2, 3]:
            ldnp = LDNP(n_patches=32, n_scales_L=L)
            feat = ldnp.transform(image)
            assert feat.shape[0] == ldnp.feature_dim

    def test_values_in_range(self, image):
        """Histograms are non-negative and each sums to ≤ 1."""
        ldnp = LDNP(n_patches=32, n_scales_L=1)
        feat = ldnp.transform(image)
        assert np.all(feat >= 0)

    def test_values_finite(self, image):
        ldnp = LDNP(n_patches=32, n_scales_L=1)
        feat = ldnp.transform(image)
        assert np.all(np.isfinite(feat))

    def test_deterministic(self, image):
        ldnp = LDNP(n_patches=32, n_scales_L=1)
        f1 = ldnp.transform(image)
        f2 = ldnp.transform(image)
        np.testing.assert_array_equal(f1, f2)

    def test_different_images_differ(self):
        ldnp = LDNP(n_patches=32, n_scales_L=1)
        img1 = make_synthetic_image(seed=1)
        img2 = make_synthetic_image(seed=99)
        f1 = ldnp.transform(img1)
        f2 = ldnp.transform(img2)
        assert not np.allclose(f1, f2)

    def test_get_params(self):
        ldnp = LDNP(n_patches=64, n_scales_L=2)
        p = ldnp.get_params()
        assert p["n_patches"] == 64
        assert p["n_scales_L"] == 2
