"""
tests/test_pipeline.py

End-to-end integration tests for the holistic OUHAWR pipeline.

Run with:  python -m pytest tests/test_pipeline.py -v
"""

import numpy as np
import pytest
import sys
import tempfile
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ouhawr.pipeline import HolisticPipeline
from ouhawr.preprocessing.image_preprocessing import ImagePreprocessor
from ouhawr.utils.metrics import top1_accuracy, topn_accuracy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_entropy_image(h=128, w=256, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 3.0, (h, w)).astype(np.float32)


def make_synthetic_dataset(
    n_classes=6, n_per_class=8, seed=0
) -> tuple:
    """Create a tiny synthetic dataset of entropy images + labels."""
    rng = np.random.default_rng(seed)
    images, labels = [], []
    for c in range(n_classes):
        for j in range(n_per_class):
            # Give each class a distinctive mean so the SVM can separate them
            img = rng.uniform(0, 3.0, (128, 256)).astype(np.float32)
            img += c * 0.5   # class-specific offset
            images.append(img)
            labels.append(f"word_{c:02d}")
    y = np.array(labels)
    return images, y


@pytest.fixture
def tiny_dataset():
    return make_synthetic_dataset(n_classes=6, n_per_class=8)


@pytest.fixture
def minimal_config():
    """Minimal config that uses small descriptors for speed."""
    return {
        "lpq_plus":  {"window_size": 3, "bins_B": 4,  "n_patches": 32,
                       "freq_scalar": 1.0, "quant_alpha": 1.0,
                       "cell_grid": [2, 2]},
        "wld":       {"excitation_bins_T": 4, "orientation_bins_M": 4,
                       "n_patches": 128, "n_scales_L": 1},
        "ldnp":      {"n_patches": 128, "n_scales_L": 1},
        "reduction": {"method": "PCA", "target_dim": 8},
        "fusion":    {"n_components": 3},
        "svm":       {"C": 1.0, "n_best": 3},
    }


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------

class TestPreprocessor:

    def test_output_shape(self):
        pre = ImagePreprocessor(target_width=300, target_height=100,
                                entropy_window=5)
        img = np.random.randint(0, 255, (60, 150, 3), dtype=np.uint8)
        out = pre.preprocess(img)
        assert out.shape == (100, 300)
        assert out.dtype == np.float32

    def test_greyscale_input(self):
        pre = ImagePreprocessor()
        img = np.random.randint(0, 255, (80, 200), dtype=np.uint8)
        out = pre.preprocess(img)
        assert out.ndim == 2

    def test_entropy_values_positive(self):
        pre = ImagePreprocessor()
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        out = pre.preprocess(img)
        assert np.all(out >= 0)


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------

class TestPipelineConstruction:

    def test_from_config_dict(self, minimal_config):
        pipeline = HolisticPipeline(config=minimal_config)
        assert pipeline.lpq is not None
        assert pipeline.wld is not None
        assert pipeline.ldnp is not None
        assert pipeline.mdca is not None
        assert pipeline.clf is not None

    def test_default_construction(self):
        pipeline = HolisticPipeline()
        assert pipeline.lpq.M == 5       # default window
        assert pipeline.clf.n_best == 10  # default n_best

    def test_from_yaml(self, tmp_path, minimal_config):
        import yaml
        cfg_path = tmp_path / "test_config.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(minimal_config, f)
        pipeline = HolisticPipeline.from_config(str(cfg_path))
        assert pipeline is not None

    def test_update_params(self, minimal_config):
        pipeline = HolisticPipeline(config=minimal_config)
        pipeline.update_params({"svm_C": 10.0, "n_best": 5})
        assert pipeline.clf.C == 10.0
        assert pipeline.clf.n_best == 5


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TestPipelineFit:

    def test_fit_runs(self, tiny_dataset, minimal_config):
        images, labels = tiny_dataset
        pipeline = HolisticPipeline(config=minimal_config)
        pipeline.fit(images, labels)
        assert pipeline._fitted is True

    def test_fit_sets_lpq_W(self, tiny_dataset, minimal_config):
        images, labels = tiny_dataset
        pipeline = HolisticPipeline(config=minimal_config)
        pipeline.fit(images, labels)
        assert pipeline.lpq.W_ is not None

    def test_fit_sets_mdca_order(self, tiny_dataset, minimal_config):
        images, labels = tiny_dataset
        pipeline = HolisticPipeline(config=minimal_config)
        pipeline.fit(images, labels)
        assert pipeline.mdca._order is not None

    def test_fit_sets_svm_classes(self, tiny_dataset, minimal_config):
        images, labels = tiny_dataset
        pipeline = HolisticPipeline(config=minimal_config)
        pipeline.fit(images, labels)
        assert pipeline.clf.classes_ is not None
        assert len(pipeline.clf.classes_) == len(np.unique(labels))


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

class TestPipelinePredict:

    @pytest.fixture(autouse=True)
    def fitted_pipeline(self, tiny_dataset, minimal_config):
        images, labels = tiny_dataset
        self.images = images
        self.labels = labels
        self.pipeline = HolisticPipeline(config=minimal_config)
        self.pipeline.fit(images, labels)

    def test_predict_returns_labels(self):
        preds = self.pipeline.predict(self.images[:4])
        assert len(preds) == 4
        assert all(p in self.labels for p in preds)

    def test_predict_nbest_structure(self):
        nbest = self.pipeline.predict_nbest(self.images[:4])
        assert len(nbest) == 4
        for hyps in nbest:
            assert len(hyps) == self.pipeline.clf.n_best
            for label, score in hyps:
                assert isinstance(score, float)

    def test_nbest_sorted(self):
        """N-best list should be sorted by ascending distance."""
        nbest = self.pipeline.predict_nbest(self.images[:3])
        for hyps in nbest:
            scores = [s for _, s in hyps]
            assert scores == sorted(scores)

    def test_predict_before_fit_raises(self, minimal_config):
        pipeline = HolisticPipeline(config=minimal_config)
        with pytest.raises(RuntimeError):
            pipeline.predict([self.images[0]])

    def test_evaluate_returns_metrics(self):
        results = self.pipeline.evaluate(
            self.images[:6], self.labels[:6], verbose=False
        )
        assert "top1" in results
        assert "top_n" in results
        assert 0.0 <= results["top1"] <= 1.0
        assert 0.0 <= results["top_n"] <= 1.0

    def test_topn_ge_top1(self):
        results = self.pipeline.evaluate(
            self.images, self.labels, verbose=False
        )
        assert results["top_n"] >= results["top1"] - 1e-6

    def test_correct_class_in_nbest_trained_data(self):
        """On training data a correct class should appear in N-best."""
        nbest = self.pipeline.predict_nbest(self.images)
        tn = topn_accuracy(self.labels, nbest)
        # With n_best=3 out of 6 classes, should find most correct
        assert tn >= 0.3


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPipelinePersistence:

    def test_save_load(self, tiny_dataset, minimal_config, tmp_path):
        images, labels = tiny_dataset
        pipeline = HolisticPipeline(config=minimal_config)
        pipeline.fit(images, labels)

        model_path = str(tmp_path / "pipeline.joblib")
        pipeline.save(model_path)
        assert os.path.exists(model_path)

        loaded = HolisticPipeline.load(model_path)
        assert loaded._fitted is True

        # Predictions should be identical
        pred_orig = pipeline.predict(images[:4])
        pred_load = loaded.predict(images[:4])
        np.testing.assert_array_equal(pred_orig, pred_load)
