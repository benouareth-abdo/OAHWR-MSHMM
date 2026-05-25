"""
ouhawr/data/lexicon_sampler.py

Builds the 100-word / 40-sample sub-lexicon used for K-fold
cross-validation of hyper-parameters (Section described in the paper).

Strategy:
  1. Pool all available images from the specified datasets.
  2. Retain only word classes that have at least `n_samples` images
     across those datasets.
  3. Randomly sample `n_words` classes from the eligible pool.
  4. For each selected class, randomly draw exactly `n_samples` images.

Returns (images, labels) ready for use in cross_validation.py.
"""

import numpy as np
from typing import List, Optional, Tuple

from .ifn_enit_loader import IFNENITLoader


def build_cv_lexicon(
    loader: IFNENITLoader,
    datasets: List[str],
    n_words: int = 100,
    n_samples: int = 40,
    random_seed: int = 42,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Build the cross-validation sub-lexicon.

    Parameters
    ----------
    loader     : fitted IFNENITLoader (must cover the specified datasets)
    datasets   : which IFN/ENIT partitions to draw samples from
    n_words    : number of word classes to select
    n_samples  : number of image samples per class
    random_seed: reproducibility seed

    Returns
    -------
    images : list of preprocessed image arrays
    labels : 1-D string array of class labels
    """
    rng = np.random.default_rng(random_seed)

    # Count images per class across all specified datasets
    counts = loader.class_counts()  # {class: total_count}

    # Keep only classes with enough samples
    eligible = [cls for cls, cnt in counts.items() if cnt >= n_samples]

    if len(eligible) < n_words:
        raise ValueError(
            f"Only {len(eligible)} classes have >= {n_samples} samples; "
            f"cannot build a {n_words}-word lexicon. "
            f"Try reducing n_words or n_samples."
        )

    # Randomly select n_words classes
    chosen_classes = sorted(
        rng.choice(eligible, size=n_words, replace=False).tolist()
    )

    # Load exactly n_samples per chosen class
    images, labels = loader.load_dataset(
        datasets=datasets,
        classes=chosen_classes,
        max_per_class=n_samples,
        random_seed=int(rng.integers(0, 2**31)),
    )

    return images, labels


def stratified_kfold_split(
    labels: np.ndarray,
    k: int = 5,
    random_seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create K stratified folds from a label array.

    Returns
    -------
    List of (train_indices, val_indices) tuples.
    """
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_seed)
    dummy_X = np.zeros((len(labels), 1))
    return list(skf.split(dummy_X, labels))
