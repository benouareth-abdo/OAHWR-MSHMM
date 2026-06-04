"""
ouhawr/data/ifn_enit_loader.py

IFN/ENIT database loader.

The IFN/ENIT database contains handwritten Arabic town (place) names
organized into 7 dataset partitions:
  set_a, set_b,set_c, set_d, set_e, set_f, set_s

Expected directory layout
-------------------------
<root>/
  set_a/
    <word_class_1>/
      image1.tif
      image2.tif
      ...
    <word_class_2>/
      ...
  set_b/
    ...
  ...

Each image is a binary (black-on-white) TIFF of a handwritten word.
The sub-directory name is used as the ground-truth class label.

Reference:
  Pechwitz et al. (2002). "IFN/ENIT: database of handwritten Arabic words."
  Proc. CIFED'02, pp. 129-136.
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2


# All 7 partitions of the IFN/ENIT database
ALL_DATASETS = ["set_a", "set_b", "set_c", "set_d",
                "set_e", "set_f", "set_s"]

DEFAULT_TRAIN = ["set_a", "set_b", "set_c", "set_d"]
DEFAULT_TEST  = ["set_e", "set_f", "set_s"]

IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".bmp", ".jpg", ".jpeg"}


def _find_images(directory: Path) -> List[Path]:
    """Recursively find all image files under *directory*."""
    found = []
    for f in sorted(directory.rglob("*")):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            found.append(f)
    return found


class IFNENITLoader:
    """
    Loads images and labels from one or more IFN/ENIT dataset partitions.

    Parameters
    ----------
    root       : path to the IFN/ENIT root directory
    datasets   : list of partition names to load (e.g. ["set_a", "set_b"])
    preprocessor : optional ImagePreprocessor; if provided, images are
                   preprocessed before being returned.
    """

    def __init__(
        self,
        root: Union[str, Path],
        datasets: Optional[List[str]] = None,
        preprocessor=None,
    ):
        self.root = Path(root)
        self.datasets = datasets or ALL_DATASETS
        self.preprocessor = preprocessor

        # Maps: partition → {class_label → [image_paths]}
        self._index: Dict[str, Dict[str, List[Path]]] = {}
        self._build_index()

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        for ds in self.datasets:
            ds_path = self.root / ds
            if not ds_path.exists():
                print(f"[IFNENITLoader] Warning: {ds_path} not found — skipping.")
                continue

            class_index: Dict[str, List[Path]] = {}

            # Iterate over immediate subdirectories (each = one word class)
            for class_dir in sorted(ds_path.iterdir()):
                if not class_dir.is_dir():
                    continue
                imgs = _find_images(class_dir)
                if imgs:
                    class_index[class_dir.name] = imgs

            self._index[ds] = class_index

    # ------------------------------------------------------------------
    # Properties / queries
    # ------------------------------------------------------------------

    @property
    def classes(self) -> List[str]:
        """All unique word classes across all loaded partitions."""
        all_cls: set = set()
        for ds_dict in self._index.values():
            all_cls.update(ds_dict.keys())
        return sorted(all_cls)

    def class_counts(self, dataset: Optional[str] = None) -> Dict[str, int]:
        """Return {class: n_images} counts for a given partition (or all)."""
        counts: Dict[str, int] = {}
        sources = [dataset] if dataset else list(self._index.keys())
        for ds in sources:
            for cls, imgs in self._index.get(ds, {}).items():
                counts[cls] = counts.get(cls, 0) + len(imgs)
        return counts

    def n_images(self, dataset: Optional[str] = None) -> int:
        return sum(self.class_counts(dataset).values())

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_dataset(
        self,
        datasets: Optional[List[str]] = None,
        classes: Optional[List[str]] = None,
        max_per_class: Optional[int] = None,
        random_seed: int = 42,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Load all images from the specified partitions.

        Parameters
        ----------
        datasets      : partitions to load (default: all loaded ones)
        classes       : restrict to these word classes (default: all)
        max_per_class : cap on number of images per class
        random_seed   : RNG seed for subsampling

        Returns
        -------
        images : list of 2-D float32 numpy arrays (preprocessed or raw grey)
        labels : 1-D array of string class labels (same length as images)
        """
        rng = np.random.default_rng(random_seed)
        sources = datasets or list(self._index.keys())
        filter_cls = set(classes) if classes else None

        images = []
        labels = []

        for ds in sources:
            ds_dict = self._index.get(ds, {})
            for cls, paths in sorted(ds_dict.items()):
                if filter_cls and cls not in filter_cls:
                    continue
                selected = list(paths)
                if max_per_class and len(selected) > max_per_class:
                    idx = rng.choice(len(selected), max_per_class,
                                     replace=False)
                    selected = [selected[i] for i in sorted(idx)]
                for p in selected:
                    img = self._load_image(p)
                    if img is not None:
                        images.append(img)
                        labels.append(cls)

        return images, np.array(labels)

    def _load_image(self, path: Path) -> Optional[np.ndarray]:
        """Load and optionally preprocess a single image file."""
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            print(f"[IFNENITLoader] Cannot read: {path}")
            return None
        if self.preprocessor is not None:
            return self.preprocessor.preprocess(raw)
        # Fallback: convert to float32 greyscale
        if raw.ndim == 3:
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        return raw.astype(np.float32)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = ["IFN/ENIT Database Summary", "=" * 40]
        total = 0
        for ds in self.datasets:
            ds_dict = self._index.get(ds, {})
            n = sum(len(v) for v in ds_dict.values())
            total += n
            lines.append(f"  {ds:<12}: {len(ds_dict):4d} classes, {n:6d} images")
        lines.append("-" * 40)
        lines.append(f"  {'TOTAL':<12}: {len(self.classes):4d} classes, {total:6d} images")
        return "\n".join(lines)
