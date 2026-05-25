"""
ouhawr/preprocessing/image_preprocessing.py

Implements the three-step preprocessing pipeline described in Section 4.1
of the paper:
  1. Resize to 256×128 pixels (bicubic interpolation)
  2. Spatial smoothing (noise reduction filter of [38])
  3. Local entropy filtering (7×7 neighbourhood, flat structuring element)

The entropy-filtered image is what the three texture descriptors operate on.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Union


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _amin_filter(image: np.ndarray) -> np.ndarray:
    """
    Spatial smoothing filter following Amin et al. [38].
    A 3×3 median filter followed by a mild Gaussian blur reproduces the
    noise-suppression intent of the original filter while remaining
    numerically stable for binary/grey handwriting images.
    """
    smoothed = cv2.medianBlur(image.astype(np.uint8), ksize=3)
    smoothed = cv2.GaussianBlur(smoothed, (3, 3), sigmaX=0.5)
    return smoothed.astype(np.float32)


def _entropy_filter(image: np.ndarray, window: int = 7) -> np.ndarray:
    """
    Compute local entropy in an N×N neighbourhood around every pixel.

    Fast vectorised implementation using quantised bins (32 levels).
    This approximates the exact per-pixel Shannon entropy at a fraction
    of the cost of scipy.ndimage.generic_filter.

    Parameters
    ----------
    image  : 2-D float array (values in [0, 255])
    window : neighbourhood side length N (default 7, as in paper)

    Returns
    -------
    entropy_image : 2-D float array of the same shape as *image*
    """
    uint8_img = np.clip(image, 0, 255).astype(np.uint8)
    H, W = uint8_img.shape
    pad = window // 2
    n_bins = 32   # quantise to 32 grey levels (fast approximation)

    # Quantise to n_bins levels
    quant = (uint8_img.astype(np.float32) * (n_bins / 256.0)).astype(np.int32)
    quant = np.clip(quant, 0, n_bins - 1)

    # Accumulate per-bin counts over the sliding window using box filters
    entropy_img = np.zeros((H, W), dtype=np.float32)
    n_pixels = window * window

    padded = np.pad(quant, pad, mode="reflect")
    counts = np.zeros((H, W, n_bins), dtype=np.float32)

    # Uniform box kernel for summing counts (normalise=False)
    kernel = np.ones((window, window), dtype=np.float32)

    for b in range(n_bins):
        mask = (padded == b).astype(np.float32)
        # filter2D on padded array: result has same shape as padded
        resp = cv2.filter2D(mask, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        # Crop back to original size
        counts[:, :, b] = resp[pad:pad+H, pad:pad+W]

    # Normalise to probabilities and compute entropy
    prob = counts / n_pixels          # (H, W, n_bins)
    eps  = 1e-10
    safe = prob + eps
    entropy_img = -np.sum(prob * np.log2(safe), axis=2).astype(np.float32)
    return entropy_img


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ImagePreprocessor:
    """
    Executes the full pre-processing pipeline:
      resize → smooth → entropy filter

    Parameters
    ----------
    target_width  : output width in pixels  (default 256)
    target_height : output height in pixels (default 128)
    entropy_window: size of the local entropy neighbourhood (default 7)
    """

    def __init__(
        self,
        target_width: int = 256,
        target_height: int = 128,
        entropy_window: int = 7,
    ):
        self.target_width = target_width
        self.target_height = target_height
        self.entropy_window = entropy_window

    # ------------------------------------------------------------------
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Apply the full pipeline to a single word image.

        Parameters
        ----------
        image : 2-D or 3-D uint8 numpy array (greyscale or BGR)

        Returns
        -------
        entropy_image : 2-D float32 array of shape
                        (target_height, target_width)
        """
        grey = self._to_grey(image)
        resized = self._resize_1(grey)
        smoothed = _amin_filter(resized)
        entropy = _entropy_filter(smoothed, window=self.entropy_window)
        return entropy

    # ------------------------------------------------------------------
    def preprocess_path(self, path: Union[str, Path]) -> np.ndarray:
        """Load an image file and preprocess it."""
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return self.preprocess(img)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_grey(image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.astype(np.float32)

    def _resize(self, grey: np.ndarray) -> np.ndarray:
        return cv2.resize(
            grey,
            (self.target_width, self.target_height),
            interpolation=cv2.INTER_CUBIC,
        )


    def _resize_1(self, image: np.ndarray) -> np.ndarray:
        """
        Resizes either grayscale or binary images.

        - Grayscale: Uses INTER_CUBIC for smooth scaling.
        - Binary: Uses INTER_NEAREST to preserve binary values.

        Args:
        image: np.ndarray, either grayscale (uint8/float) or binary (bool/uint8).

        Returns:
        Resized image with same dtype semantics as input.
        """
        # Step 1: Detect if the image is binary
        is_binary = False
        if image.dtype == bool:
        is_binary = True
        elif image.dtype == np.uint8:
        # Check if only 0 and 255 (or 0 and 1) are present
        unique_vals = np.unique(image)
        if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 255}):
            is_binary = True
        elif image.dtype in [np.float32, np.float64]:
        unique_vals = np.unique(image)
        if len(unique_vals) <= 2 and set(unique_vals).issubset({0.0, 1.0}):
            is_binary = True

        # Step 2: Choose interpolation method
        if is_binary:
        interpolation = cv2.INTER_NEAREST
        else:
        interpolation = cv2.INTER_CUBIC

        # Step 3: Resize
        resized = cv2.resize(
        image,
        (self.target_width, self.target_height),
        interpolation=interpolation,
        )

        # Step 4: Preserve original dtype semantics
        if is_binary:
        if image.dtype == bool:
            # Convert back to bool if input was bool
            # Note: cv2.resize returns uint8, so we threshold
            resized = resized > 0
        elif image.max() <= 1:
            # If input was float in [0,1], ensure output is also in [0,1]
            # cv2.resize on uint8-like binary might return 0/255, so normalize
            if resized.max() > 1:
                resized = (resized > 0).astype(image.dtype)
            else:
                resized = resized.astype(image.dtype)
        else:
            # Input was uint8 with 0/255, output is already uint8 from cv2
            pass
        else:
        # For grayscale, ensure dtype matches input if necessary
        if resized.dtype != image.dtype:
            resized = resized.astype(image.dtype)

        return resized