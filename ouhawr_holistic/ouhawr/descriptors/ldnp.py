"""
ouhawr/descriptors/ldnp.py

Local Directional Number Pattern (LDNP).

Implements the descriptor described in Section 4.2.3, following:
  Rivera, Castillo & Chae (2013). "Local Directional Number Pattern for
  Face Analysis: Face and Expression Recognition."
  IEEE Trans. Image Processing, 22(5), 1740-1752.

Steps:
  1. Convolve the image with 8 Kirsch compass masks (K=8 directions).
  2. For each pixel, find the top positive and top negative direction
     indices → assign a 6-bit binary code (3 bits + 3 bits).
  3. Partition the LDNP-coded image into N non-overlapping regions.
  4. Compute a histogram over LDNP codes in each region.
  5. Repeat at L scales (different mask sizes); concatenate.

Applied to the entropy-filtered word image.
"""

import numpy as np
import cv2
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Kirsch compass masks — 8 directions (0°, 45°, 90°, ..., 315°)
# ---------------------------------------------------------------------------

def _kirsch_masks(size: int = 3) -> List[np.ndarray]:
    """
    Generate 8 Kirsch compass masks rotated by 45° increments.
    The base mask encodes the N (0°) direction; subsequent masks
    are obtained by rotating the entries cyclically.

    For size=3 the standard 3×3 Kirsch masks are returned.
    For larger sizes, scaled versions are generated via cv2.resize.
    """
    # Standard 3×3 Kirsch (N direction)
    base = np.array([
        [ 5,  5,  5],
        [-3,  0, -3],
        [-3, -3, -3],
    ], dtype=np.float32)

    masks = []
    M3 = base.copy()
    for _ in range(8):
        if size == 3:
            masks.append(M3.copy())
        else:
            # Up-scale the 3×3 mask
            scaled = cv2.resize(M3, (size, size),
                                interpolation=cv2.INTER_NEAREST)
            masks.append(scaled)
        # Rotate 45°: shift the border elements cyclically
        M3 = _rotate45_kirsch(M3)
    return masks


def _rotate45_kirsch(mask: np.ndarray) -> np.ndarray:
    """Cyclically shift the 8 border elements of a 3×3 mask by one step."""
    m = mask.copy()
    border_indices = [
        (0, 0), (0, 1), (0, 2),
        (1, 2), (2, 2), (2, 1),
        (2, 0), (1, 0),
    ]
    vals = [mask[r, c] for r, c in border_indices]
    vals = vals[1:] + [vals[0]]
    for (r, c), v in zip(border_indices, vals):
        m[r, c] = v
    return m


# ---------------------------------------------------------------------------
# LDNP class
# ---------------------------------------------------------------------------

class LDNP:
    """
    Local Directional Number Pattern descriptor.

    Parameters
    ----------
    n_directions   : K  — number of Kirsch compass masks (always 8)
    n_dominant     : n  — number of dominant directions retained (always 2)
    window_size    : S  — mask convolution size (3 for standard Kirsch)
    n_patches      : N  — number of spatial regions
    n_scales_L     : L  — number of analysis scales (mask sizes)
    """

    # Number of possible LDNP codes: C(8,2) × 2 = 56
    # (choose 2 directions from 8 for positive, encode sign → 56 codes)
    N_CODES = 64  # use 64 bins (6-bit code → 2^6)

    def __init__(
        self,
        n_directions: int = 8,
        n_dominant: int = 2,
        window_size: int = 3,
        n_patches: int = 25,
        n_scales_L: int = 3,
    ):
        self.K = n_directions
        self.n = n_dominant
        self.S = window_size
        self.N = n_patches
        self.L = n_scales_L

        # Per-patch descriptor dimension = N_CODES (one histogram per region)
        self._per_patch_dim = self.N_CODES

        # Total feature dim: N × L × N_CODES
        self.feature_dim = self.N * self.L * self.N_CODES

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(self, image: np.ndarray) -> np.ndarray:
        """
        Extract LDNP feature vector from an entropy-filtered word image.

        Parameters
        ----------
        image : 2-D float32 array

        Returns
        -------
        feature : 1-D float32 array of length ``feature_dim``
        """
        scale_descs = []
        for scale in range(self.L):
            # Larger masks encode coarser-scale structure
            mask_size = self.S + 2 * scale   # 3, 5, 7, ...
            masks = _kirsch_masks(mask_size)
            ldnp_map = self._compute_ldnp_map(image, masks)
            desc = self._build_histograms(ldnp_map)  # (N × N_CODES,)
            scale_descs.append(desc)

        return np.concatenate(scale_descs).astype(np.float32)

    # ------------------------------------------------------------------
    # LDNP code map computation
    # ------------------------------------------------------------------

    def _compute_ldnp_map(
        self, image: np.ndarray, masks: List[np.ndarray]
    ) -> np.ndarray:
        """
        Convolve image with all K masks and assign a 6-bit LDNP code to
        each pixel: MSB 3 bits = index of max positive response,
        LSB 3 bits = index of max (in magnitude) negative response.

        Returns an int32 array of shape equal to image.shape.
        """
        img = image.astype(np.float32)
        H, W = img.shape

        # Compute 8 response images
        responses = np.zeros((self.K, H, W), dtype=np.float32)
        for k, mask in enumerate(masks[:self.K]):
            responses[k] = cv2.filter2D(
                img, cv2.CV_32F, mask, borderType=cv2.BORDER_REFLECT
            )

        # Separate positive and negative responses
        pos_resp = np.where(responses > 0, responses, 0.0)
        neg_resp = np.where(responses < 0, -responses, 0.0)

        # Top positive direction
        top_pos = np.argmax(pos_resp, axis=0).astype(np.int32)   # in [0,7]
        # Top negative direction (largest magnitude negative)
        top_neg = np.argmax(neg_resp, axis=0).astype(np.int32)   # in [0,7]

        # Encode as 6-bit code: (top_pos << 3) | top_neg  → [0, 63]
        ldnp_code = (top_pos << 3) | top_neg
        return ldnp_code.astype(np.int32)

    # ------------------------------------------------------------------
    # Histogram aggregation over patches
    # ------------------------------------------------------------------

    def _build_histograms(self, ldnp_map: np.ndarray) -> np.ndarray:
        """
        Partition ldnp_map into N patches and build a code histogram
        (N_CODES bins) per patch; concatenate and L1-normalise.
        """
        H, W = ldnp_map.shape
        n_rows, n_cols = self._grid_shape(H, W)
        row_edges = np.linspace(0, H, n_rows + 1, dtype=int)
        col_edges = np.linspace(0, W, n_cols + 1, dtype=int)

        hists = []
        count = 0
        for i in range(n_rows):
            for j in range(n_cols):
                if count >= self.N:
                    break
                r0, r1 = row_edges[i], row_edges[i + 1]
                c0, c1 = col_edges[j], col_edges[j + 1]
                region = ldnp_map[r0:r1, c0:c1].ravel()
                hist, _ = np.histogram(region, bins=self.N_CODES,
                                       range=(0, self.N_CODES))
                hist = hist.astype(np.float32)
                s = hist.sum()
                if s > 0:
                    hist /= s
                hists.append(hist)
                count += 1
            if count >= self.N:
                break

        while len(hists) < self.N:
            hists.append(np.zeros(self.N_CODES, dtype=np.float32))

        return np.concatenate(hists[:self.N])

    def _grid_shape(self, H: int, W: int) -> Tuple[int, int]:
        n = self.N
        n_rows = max(int(np.round(np.sqrt(n * H / W))), 1)
        n_cols = int(np.ceil(n / n_rows))
        return n_rows, n_cols

    # ------------------------------------------------------------------

    def get_params(self) -> dict:
        return dict(
            n_directions=self.K,
            n_dominant=self.n,
            window_size=self.S,
            n_patches=self.N,
            n_scales_L=self.L,
        )
