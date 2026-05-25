"""
ouhawr/descriptors/wld.py

Weber Local Descriptor (WLD).

Implements the descriptor described in Section 4.2.2, following:
  Chen et al. (2010). "WLD: A Robust Local Image Descriptor."
  IEEE TPAMI, 32(9), 1705-1720.

Two components are combined per pixel:
  (1) Differential excitation ξ — ratio of neighbourhood contrast to
      centre pixel intensity (inspired by Weber's psychophysical law)
  (2) Gradient orientation θ

A joint (ξ, θ) histogram is computed over each spatial patch/region at
each analysis scale, then histograms are concatenated across patches and
scales to form the full descriptor.

Applied to the entropy-filtered word image (not raw pixels).
"""

import numpy as np
import cv2
from typing import List, Tuple


class WLD:
    """
    Weber Local Descriptor.

    Parameters
    ----------
    neighbourhood_size : S  — side of the local square neighbourhood (S×S)
    excitation_bins_T  : T  — number of differential-excitation histogram bins
    orientation_bins_M : M  — total number of gradient orientation bins
    dominant_orient_K  : K  — number of dominant orientations retained per patch
    n_patches          : N  — number of non-overlapping spatial patches
    n_scales_L         : L  — number of analysis scales
    """

    def __init__(
        self,
        neighbourhood_size: int = 5,
        excitation_bins_T: int = 8,
        orientation_bins_M: int = 6,
        dominant_orient_K: int = 2,
        n_patches: int = 21,
        n_scales_L: int = 2,
    ):
        self.S = neighbourhood_size
        self.T = excitation_bins_T
        self.M = orientation_bins_M
        self.K = dominant_orient_K
        self.N = n_patches
        self.L = n_scales_L

        # Dimension of a single-patch, single-scale descriptor
        # K sub-histograms each of size T × (M/K)
        self._orient_per_dom = max(self.M // self.K, 1)
        self._per_patch_dim = self.T * self.M  # T × M joint histogram

        # Total feature dimension: T × M × N × L
        self.feature_dim = self.T * self.M * self.N * self.L

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(self, image: np.ndarray) -> np.ndarray:
        """
        Extract WLD feature vector from an entropy-filtered word image.

        Parameters
        ----------
        image : 2-D float32 array

        Returns
        -------
        feature : 1-D float32 array of length ``feature_dim``
        """
        descriptors = []
        for scale in range(self.L):
            # Scale is simulated by varying the neighbourhood kernel size
            s = self.S + 2 * scale   # 5, 7, 9, ...
            diff_exc = self._differential_excitation(image, s)
            grad_ori = self._gradient_orientation(image)
            scale_desc = self._build_histogram(diff_exc, grad_ori)
            descriptors.append(scale_desc)

        return np.concatenate(descriptors).astype(np.float32)

    # ------------------------------------------------------------------
    # Differential excitation
    # ------------------------------------------------------------------

    def _differential_excitation(
        self, image: np.ndarray, s: int
    ) -> np.ndarray:
        """
        ξ(x_c) = arctan( Σ_{i} (x_i - x_c) / x_c )

        where the sum is over the S×S-1 neighbours of the centre pixel x_c.

        Returns a float map of the same spatial dimensions as *image*.
        """
        img = image.astype(np.float64)
        # Compute the sum of (neighbour − centre) over all S×S neighbours
        # using a box filter minus the centre contribution
        kernel = np.ones((s, s), dtype=np.float64)
        kernel[s // 2, s // 2] = 0.0

        neighbour_sum = cv2.filter2D(img, -1, kernel,
                                     borderType=cv2.BORDER_REFLECT)
        n_neighbours = s * s - 1

        eps = 1e-6
        centre = img + eps  # avoid division by zero
        ratio = neighbour_sum / (n_neighbours * centre)
        xi = np.arctan(ratio).astype(np.float32)
        return xi

    # ------------------------------------------------------------------
    # Gradient orientation
    # ------------------------------------------------------------------

    @staticmethod
    def _gradient_orientation(image: np.ndarray) -> np.ndarray:
        """
        θ(x_c) = arctan(g_y / g_x)   mapped to [0, 2π)
        """
        img = image.astype(np.float32)
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        orientation = np.arctan2(gy, gx).astype(np.float32)
        # Map from (−π, π] to [0, 2π)
        orientation[orientation < 0] += 2 * np.pi
        return orientation

    # ------------------------------------------------------------------
    # Joint histogram over spatial patches
    # ------------------------------------------------------------------

    def _build_histogram(
        self, diff_exc: np.ndarray, grad_ori: np.ndarray
    ) -> np.ndarray:
        """
        Tile the image into N patches, compute a joint (ξ, θ) histogram
        of size T×M for each patch, and concatenate.
        """
        H, W = diff_exc.shape
        n_rows, n_cols = self._grid_shape(H, W)
        row_edges = np.linspace(0, H, n_rows + 1, dtype=int)
        col_edges = np.linspace(0, W, n_cols + 1, dtype=int)

        patch_hists = []
        count = 0
        for i in range(n_rows):
            for j in range(n_cols):
                if count >= self.N:
                    break
                r0, r1 = row_edges[i], row_edges[i + 1]
                c0, c1 = col_edges[j], col_edges[j + 1]
                xi_p = diff_exc[r0:r1, c0:c1].ravel()
                th_p = grad_ori[r0:r1, c0:c1].ravel()
                hist = self._joint_histogram(xi_p, th_p)
                patch_hists.append(hist)
                count += 1
            if count >= self.N:
                break

        # Pad with zeros if fewer patches were generated
        while len(patch_hists) < self.N:
            patch_hists.append(np.zeros(self._per_patch_dim, dtype=np.float32))

        return np.concatenate(patch_hists[:self.N])

    def _joint_histogram(
        self, xi: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        """
        Build a T×M 2-D joint histogram of excitation vs orientation,
        flatten and L1-normalise.
        """
        xi_bins = np.linspace(-np.pi / 2, np.pi / 2, self.T + 1)
        th_bins = np.linspace(0, 2 * np.pi, self.M + 1)

        hist2d, _, _ = np.histogram2d(
            xi, theta, bins=[xi_bins, th_bins]
        )
        hist2d = hist2d.astype(np.float32)
        norm = hist2d.sum()
        if norm > 0:
            hist2d /= norm
        return hist2d.ravel()

    def _grid_shape(self, H: int, W: int) -> Tuple[int, int]:
        """Return (n_rows, n_cols) for a grid of ≥ N patches."""
        n = self.N
        n_rows = int(np.round(np.sqrt(n * H / W)))
        n_rows = max(n_rows, 1)
        n_cols = int(np.ceil(n / n_rows))
        return n_rows, n_cols

    # ------------------------------------------------------------------

    def get_params(self) -> dict:
        return dict(
            neighbourhood_size=self.S,
            excitation_bins_T=self.T,
            orientation_bins_M=self.M,
            dominant_orient_K=self.K,
            n_patches=self.N,
            n_scales_L=self.L,
        )
