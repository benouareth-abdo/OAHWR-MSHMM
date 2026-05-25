"""
ouhawr/descriptors/lpq_plus.py

Local Phase Quantization Plus (LPQ+) descriptor.

Implements Algorithm 3 from the paper (Section 4.2.1) exactly:
  - Dense STFT computation at 4 frequency points
  - Decorrelation via matrix W (derived from PCA of training STFT vectors)
  - Soft quantisation into B bins per STFT component
  - Sub-cell splitting (default 2×2) for spatial structure preservation
  - Statistical aggregation (mean + std) over patches
  - Fit / transform API for the decorrelation matrix W

Reference:
  Xiao et al. (2017). "Local Phase Quantization Plus: A principled method
  for embedding local phase quantization into Fisher vector for blurred image
  recognition." Information Sciences, 420, 77-95.
"""

import numpy as np
import cv2
from sklearn.decomposition import PCA
from typing import Optional, Tuple, List


class LPQPlus:
    """
    LPQ+ descriptor (Algorithm 3).

    Parameters
    ----------
    window_size : M  — side length of the local STFT window (odd integer)
    freq_scalar : a  — scalar controlling blur-insensitivity of freq. points
    quant_alpha : α  — soft-quantisation sharpness
    bins_B      : B  — number of histogram bins per STFT component ∈ {4,8,16}
    n_patches   : N  — number of spatial patches (tiled over the image)
    cell_grid   : (rows, cols) sub-cell grid within each patch (default 2×2)
    """

    def __init__(
        self,
        window_size: int = 5,
        freq_scalar: float = 1.0,
        quant_alpha: float = 1.0,
        bins_B: int = 8,
        n_patches: int = 25,
        cell_grid: Tuple[int, int] = (2, 2),
    ):
        self.M = window_size
        self.a = freq_scalar
        self.alpha = quant_alpha
        self.B = bins_B
        self.n_patches = n_patches
        self.cell_rows, self.cell_cols = cell_grid

        # Frequency points: u1..u4 as in the paper
        self._u = np.array([
            [self.a, 0.0],
            [0.0, self.a],
            [self.a, self.a],
            [self.a, -self.a],
        ])  # shape (4, 2)

        # Decorrelation matrix W — set after fit()
        self.W_: Optional[np.ndarray] = None
        self._pca: Optional[PCA] = None

        # Derived dimensions
        # Per pixel: 4 freq × 2 (re+im) = 8 real components → after decorr: 8
        # Per cell:  8 components × B bins = 8B
        # Per patch: cell_rows × cell_cols × 8B
        self._n_cells = self.cell_rows * self.cell_cols
        self._raw_dim = 8  # 4 freq × (re+im)
        self._cell_dim = self._raw_dim * self.B  # dimension of one cell desc.
        self._patch_dim = self._n_cells * self._cell_dim  # one patch descriptor
        # Final feature: mean+std of patch descriptors across N patches
        # → concatenate [mean(N, patch_dim), std(N, patch_dim)] = 2 × patch_dim
        self.feature_dim = 2 * self._patch_dim

    # ------------------------------------------------------------------
    # Fit / Transform API
    # ------------------------------------------------------------------

    def fit(self, images: List[np.ndarray]) -> "LPQPlus":
        """
        Estimate the decorrelation matrix W from a list of entropy images.

        W is the 8×8 matrix whose columns are the eigenvectors of the
        covariance matrix of the raw STFT coefficient vectors collected
        over all training images (Algorithm 3, step 'Decorrelation using
        Matrix W').

        Parameters
        ----------
        images : list of 2-D float32 arrays (entropy-filtered word images)
        """
        stft_vectors = []
        for img in images:
            vecs = self._collect_stft_vectors(img)
            if vecs.shape[0] > 0:
                stft_vectors.append(vecs)

        if not stft_vectors:
            raise ValueError("No STFT vectors collected — check input images.")

        all_vecs = np.vstack(stft_vectors)  # (N_total, 8)

        # PCA to get decorrelation matrix W (eigenvectors of covariance)
        pca = PCA(n_components=self._raw_dim, whiten=False)
        pca.fit(all_vecs)

        # W: rows = principal components  →  shape (8, 8)
        self.W_ = pca.components_.astype(np.float32)  # (8, 8)
        self._pca = pca
        return self

    def transform(self, image: np.ndarray) -> np.ndarray:
        """
        Extract LPQ+ feature vector from a single entropy-filtered image.

        Returns
        -------
        feature : 1-D float32 array of length ``feature_dim``
        """
        if self.W_ is None:
            raise RuntimeError("Call fit() (or set W_ directly) before transform().")

        patches = self._partition_image(image)  # list of N patch arrays
        V_final = []

        for patch in patches:
            patch_desc = self._describe_patch(patch)  # (patch_dim,)
            V_final.append(patch_desc)

        # Statistical aggregation across patches: [mean; std]
        V_mat = np.vstack(V_final)  # (N, patch_dim)
        feature = np.concatenate([V_mat.mean(axis=0), V_mat.std(axis=0)])
        return feature.astype(np.float32)

    def fit_transform(self, images: List[np.ndarray]) -> np.ndarray:
        """Fit on images and return their feature matrix (N × feature_dim)."""
        self.fit(images)
        return np.vstack([self.transform(img) for img in images])

    # ------------------------------------------------------------------
    # Internal: STFT computation
    # ------------------------------------------------------------------

    def _build_stft_maps(self, image: np.ndarray) -> np.ndarray:
        """
        Compute per-pixel STFT response maps for all 4 frequency points.

        Uses cv2.filter2D with precomputed complex kernels, giving an 8-channel
        map of shape (H, W, 8) = [Re(G_u1), Re(G_u2), Re(G_u3), Re(G_u4),
                                   Im(G_u1), Im(G_u2), Im(G_u3), Im(G_u4)].
        """
        img = image.astype(np.float32)
        R = (self.M - 1) // 2
        rows = np.arange(-R, R + 1, dtype=np.float32)
        cols = np.arange(-R, R + 1, dtype=np.float32)
        RR, CC = np.meshgrid(rows, cols, indexing="ij")

        maps = []
        for (uj1, uj2) in self._u:
            phase = 2 * np.pi * (uj1 * RR + uj2 * CC) / self.M
            k_re = np.cos(phase).astype(np.float32)
            k_im = np.sin(phase).astype(np.float32)
            g_re = cv2.filter2D(img, -1, k_re,
                                borderType=cv2.BORDER_REFLECT)
            g_im = cv2.filter2D(img, -1, k_im,
                                borderType=cv2.BORDER_REFLECT)
            maps.append(g_re)
            maps.append(g_im)

        # Reorder to [Re1,Re2,Re3,Re4,Im1,Im2,Im3,Im4] → match _stft_at_pixel
        re_maps = maps[0::2]   # [Re1,Re2,Re3,Re4]
        im_maps = maps[1::2]   # [Im1,Im2,Im3,Im4]
        combined = re_maps + im_maps   # 8 maps
        return np.stack(combined, axis=-1)  # (H, W, 8)

    def _stft_at_pixel(
        self, image: np.ndarray, x: int, y: int
    ) -> np.ndarray:
        """Single-pixel STFT via the precomputed map (used sparingly)."""
        maps = self._build_stft_maps(image)
        return maps[x, y, :]  # shape (8,)

    def _collect_stft_vectors(self, image: np.ndarray) -> np.ndarray:
        """
        Collect all per-pixel STFT vectors from an image (for fitting W).
        Vectorised: returns (H*W//stride^2, 8) array.
        """
        maps = self._build_stft_maps(image)  # (H, W, 8)
        # Subsample every 3rd pixel for efficiency
        sampled = maps[::3, ::3, :]           # (H//3, W//3, 8)
        H2, W2, _ = sampled.shape
        if H2 * W2 == 0:
            return np.empty((0, 8), dtype=np.float32)
        return sampled.reshape(-1, 8).astype(np.float32)

    # ------------------------------------------------------------------
    # Internal: patch-level descriptor
    # ------------------------------------------------------------------

    def _soft_quantise_vec(self, z_vec: np.ndarray) -> np.ndarray:
        """
        Vectorised soft quantisation for all 8 STFT components at one pixel.
        z_vec : (8,) decorrelated STFT vector
        Returns (8*B,) descriptor.
        """
        theta_bars = np.arange(1, self.B + 1) * (np.pi / self.B)  # (B,)
        angles = np.arctan2(np.zeros(8), z_vec)                    # (8,) phase of real values
        # (8, B): cos((angle_m - theta_i) / alpha)
        diff = (angles[:, None] - theta_bars[None, :]) / self.alpha
        return np.cos(diff).ravel().astype(np.float32)             # (8*B,)

    def _soft_quantise(self, z_val: float) -> np.ndarray:
        """Single-component soft quantisation (kept for compatibility)."""
        theta_bars = np.arange(1, self.B + 1) * (np.pi / self.B)
        angle = np.arctan2(0.0, z_val)
        return np.cos((angle - theta_bars) / self.alpha).astype(np.float32)

    def _lpq_plus_pixel(self, image: np.ndarray, x: int, y: int) -> np.ndarray:
        """LPQ+ descriptor at a single pixel. Uses precomputed STFT map."""
        # Build map for the whole patch (cached via _describe_patch_from_map)
        maps = self._build_stft_maps(image)
        g = maps[x, y, :]          # length-8 raw STFT
        z = self.W_ @ g            # decorrelate
        return self._soft_quantise_vec(z)

    def _describe_cell(self, cell: np.ndarray) -> np.ndarray:
        """Aggregate per-pixel LPQ+ descriptors over a cell region."""
        H, W = cell.shape
        if H == 0 or W == 0:
            return np.zeros(self._cell_dim, dtype=np.float32)

        # Vectorised: compute STFT map for the whole cell at once
        maps = self._build_stft_maps(cell)            # (H, W, 8)
        flat = maps.reshape(-1, 8)                    # (H*W, 8)
        Z    = (self.W_ @ flat.T).T                   # (H*W, 8) decorrelated

        theta_bars = np.arange(1, self.B + 1) * (np.pi / self.B)  # (B,)
        angles = np.arctan2(np.zeros_like(Z), Z)                   # (H*W, 8)
        diff   = (angles[:, :, None] - theta_bars[None, None, :]) / self.alpha
        # soft_q: (H*W, 8, B) → (H*W, 8*B)
        soft_q = np.cos(diff).reshape(len(flat), -1)
        return soft_q.mean(axis=0).astype(np.float32)              # (8*B,)

    def _describe_patch(self, patch: np.ndarray) -> np.ndarray:
        """
        Split patch into cell_rows×cell_cols cells, describe each cell,
        and concatenate. Returns vector of length ``_patch_dim``.
        """
        H, W = patch.shape
        cell_h = max(H // self.cell_rows, 1)
        cell_w = max(W // self.cell_cols, 1)
        cell_descs = []
        for i in range(self.cell_rows):
            for j in range(self.cell_cols):
                r0, r1 = i * cell_h, min((i + 1) * cell_h, H)
                c0, c1 = j * cell_w, min((j + 1) * cell_w, W)
                cell = patch[r0:r1, c0:c1]
                cell_descs.append(self._describe_cell(cell))
        return np.concatenate(cell_descs)

    # ------------------------------------------------------------------
    # Internal: image partitioning into N patches
    # ------------------------------------------------------------------

    def _partition_image(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Divide the image into ``n_patches`` non-overlapping patches arranged
        in a roughly square grid.  Returns a list of 2-D arrays.
        """
        H, W = image.shape
        n = self.n_patches
        n_rows = int(np.round(np.sqrt(n * H / W)))
        n_cols = int(np.ceil(n / max(n_rows, 1)))
        n_rows = int(np.ceil(n / max(n_cols, 1)))

        row_edges = np.linspace(0, H, n_rows + 1, dtype=int)
        col_edges = np.linspace(0, W, n_cols + 1, dtype=int)

        patches = []
        for i in range(n_rows):
            for j in range(n_cols):
                if len(patches) >= n:
                    break
                patches.append(
                    image[row_edges[i]:row_edges[i + 1],
                          col_edges[j]:col_edges[j + 1]]
                )
            if len(patches) >= n:
                break
        # Pad if needed (edge case)
        while len(patches) < n:
            patches.append(patches[-1])
        return patches[:n]

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def get_params(self) -> dict:
        return dict(
            window_size=self.M,
            freq_scalar=self.a,
            quant_alpha=self.alpha,
            bins_B=self.B,
            n_patches=self.n_patches,
            cell_grid=(self.cell_rows, self.cell_cols),
        )

    def set_W(self, W: np.ndarray) -> None:
        """Manually set decorrelation matrix (e.g. loaded from disk)."""
        assert W.shape == (self._raw_dim, self._raw_dim), (
            f"Expected W of shape ({self._raw_dim},{self._raw_dim}), got {W.shape}"
        )
        self.W_ = W.astype(np.float32)
