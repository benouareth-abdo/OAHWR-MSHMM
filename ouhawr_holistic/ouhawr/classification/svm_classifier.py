"""
ouhawr/classification/svm_classifier.py

One-vs-rest linear SVM classifier that produces N-best word hypotheses.

Following Section 4.5 of the paper:
  - Linear kernel, one-vs-rest multi-class strategy
  - Classification is based on signed decision function distances
  - N-best hypotheses = N classes with the smallest (most negative → most
    confident) distances, sorted in ascending order
"""

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from typing import List, Tuple, Optional


class SVMClassifier:
    """
    Linear one-vs-rest SVM with N-best hypothesis output.

    Parameters
    ----------
    C      : SVM regularisation parameter
    n_best : number of top hypotheses to return
    max_iter: maximum iterations for LinearSVC solver
    """

    def __init__(
        self,
        C: float = 1.0,
        n_best: int = 10,
        max_iter: int = 2000,
    ):
        self.C = C
        self.n_best = n_best
        self.max_iter = max_iter

        self._svm: Optional[LinearSVC] = None
        self._le = LabelEncoder()
        self.classes_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        class_names: Optional[List[str]] = None,
    ) -> "SVMClassifier":
        """
        Train the SVM classifier.

        Parameters
        ----------
        X          : (n, d) fused feature matrix
        y          : (n,)   integer or string class labels
        class_names: optional list mapping integer label → word string
        """
        y_enc = self._le.fit_transform(y)
        self.classes_ = self._le.classes_

        self._svm = LinearSVC(
            C=self.C,
            multi_class="ovr",
            max_iter=self.max_iter,
            dual="auto",
        )
        self._svm.fit(X, y_enc)
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the top-1 predicted class label for each sample."""
        if self._svm is None:
            raise RuntimeError("Call fit() first.")
        y_enc = self._svm.predict(X)
        return self._le.inverse_transform(y_enc)

    def predict_nbest(
        self, X: np.ndarray
    ) -> List[List[Tuple[object, float]]]:
        """
        Return N-best hypotheses for each test sample.

        Returns
        -------
        List of length n, each element is a list of (class_label, distance)
        tuples sorted in ascending distance order (smallest = most confident).
        """
        if self._svm is None:
            raise RuntimeError("Call fit() first.")
        # decision_function shape: (n_samples, n_classes)
        decisions = self._svm.decision_function(X)   # (n, C)
        # Negate: higher SVM score → more confident → we want ascending sort
        scores = -decisions   # (n, C)

        n_classes = len(self.classes_)
        k = min(self.n_best, n_classes)

        results = []
        for i in range(X.shape[0]):
            ranked_idx = np.argsort(scores[i])[:k]
            hypotheses = [
                (self._le.inverse_transform([idx])[0], float(scores[i, idx]))
                for idx in ranked_idx
            ]
            results.append(hypotheses)
        return results

    def predict_top1_from_nbest(
        self, nbest: List[List[Tuple[object, float]]]
    ) -> List[object]:
        """Extract the top-1 hypothesis from an N-best list."""
        return [hyps[0][0] for hyps in nbest]

    # ------------------------------------------------------------------

    def get_params(self) -> dict:
        return dict(C=self.C, n_best=self.n_best)
