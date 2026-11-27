"""CSP + LDA baseline (the classic motor-imagery pipeline), multi-class via one-vs-rest.

Implemented on per-window covariance matrices rather than via ``mne.decoding.CSP``:
MNE's version converts all windows to float64 at once (about 6.5 GB for our 64-channel
training set), while covariances are 64 x 64 per window. The math is the same: for each
class c, solve the generalized eigenproblem ``C_c w = lambda C_all w`` and keep the m
filters with the largest and smallest eigenvalues, i.e. the directions where class c has
the most and least power relative to everything else. Features are the log-variances of
the projected signals.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def window_covariances(X: npt.NDArray[np.floating], chunk: int = 2048) -> npt.NDArray[np.float64]:
    """(N, C, T) -> (N, C, C) covariance of each (zero-mean) window."""
    out = np.empty((X.shape[0], X.shape[1], X.shape[1]), dtype=np.float64)
    for s in range(0, X.shape[0], chunk):
        x = np.asarray(X[s : s + chunk], dtype=np.float64)
        x = x - x.mean(axis=-1, keepdims=True)
        out[s : s + chunk] = np.einsum("nct,ndt->ncd", x, x) / x.shape[-1]
    return out


class CSPLDA:
    def __init__(self, n_filters_per_side: int = 3, reg: float = 1e-3, n_classes: int = 3) -> None:
        self.m = n_filters_per_side
        self.reg = reg
        self.n_classes = n_classes
        self.W: npt.NDArray[np.float64] | None = None  # (C, n_features)
        self.lda = LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto", priors=np.full(n_classes, 1 / n_classes)
        )

    def _regularize(self, C: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        n = C.shape[0]
        return np.asarray(C + self.reg * np.trace(C) / n * np.eye(n))

    def fit_covs(self, covs: npt.NDArray[np.float64], y: npt.NDArray[np.integer]) -> CSPLDA:
        # Average class covariances with equal class weight, so 'rest' (the largest class)
        # doesn't dominate the reference covariance.
        class_covs = [covs[y == c].mean(axis=0) for c in range(self.n_classes)]
        C_all = self._regularize(np.mean(class_covs, axis=0))
        filters = []
        for C_c in class_covs:
            _, vecs = eigh(self._regularize(C_c), C_all)  # ascending eigenvalues
            filters += [vecs[:, : self.m], vecs[:, -self.m :]]
        self.W = np.concatenate(filters, axis=1)
        self.lda.fit(self._features(covs), y)
        return self

    def _features(self, covs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        assert self.W is not None, "fit first"
        var = np.einsum("cf,ncd,df->nf", self.W, covs, self.W)
        return np.asarray(np.log(np.maximum(var, 1e-12)))

    def predict_covs(self, covs: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
        return np.asarray(self.lda.predict(self._features(covs)))
