"""per-channel standardization"""
from __future__ import annotations
import numpy as np

class ChannelNormalizer:
    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    @staticmethod
    def _ndl(x: np.ndarray):
        X = np.asarray(x, dtype=np.float64)
        was_2d = X.ndim == 2
        if was_2d:
            X = X[:, None, :]
        if X.ndim != 3:
            raise ValueError(f"expected (N, L) or (N, d, L) data, got {X.shape}")
        return X, was_2d

    def fit(self, x: np.ndarray) -> "ChannelNormalizer":
        X, _ = self._ndl(x)
        self.mean_ = X.mean(axis=(0, 2))
        s = X.std(axis=(0, 2))
        self.std_ = np.where(s > 0.0, s, 1.0)
        return self

    def _check(self):
        if self.mean_ is None:
            raise RuntimeError("call fit() before transform()/inverse()")

    def transform(self, x: np.ndarray) -> np.ndarray:
        self._check()
        X, was_2d = self._ndl(x)
        out = (X - self.mean_[None, :, None]) / self.std_[None, :, None]
        return out[:, 0, :] if was_2d else out

    def inverse(self, x: np.ndarray) -> np.ndarray:
        self._check()
        X, was_2d = self._ndl(x)
        out = X * self.std_[None, :, None] + self.mean_[None, :, None]
        return out[:, 0, :] if was_2d else out
