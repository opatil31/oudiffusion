from __future__ import annotations
import numpy as np

class FittedGaussianBaseline:
    def __init__(self):
        self.mu = None
        self._V = None
        self._sqrt_lam = None
        self.d = None
        self.L = None

    def fit(self, x0: np.ndarray) -> "FittedGaussianBaseline":
        X = np.asarray(x0, dtype=np.float64)
        if X.ndim == 2:
            X = X[:, None, :]
        if X.ndim != 3:
            raise ValueError(f"expected (N, L) or (N, d, L) data, got {X.shape}")
        N, self.d, self.L = X.shape
        F = X.reshape(N, self.d * self.L)
        self.mu = F.mean(axis=0)
        C = np.cov(F.T, bias=True)
        lam, V = np.linalg.eigh(np.atleast_2d(C))
        self._sqrt_lam = np.sqrt(np.clip(lam, 0.0, None))
        self._V = V
        return self

    def sample(self, n: int, seed: int | None = None) -> np.ndarray:
        if self.mu is None:
            raise RuntimeError("call fit() before sample()")
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((n, self.mu.size))
        F = self.mu + (z * self._sqrt_lam) @ self._V.T
        return F.reshape(n, self.d, self.L)
