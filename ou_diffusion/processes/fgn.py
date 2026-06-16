"""
Fractional Gaussian noise (fGn): the stationary long-memory Gaussian process
"""
from __future__ import annotations
import numpy as np
from .base import Process, ProcessReport, StatCheck, as_ndl


class FGNProcess(Process):
    name = "fGn"
    d = 1
    is_gaussian = True

    def __init__(self, hurst: float = 0.75, sigma: float = 1.0):
        if not (0.0 < hurst < 1.0):
            raise ValueError(f"hurst H must be in (0,1); got {hurst}.")
        if sigma <= 0:
            raise ValueError("sigma must be positive.")
        self.H = float(hurst)
        self.sigma = float(sigma)

    def describe(self) -> dict:
        return dict(hurst=self.H, sigma=self.sigma)

    def _gamma(self, k) -> np.ndarray:
        k = np.abs(np.asarray(k, dtype=np.float64))
        h2 = 2.0 * self.H
        return 0.5 * self.sigma**2 * (np.abs(k + 1.0)**h2 - 2.0 * k**h2 + np.abs(k - 1.0)**h2)

    def rho(self, k: int) -> float:
        return float(self._gamma(k) / self._gamma(0))

    def mean(self, L: int) -> np.ndarray:
        return np.zeros(L)

    def covariance(self, L: int) -> np.ndarray:
        idx = np.arange(L)
        return self._gamma(idx[:, None] - idx[None, :])

    def exact_sample(self, N: int, L: int, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        Sigma = self.covariance(L)
        chol = np.linalg.cholesky(Sigma + 1e-12 * np.eye(L))
        x = rng.standard_normal((N, L)) @ chol.T
        return x[:, None, :]

    def _aggregated_variance_hurst(self, x: np.ndarray) -> float:
        N, L = x.shape
        logm, logv = [], []
        m = 1
        while m <= L // 4:
            nb = L // m
            block_means = x[:, :nb * m].reshape(N, nb, m).mean(axis=2)   # (N, nb)
            logm.append(np.log(m)); logv.append(np.log(block_means.var()))
            m *= 2
        slope = np.polyfit(np.array(logm), np.array(logv), 1)[0]
        return 0.5 * slope + 1.0

    def validate(self, traj: np.ndarray) -> ProcessReport:
        x = as_ndl(traj, self.d)[:, 0, :]                   # (N, L)
        L = x.shape[1]
        xc = x - x.mean()
        denom = float((xc * xc).mean())

        def ac(k: int) -> float:
            return float((xc[:, k:] * xc[:, :L - k]).mean() / denom)

        checks = [
            StatCheck("marginal variance", float(x.var()), self.sigma**2, "rel"),
            StatCheck("lag-1 autocorr", ac(1), self.rho(1), "abs"),
        ]
        for k in sorted({L // 8, L // 4, L // 2}):
            if k >= 2:
                checks.append(StatCheck(f"lag-{k} autocorr", ac(k), self.rho(k), "abs"))
        checks.append(StatCheck("effective Hurst (agg-var)",
                                self._aggregated_variance_hurst(x), self.H, "abs"))
        return ProcessReport(self.name, checks)


if __name__ == "__main__":
    H, L, N = 0.75, 64, 20000
    proc = FGNProcess(hurst=H, sigma=1.0)
    print(f"fGn {proc.describe()}  (long memory for H>0.5; gamma(0)=sigma^2)")

    x = proc.exact_sample(N=N, L=L, seed=0)
    print("\n[exact-sampler self-check vs fGn theory]")
    print(proc.validate(x))

    emp = np.cov(x[:, 0, :].T, bias=True)
    fro = np.linalg.norm(emp - proc.covariance(L)) / np.linalg.norm(proc.covariance(L))
    print(f"\ncovariance Frobenius rel-error (empirical vs exact): {fro:.4f}")

    a = proc.rho(1)
    print(f"\n[autocorrelation decay: fGn vs lag-1-matched exponential a={a:.4f} (Markov)]")
    for k in (1, 2, 4, 8, 16, 32):
        print(f"  lag {k:2d}:  fGn rho={proc.rho(k):.4f}   exponential a^k={a**k:.6f}")