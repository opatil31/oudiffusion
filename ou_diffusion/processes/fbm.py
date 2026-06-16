"""
Fractional Brownian motion (fBM): non-stat long memory
"""
from __future__ import annotations
import numpy as np
from .base import Process, ProcessReport, StatCheck, as_ndl


class FBMProcess(Process):
    name = "fBM"
    d = 1
    is_gaussian = True

    def __init__(self, hurst: float = 0.75, sigma: float = 1.0, dt: float = 0.05,
                 mu: float = 0.0, x0: float = 0.0):
        if not (0.0 < hurst < 1.0):
            raise ValueError(f"hurst H must be in (0,1); got {hurst}.")
        if sigma <= 0 or dt <= 0:
            raise ValueError("sigma and dt must be positive.")
        self.H = float(hurst)
        self.sigma = float(sigma)
        self.dt = float(dt)
        self.mu = float(mu)
        self.x0 = float(x0)
        self._c = 0.5 * self.sigma**2 * self.dt**(2.0 * self.H)

    def describe(self) -> dict:
        return dict(hurst=self.H, sigma=self.sigma, dt=self.dt, mu=self.mu, x0=self.x0)

    def covariance(self, L: int) -> np.ndarray:
        i = np.arange(L, dtype=np.float64)
        h2 = 2.0 * self.H
        return self._c * (i[:, None]**h2 + i[None, :]**h2 - np.abs(i[:, None] - i[None, :])**h2)

    def mean(self, L: int) -> np.ndarray:
        return self.x0 + self.mu * self.dt * np.arange(L)

    def exact_sample(self, N: int, L: int, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        m = self.mean(L)
        x = np.empty((N, L), dtype=np.float64)
        x[:, 0] = m[0]
        if L > 1:
            sub = self.covariance(L)[1:, 1:]
            chol = np.linalg.cholesky(sub + 1e-12 * np.eye(L - 1))
            x[:, 1:] = m[1:] + rng.standard_normal((N, L - 1)) @ chol.T
        return x[:, None, :]

    def validate(self, traj: np.ndarray) -> ProcessReport:
        x = as_ndl(traj, self.d)[:, 0, :]
        L = x.shape[1]
        k = np.arange(1, L)
        var_k = x[:, 1:].var(axis=0)
        slope = float(np.polyfit(np.log(k), np.log(var_k), 1)[0])

        d = np.diff(x, axis=1)
        dc = d - d.mean()
        rho1_inc = float((dc[:, 1:] * dc[:, :-1]).mean() / (dc * dc).mean())
        inc_var_target = self.sigma**2 * self.dt**(2.0 * self.H)
        rho1_target = 2.0**(2.0 * self.H - 1.0) - 1.0

        return ProcessReport(self.name, [
            StatCheck("var-growth exponent 2H", slope, 2.0 * self.H, "abs"),
            StatCheck("increment variance", float(d.var()), inc_var_target, "rel"),
            StatCheck("increment lag-1 corr", rho1_inc, rho1_target, "abs"),
            StatCheck("increment mean", float(d.mean()), self.mu * self.dt, "abs"),
        ])


if __name__ == "__main__":
    L, N = 64, 20000
    for H in (0.5, 0.75):
        proc = FBMProcess(hurst=H, sigma=1.0, dt=0.05)
        tag = "  (== Brownian motion)" if H == 0.5 else "  (long-memory increments)"
        print(f"\n=== fBM {proc.describe()}{tag} ===")
        x = proc.exact_sample(N=N, L=L, seed=0)
        print(proc.validate(x))
        emp = np.cov(x[:, 0, :].T, bias=True)
        cov = proc.covariance(L)
        fro = np.linalg.norm(emp - cov) / np.linalg.norm(cov)
        print(f"covariance Frobenius rel-error: {fro:.4f}   "
              f"increment lag-1 target {2.0**(2.0*H-1.0)-1.0:+.4f}")