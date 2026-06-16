"""
Cox-Ingersoll-Ross (CIR): the gamma-marginal Pearson diffusion.
dx = a (b - x) dt + sigma sqrt(x) dW,      a, b, sigma > 0
"""
from __future__ import annotations
import numpy as np
from .pearson import PearsonDiffusion

class CIRProcess(PearsonDiffusion):
    name = "CIR"
    support = (0.0, None)

    def __init__(self, theta: float = 1.0, mu: float = 1.0, sigma: float = 1.0,
                 dt: float = 0.05):
        if theta <= 0 or mu <= 0 or sigma <= 0 or dt <= 0:
            raise ValueError("theta (a), mu (b), sigma, dt must all be positive.")
        self.a = float(theta)
        self.b = float(mu)
        self.sigma = float(sigma)
        self.dt = float(dt)

        self.shape = 2.0 * self.a * self.b / self.sigma**2
        self.scale = self.sigma**2 / (2.0 * self.a)
        if self.shape < 1.0:
            raise ValueError(
                f"Feller condition violated: 2ab/sigma^2 = {self.shape:.3f} < 1; the process "
                "would touch 0. Increase a or b, or decrease sigma."
            )

        self.stat_mean = self.shape * self.scale
        self.stat_var = self.shape * self.scale**2
        self.stat_skew = 2.0 / np.sqrt(self.shape)
        self.rho1 = self.rho(1)

        self._e = float(np.exp(-self.a * self.dt))
        self._c = 2.0 * self.a / (self.sigma**2 * (1.0 - self._e))
        self._df = 4.0 * self.a * self.b / self.sigma**2

    def describe(self) -> dict:
        return dict(a=self.a, b=self.b, sigma=self.sigma, dt=self.dt,
                    gamma_shape=round(self.shape, 4))

    def exact_sample(self, N: int, L: int, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        x = np.empty((N, L), dtype=np.float64)
        x[:, 0] = rng.gamma(self.shape, self.scale, size=N)
        for l in range(1, L):
            lam = 2.0 * self._c * x[:, l - 1] * self._e
            y = rng.noncentral_chisquare(self._df, lam)
            x[:, l] = y / (2.0 * self._c)
        return x[:, None, :]


if __name__ == "__main__":
    proc = CIRProcess(theta=1.0, mu=1.0, sigma=1.0, dt=0.05)
    print(f"CIR {proc.describe()}  (stat mean={proc.stat_mean:.3f}, var={proc.stat_var:.3f}, "
          f"skew={proc.stat_skew:.3f}, rho1={proc.rho1:.3f})")
    x = proc.exact_sample(N=20000, L=64, seed=0)
    print("\n[exact-sampler self-check]")
    print(proc.validate(x))