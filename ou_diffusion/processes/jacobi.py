"""
Jacobi (a.k.a. Wright-Fisher) diffusion: the beta-marginal Pearson diffusion on [0, 1].
dx = a (b - x) dt + sigma sqrt(x (1 - x)) dW,    a, sigma > 0,  b in (0, 1)
"""
from __future__ import annotations
import numpy as np
from .pearson import PearsonDiffusion


class JacobiProcess(PearsonDiffusion):
    name = "Jacobi"
    support = (0.0, 1.0)

    def __init__(self, theta: float = 2.0, mu: float = 0.3, sigma: float = 1.0,
                 dt: float = 0.05):
        if theta <= 0 or sigma <= 0 or dt <= 0:
            raise ValueError("theta (a), sigma, dt must all be positive.")
        if not (0.0 < mu < 1.0):
            raise ValueError(f"mu (b, the long-run mean) must lie in (0, 1); got {mu}.")
        self.a = float(theta)
        self.b = float(mu)
        self.sigma = float(sigma)
        self.dt = float(dt)

        self.alpha = 2.0 * self.a * self.b / self.sigma**2
        self.beta = 2.0 * self.a * (1.0 - self.b) / self.sigma**2
        if self.alpha < 1.0 or self.beta < 1.0:
            raise ValueError(
                f"Boundary-attainment (Feller-type) condition violated: "
                f"alpha=2ab/sigma^2={self.alpha:.3f}, beta=2a(1-b)/sigma^2={self.beta:.3f}"
            )

        ab = self.alpha + self.beta
        self.stat_mean = self.b
        self.stat_var = self.b * (1.0 - self.b) / (ab + 1.0)
        self.stat_skew = (2.0 * (self.beta - self.alpha) * np.sqrt(ab + 1.0)
                          / ((ab + 2.0) * np.sqrt(self.alpha * self.beta)))
        self.rho1 = self.rho(1)

        self._e = float(np.exp(-self.a * self.dt))
        self._lam2 = 2.0 * self.a + self.sigma**2
        self._c2 = 2.0 * self.a * self.b + self.sigma**2
        self._e2 = float(np.exp(-self._lam2 * self.dt))

    def describe(self) -> dict:
        return dict(a=self.a, b=self.b, sigma=self.sigma, dt=self.dt,
                    beta_alpha=round(self.alpha, 4), beta_beta=round(self.beta, 4))

    def _cond_moments(self, x0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m1 = self.b + (x0 - self.b) * self._e
        m2 = (self._e2 * x0**2
              + self._c2 * self.b * (1.0 - self._e2) / self._lam2
              + self._c2 * (x0 - self.b) * (self._e - self._e2) / (self._lam2 - self.a))
        v = m2 - m1**2
        return m1, v

    def exact_sample(self, N: int, L: int, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        eps = 1e-9
        x = np.empty((N, L), dtype=np.float64)
        x[:, 0] = rng.beta(self.alpha, self.beta, size=N)
        for l in range(1, L):
            m1, v = self._cond_moments(x[:, l - 1])
            mu = np.clip(m1, eps, 1.0 - eps)
            vmax = mu * (1.0 - mu)
            v = np.clip(v, 1e-12, vmax * (1.0 - 1e-9))
            kappa = vmax / v - 1.0
            x[:, l] = rng.beta(mu * kappa, (1.0 - mu) * kappa)
        return x[:, None, :]


if __name__ == "__main__":
    proc = JacobiProcess(theta=2.0, mu=0.3, sigma=1.0, dt=0.05)
    print(f"Jacobi {proc.describe()}  (stat mean={proc.stat_mean:.3f}, var={proc.stat_var:.4f}, "
          f"skew={proc.stat_skew:.3f}, rho1={proc.rho1:.3f})")

    x = proc.exact_sample(N=20000, L=64, seed=0)
    print("\n[sampler self-check vs exact Beta targets]")
    rep = proc.validate(x)
    print(rep)
    skew_chk = [c for c in rep.checks if c.name == "marginal skewness"][0]
    print(f"  -> skewness is the only approximate stat (moment-matched transition); "
          f"deviation {skew_chk.error*100:.2f}% from the exact Beta value.")

    x0 = 0.6
    m1, v = proc._cond_moments(np.array([x0]))
    rng = np.random.default_rng(1)
    mu = float(m1[0]); kappa = mu * (1 - mu) / float(v[0]) - 1.0
    y = rng.beta(mu * kappa, (1 - mu) * kappa, size=400_000)
    print(f"\n[one-step conditional moments from x0={x0}]")
    print(f"  E[x|x0]   target {float(m1[0]):.5f}  (sampler {y.mean():.5f})")
    print(f"  Var[x|x0] target {float(v[0]):.6f}  (sampler {y.var():.6f})")
    lead = proc.sigma**2 * x0 * (1 - x0) * proc.dt
    print(f"  small-dt check: Var target {float(v[0]):.6f}  vs  sigma^2 x0(1-x0) dt = {lead:.6f}")
