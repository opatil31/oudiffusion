"""
Fractional Ornstein-Uhlenbeck: stat long-memory gaussian process
"""
from __future__ import annotations
from math import gamma as _Gamma, sin, pi
import numpy as np
from scipy import integrate
from .base import Process, ProcessReport, StatCheck, as_ndl


class FOUProcess(Process):
    name = "fOU"
    d = 1
    is_gaussian = True

    def __init__(self, theta: float = 1.0, sigma: float = 1.0, hurst: float = 0.75,
                 dt: float = 0.05):
        if theta <= 0 or sigma <= 0 or dt <= 0:
            raise ValueError("theta, sigma, dt must all be positive.")
        if not (0.0 < hurst < 1.0):
            raise ValueError(f"hurst H must be in (0,1); got {hurst}.")
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.H = float(hurst)
        self.dt = float(dt)
        self._VH = _Gamma(2.0 * self.H + 1.0) * sin(pi * self.H)
        self._gamma0 = self.sigma**2 * _Gamma(2.0 * self.H + 1.0) / (2.0 * self.theta**(2.0 * self.H))
        self._gcache: dict[float, float] = {0.0: self._gamma0}

    def describe(self) -> dict:
        return dict(theta=self.theta, sigma=self.sigma, hurst=self.H, dt=self.dt)

    def _gamma(self, tau: float) -> float:
        key = round(float(tau), 12)
        if key in self._gcache:
            return self._gcache[key]
        th, h = self.theta, self.H
        if tau == 0.0:
            g = self._gamma0
        elif h == 0.5:
            g = self._gamma0 * float(np.exp(-th * tau))
        elif h > 0.5:
            b = 2.0 * h - 2.0
            C = self.sigma**2 * h * (2.0 * h - 1.0) / (2.0 * th)
            i1, _ = integrate.quad(lambda r: np.exp(-th * r) * (tau + r)**b, 0.0, np.inf, limit=200)
            i2, _ = integrate.quad(lambda r: np.exp(-th * r) * (tau - r)**b, 0.0, tau, limit=200)
            i3, _ = integrate.quad(lambda r: np.exp(-th * r) * (r - tau)**b, tau, np.inf, limit=200)
            g = C * (i1 + i2 + i3)
        else:
            integrand = lambda w: w**(1.0 - 2.0 * h) / (th**2 + w * w)
            val, _ = integrate.quad(integrand, 0.0, np.inf, weight="cos", wvar=tau, limit=200)
            g = self.sigma**2 * self._VH / pi * val
        self._gcache[key] = g
        return g

    def rho(self, k: int) -> float:
        return self._gamma(k * self.dt) / self._gamma0

    def mean(self, L: int) -> np.ndarray:
        return np.zeros(L)

    def covariance(self, L: int) -> np.ndarray:
        g = np.array([self._gamma(k * self.dt) for k in range(L)])
        i = np.arange(L)
        return g[np.abs(i[:, None] - i[None, :])]

    def exact_sample(self, N: int, L: int, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        Sigma = self.covariance(L)
        chol = np.linalg.cholesky(Sigma + 1e-12 * np.eye(L))
        x = rng.standard_normal((N, L)) @ chol.T
        return x[:, None, :]

    def validate(self, traj: np.ndarray) -> ProcessReport:
        x = as_ndl(traj, self.d)[:, 0, :]
        L = x.shape[1]
        xc = x - x.mean()
        denom = float((xc * xc).mean())

        def ac(k: int) -> float:
            return float((xc[:, k:] * xc[:, :L - k]).mean() / denom)

        far = min(L // 4, 16)
        return ProcessReport(self.name, [
            StatCheck("marginal variance", float(x.var()), self._gamma0, "rel"),
            StatCheck("lag-1 autocorr", ac(1), self.rho(1), "abs"),
            StatCheck(f"lag-{far} autocorr", ac(far), self.rho(far), "abs"),
        ])


if __name__ == "__main__":
    fou_half = FOUProcess(theta=1.0, sigma=1.0, hurst=0.5, dt=0.05)
    s2 = 1.0 / (2.0 * 1.0)
    print("[H=1/2 must reproduce OU gamma(tau)=s2 e^{-theta tau}, s2=0.5]")
    for k in (0, 1, 4, 16):
        tau = k * 0.05
        print(f"  lag {k:2d}: fOU gamma={fou_half._gamma(tau):.6f}   OU s2 e^-theta tau="
              f"{s2*np.exp(-1.0*tau):.6f}")

    H, L, N = 0.8, 64, 20000
    proc = FOUProcess(theta=1.0, sigma=1.0, hurst=H, dt=0.05)
    print(f"\n=== fOU {proc.describe()}  (gamma0={proc._gamma0:.4f}) ===")
    x = proc.exact_sample(N=N, L=L, seed=0)
    print(proc.validate(x))

    ou_ref = FOUProcess(theta=1.0, sigma=1.0, hurst=0.5, dt=0.05)
    print("\n[autocorr: fOU(H=0.8) vs OU(H=0.5), same theta -- OU dead in the tail, fOU polynomial]")
    for k in (1, 4, 8, 16, 32, 48, 63):
        print(f"  lag {k:2d}:  fOU rho={proc.rho(k):.4f}   OU rho={ou_ref.rho(k):.6f}")