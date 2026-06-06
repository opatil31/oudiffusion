from __future__ import annotations
import numpy as np
from .vector_ou import VectorOU

def ring_laplacian(sites: int) -> np.ndarray:
    i = np.arange(sites)
    Lap = np.zeros((sites, sites))
    Lap[i, i] = 2.0
    Lap[i, (i + 1) % sites] = -1.0
    Lap[i, (i - 1) % sites] = -1.0
    return Lap


class StochasticHeat(VectorOU):
    name = "HEAT"

    def __init__(self, sites: int = 16, lam: float = 0.25, kappa: float = 2.0,
                 sigma: float = 1.0, dt: float = 0.05):
        if sites < 3:
            raise ValueError("need at least 3 spatial sites (periodic ring)")
        if lam <= 0:
            raise ValueError("lam > 0 required (keeps the constant mode stationary)")
        if kappa < 0 or sigma <= 0:
            raise ValueError("kappa >= 0 and sigma > 0 required")
        Theta = lam * np.eye(sites) + kappa * ring_laplacian(sites)
        super().__init__(Theta=Theta, B=sigma * np.eye(sites), dt=dt)
        self.sites, self.lam, self.kappa, self.sigma = sites, float(lam), float(kappa), float(sigma)
        self._desc = dict(sites=sites, lam=lam, kappa=kappa, sigma=sigma, dt=dt)

        theta_j, V = np.linalg.eigh(Theta)
        self.mode_rates = theta_j
        self.mode_V = V
        self.mode_var = self.sigma**2 / (2.0 * theta_j)
        self.mode_a = np.exp(-theta_j * self.dt)

    def _stat_names(self, h: int) -> list[str]:
        return super()._stat_names(h) + [
            "mode var max rel dev",
            "mode lag-1 max abs dev",
            "cross-mode max |corr|",
        ]

    def _stats(self, x: np.ndarray):
        h, base = super()._stats(x)
        m = np.einsum("dj,ndl->njl", self.mode_V, x)
        mc = m - m.mean(axis=(0, 2), keepdims=True)
        var = (mc**2).mean(axis=(0, 2))
        var_dev = float(np.abs(var / self.mode_var - 1.0).max())
        rho = (mc[:, :, 1:] * mc[:, :, :-1]).mean(axis=(0, 2)) / var
        lag_dev = float(np.abs(rho - self.mode_a).max())
        flat = mc.transpose(0, 2, 1).reshape(-1, self.d)
        C = np.corrcoef(flat.T)
        cross = float(np.abs(C - np.diag(np.diag(C))).max())
        return h, np.concatenate([base, [var_dev, lag_dev, cross]])
