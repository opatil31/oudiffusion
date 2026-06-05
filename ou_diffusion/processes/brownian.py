"""
I've implemented brownian motion as one of my additional process (wiener) because it has
nonstationarity as a property but is still gaussian so we can have an analytic oracle.
"""
from __future__ import annotations
import numpy as np
from .base import LinearTransition, Process, ProcessReport, StatCheck, as_ndl

class BrownianMotion(Process):
    name = "BM"
    d = 1
    is_gaussian = True

    def __init__(self, mu: float = 0.0, sigma: float = 1.0, dt: float = 0.05,
                 x0: float = 0.0):
        if sigma <= 0 or dt <= 0:
            raise ValueError("sigma and dt must be positive.")
        self.mu, self.sigma, self.dt, self.x0 = float(mu), float(sigma), float(dt), float(x0)
    
    def describe(self) -> dict:
        return dict(theta=self.c.theta, sigma=self.c.sigma, dt=self.c.dt)

    def exact_sample(self, N: int, L: int, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        steps = self.mu * self.dt + self.sigma * np.sqrt(self.dt) * rng.standard_normal(
            (N, L - 1)
        )
        x = np.empty((N, L), dtype=np.float64)
        x[:, 0] = self.x0
        np.cumsum(steps, axis=1, out=x[:, 1:])
        x[:, 1:] += self.x0
        return x[:, None, :]

    def mean(self, L: int) -> np.ndarray:
        return self.x0 + self.mu * self.dt * np.arange(L)

    def covariance(self, L: int) -> np.ndarray:
        idx = np.arange(L)
        return self.sigma**2 * self.dt * np.minimum(idx[:, None], idx[None, :])

    def transition(self) -> LinearTransition:
        return LinearTransition(
            A=np.array([[1.0]]),
            b=np.array([self.mu * self.dt]),
            Q=np.array([[self.sigma**2 * self.dt]]),
        )

    def validate(self, traj: np.ndarray) -> ProcessReport:
        x = as_ndl(traj, self.d)[:, 0, :]
        L = x.shape[1]
        ls = np.arange(L)

        v = x.var(axis=0)
        slope = float((ls * v).sum() / (ls * ls).sum())

        d = np.diff(x, axis=1)
        dc = d - d.mean()
        rho1_inc = float((dc[:, 1:] * dc[:, :-1]).mean() / (dc * dc).mean())

        return ProcessReport(
            self.name,
            [
                StatCheck("var growth slope", slope, self.sigma**2 * self.dt, "rel"),
                StatCheck("increment variance", float(d.var()), self.sigma**2 * self.dt, "rel"),
                StatCheck("increment mean", float(d.mean()), self.mu * self.dt, "abs"),
                StatCheck("increment lag-1 corr", rho1_inc, 0.0, "abs"),
            ],
        )
