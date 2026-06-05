from __future__ import annotations
import numpy as np
from .base import Process, ProcessReport, StatCheck, as_ndl
from .brownian import BrownianMotion

class GeometricBrownianMotion(Process):
    name = "GBM"
    d = 1
    is_gaussian = False

    def __init__(self, mu: float = 0.2, sigma: float = 0.4, dt: float = 0.05,
                 x0: float = 1.0):
        if sigma <= 0 or dt <= 0:
            raise ValueError("sigma and dt must be positive.")
        if x0 <= 0:
            raise ValueError("GBM requires a positive start value x0.")
        self.mu, self.sigma, self.dt, self.x0 = float(mu), float(sigma), float(dt), float(x0)
        self.nu = self.mu - 0.5 * self.sigma**2          # log-drift

    def describe(self) -> dict:
        return dict(mu=self.mu, sigma=self.sigma, dt=self.dt, x0=self.x0)

    def log_process(self) -> BrownianMotion:
        """The exact Gaussian reduction: log x is Brownian motion with drift nu."""
        return BrownianMotion(mu=self.nu, sigma=self.sigma, dt=self.dt,
                              x0=float(np.log(self.x0)))

    def exact_sample(self, N: int, L: int, seed: int | None = None) -> np.ndarray:
        return np.exp(self.log_process().exact_sample(N, L, seed))

    def validate(self, traj: np.ndarray) -> ProcessReport:
        x = as_ndl(traj, self.d)[:, 0, :]
        L = x.shape[1]
        checks: list[StatCheck] = []

        checks.append(StatCheck("frac values <= 0", float((x <= 0.0).mean()), 0.0, "abs"))

        pos = x[(x > 0.0).all(axis=1)]
        if pos.shape[0] >= 2:
            r = np.diff(np.log(pos), axis=1)
            rc = r - r.mean()
            rho = float((rc[:, 1:] * rc[:, :-1]).mean() / (rc * rc).mean())
            checks += [
                StatCheck("log-ret mean", float(r.mean()), self.nu * self.dt, "abs"),
                StatCheck("log-ret variance", float(r.var()),
                          self.sigma**2 * self.dt, "rel"),
                StatCheck("log-ret lag-1 corr", rho, 0.0, "abs"),
            ]

        term = x[:, -1]
        checks.append(StatCheck(
            "terminal mean", float(term.mean()),
            self.x0 * np.exp(self.mu * (L - 1) * self.dt), "rel"))
        tc = term - term.mean()
        std = float(tc.std())
        skew = float((tc**3).mean() / std**3) if std > 0 else 0.0
        v = self.sigma**2 * (L - 1) * self.dt
        skew_target = (np.exp(v) + 2.0) * np.sqrt(np.exp(v) - 1.0)
        checks.append(StatCheck("terminal skewness", skew, skew_target, "rel"))

        return ProcessReport(self.name, checks)
