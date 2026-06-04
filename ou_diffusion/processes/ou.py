from __future__ import annotations
import numpy as np
from ..ou_process import OUConstants, ou_constants, generate_ou_dataset
from .base import LinearTransition, Process, ProcessReport, StatCheck, as_ndl

class OUProcess(Process):
    name = "OU"
    d = 1
    is_gaussian = True

    def __init__(self, theta: float = 1.0, sigma: float = 1.0, dt: float = 0.05):
        self.c: OUConstants = ou_constants(theta, sigma, dt)

    def exact_sample(self, N: int, L: int, seed: int | None = None) -> np.ndarray:
        x = generate_ou_dataset(self.c.theta, self.c.sigma, self.c.dt, L, N, seed)
        return x[:, None, :]

    def mean(self, L: int) -> np.ndarray:
        return np.zeros(L)

    def covariance(self, L: int) -> np.ndarray:
        idx = np.arange(L)
        return self.c.s2 * self.c.a ** np.abs(idx[:, None] - idx[None, :])

    def transition(self) -> LinearTransition:
        return LinearTransition(
            A=np.array([[self.c.a]]),
            b=np.zeros(1),
            Q=np.array([[self.c.q]]),
        )

    def validate(self, traj: np.ndarray) -> ProcessReport:
        x = as_ndl(traj, self.d)[:, 0, :]                      # (N, L)
        xc = x - x.mean()
        rho1 = float((xc[:, 1:] * xc[:, :-1]).mean() / (xc * xc).mean())
        return ProcessReport(
            self.name,
            [
                StatCheck("marginal variance", float(x.var()), self.c.s2, "rel"),
                StatCheck("lag-1 autocorr", rho1, self.c.a, "abs"),
            ],
        )
