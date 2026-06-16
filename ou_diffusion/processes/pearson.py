"""
PearsonDiffusion: shared base for the Pearson-diffusion members like jacobi and CIR
"""
from __future__ import annotations
import numpy as np
from .base import Process, ProcessReport, StatCheck, as_ndl


class PearsonDiffusion(Process):
    d = 1
    is_gaussian = False
    support: tuple[float | None, float | None] = (None, None)

    def rho(self, h: int = 1) -> float:
        return float(np.exp(-self.a * h * self.dt))

    def _support_checks(self, flat: np.ndarray) -> list[StatCheck]:
        checks: list[StatCheck] = []
        low, high = self.support
        if low is not None:
            checks.append(StatCheck(f"frac values <= {low:g}",
                                    float((flat <= low).mean()), 0.0, "abs"))
        if high is not None:
            checks.append(StatCheck(f"frac values >= {high:g}",
                                    float((flat >= high).mean()), 0.0, "abs"))
        return checks

    def validate(self, traj: np.ndarray) -> ProcessReport:
        x = as_ndl(traj, self.d)[:, 0, :]
        N = x.shape[0]
        flat = x.reshape(-1)
        mean = float(flat.mean())

        xc = x - mean
        rho1 = float((xc[:, 1:] * xc[:, :-1]).mean() / (xc * xc).mean())

        s = float(flat.std())
        skew = float(((flat - mean) ** 3).mean() / s**3) if s > 0 else 0.0
        se = None
        if N >= 50:
            rng = np.random.default_rng(0)
            B = 200
            idx = rng.integers(0, N, size=(B, N))
            sk = np.empty(B)
            for j in range(B):
                f = x[idx[j]].reshape(-1)
                fc = f - f.mean()
                sd = fc.std()
                sk[j] = (fc**3).mean() / sd**3 if sd > 0 else 0.0
            se = float(sk.std())

        checks = self._support_checks(flat)
        checks += [
            StatCheck("marginal mean", mean, self.stat_mean, "rel"),
            StatCheck("marginal variance", float(flat.var()), self.stat_var, "rel"),
            StatCheck("lag-1 autocorr", rho1, self.rho1, "abs"),
            StatCheck("marginal skewness", skew, self.stat_skew, "rel", se=se),
        ]
        return ProcessReport(self.name, checks)
