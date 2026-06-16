"""
PearsonDiffusion: shared base for the Pearson-diffusion members
"""
from __future__ import annotations
import numpy as np
from .base import Process, ProcessReport, StatCheck, as_ndl


class PearsonDiffusion(Process):
    d = 1
    is_gaussian = False
    # subclasses set: name; a, b, sigma, dt; stat_mean, stat_var, stat_skew, rho1.
    support: tuple[float | None, float | None] = (None, None)
    stat_exkurt: float | None = None   # set by subclasses whose signature is the tail (Student-t)

    def rho(self, h: int = 1) -> float:
        """Exact lag-h autocorrelation of any linear-drift (Pearson) diffusion."""
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
        x = as_ndl(traj, self.d)[:, 0, :]                      # (N, L)
        N = x.shape[0]
        flat = x.reshape(-1)
        mean = float(flat.mean())

        # lag-1 autocorrelation, pooled over consecutive pairs
        xc = x - mean
        rho1 = float((xc[:, 1:] * xc[:, :-1]).mean() / (xc * xc).mean())

        # pooled marginal skewness (and excess kurtosis if the subclass declares a target);
        # SEs from a trajectory bootstrap (resampling whole rows) that respects within-
        # trajectory correlation. Heavy-tailed kurtosis is high-variance -- the SE is the
        # honest signal, not the point estimate.
        want_kurt = self.stat_exkurt is not None

        def _shape(v: np.ndarray) -> tuple[float, float]:
            vc = v - v.mean()
            sd = vc.std()
            if sd <= 0:
                return 0.0, 0.0
            return float((vc**3).mean() / sd**3), float((vc**4).mean() / sd**4 - 3.0)

        skew, exkurt = _shape(flat)
        skew_se = kurt_se = None
        if N >= 50:
            rng = np.random.default_rng(0)
            B = 200
            idx = rng.integers(0, N, size=(B, N))
            sk = np.empty(B); ek = np.empty(B)
            for j in range(B):
                sk[j], ek[j] = _shape(x[idx[j]].reshape(-1))
            skew_se = float(sk.std())
            kurt_se = float(ek.std())

        # a symmetric marginal has skew target 0, and a centered marginal has mean target 0
        # -> relative error is undefined there, use absolute
        skew_kind = "abs" if abs(self.stat_skew) < 1e-6 else "rel"
        mean_kind = "abs" if abs(self.stat_mean) < 1e-6 else "rel"
        checks = self._support_checks(flat)
        checks += [
            StatCheck("marginal mean", mean, self.stat_mean, mean_kind),
            StatCheck("marginal variance", float(flat.var()), self.stat_var, "rel"),
            StatCheck("lag-1 autocorr", rho1, self.rho1, "abs"),
            StatCheck("marginal skewness", skew, self.stat_skew, skew_kind, se=skew_se),
        ]
        if want_kurt:
            checks.append(StatCheck("marginal excess kurtosis", exkurt,
                                    self.stat_exkurt, "rel", se=kurt_se))
        return ProcessReport(self.name, checks)