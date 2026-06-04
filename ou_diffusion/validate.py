"""
5.4 - generated trajectories should match:

  - marginal variance        Var(x) ~= sigma^2 / (2 theta) = s2
  - lag-one autocorrelation   rho_1  ~= a = exp(-theta dt)
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .ou_process import OUConstants

def marginal_variance(traj: np.ndarray) -> float:
    traj = np.asarray(traj, dtype=np.float64)
    return float(traj.var())


def lag1_autocorrelation(traj: np.ndarray) -> float:
    traj = np.asarray(traj, dtype=np.float64)
    mu = traj.mean()
    x = traj - mu
    num = (x[:, 1:] * x[:, :-1]).mean()
    den = (x * x).mean()
    return float(num / den)


@dataclass
class ValidationReport:
    var_gen: float
    var_true: float
    rho1_gen: float
    rho1_true: float

    @property
    def var_rel_err(self) -> float:
        return abs(self.var_gen - self.var_true) / self.var_true

    @property
    def rho1_abs_err(self) -> float:
        return abs(self.rho1_gen - self.rho1_true)

    def __str__(self) -> str:
        return (
            "OU statistic        generated      target      error\n"
            f"marginal variance   {self.var_gen:9.4f}   {self.var_true:9.4f}   "
            f"{self.var_rel_err*100:6.2f}%  (rel)\n"
            f"lag-1 autocorr      {self.rho1_gen:9.4f}   {self.rho1_true:9.4f}   "
            f"{self.rho1_abs_err:9.4f}  (abs)"
        )


def validate(traj: np.ndarray, c: OUConstants) -> ValidationReport:
    return ValidationReport(
        var_gen=marginal_variance(traj),
        var_true=c.s2,
        rho1_gen=lag1_autocorrelation(traj),
        rho1_true=c.a,
    )

def irreducible_eps_loss(c: OUConstants, L: int, alphas_cumprod) -> float:
    from .oracle import gaussian_loss_floor

    idx = np.arange(L)
    Sigma = c.s2 * c.a ** np.abs(idx[:, None] - idx[None, :])
    return gaussian_loss_floor(Sigma, alphas_cumprod)
