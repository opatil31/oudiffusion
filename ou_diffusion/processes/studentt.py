"""
Student-t Pearson diffusion: heavy-tailed, unbounded member.
dx = a (b - x) dt + sqrt( c [ (x - b)^2 + nu * s^2 ] ) dW,    c = 2a/(nu - 1)
"""
from __future__ import annotations
import numpy as np
from .base import StatCheck, as_ndl
from .pearson import PearsonDiffusion


class StudentTProcess(PearsonDiffusion):
    name = "StudentT"
    support = (None, None)                      # unbounded -> no boundary checks

    def __init__(self, theta: float = 1.0, mu: float = 0.0, sigma: float = 1.0,
                 nu: float = 9.0, dt: float = 0.05, substeps: int = 8):
        if theta <= 0 or sigma <= 0 or dt <= 0:
            raise ValueError("theta (a), sigma (scale s), dt must all be positive.")
        if nu <= 4.0:
            raise ValueError(f"nu must be > 4 for a finite excess kurtosis target; got {nu}.")
        if substeps < 1:
            raise ValueError("substeps must be >= 1.")
        self.a = float(theta)
        self.b = float(mu)
        self.sigma = float(sigma)               # stationary t scale s
        self.nu = float(nu)
        self.dt = float(dt)
        self.substeps = int(substeps)           # internal Gaussian substeps per dt (tail fidelity)

        self.stat_mean = self.b
        self.stat_var = self.sigma**2 * self.nu / (self.nu - 2.0)
        self.stat_skew = 0.0                     # symmetric
        self.stat_exkurt = 6.0 / (self.nu - 4.0)
        self.rho1 = self.rho(1)

        # conditional-moment constants for ONE substep of size dt/substeps. A single big
        # Gaussian step (substeps=1) gets the wrong discrete tail index and under-disperses
        # the kurtosis ~20%; substepping converges the tail to the true t-diffusion while the
        # exact conditional mean/variance compose exactly (so marginal mean/var/autocorr stay
        # exact at every substeps).
        dts = self.dt / self.substeps
        self._e = float(np.exp(-self.a * dts))
        self._lam2 = 2.0 * self.a * (self.nu - 2.0) / (self.nu - 1.0)
        self._e2 = float(np.exp(-self._lam2 * dts))
        self._Vstar = self.stat_var

        # robust tail-fidelity target: P(|x - b| > 3 stationary std) under the exact t_nu.
        # Sample kurtosis is unreliable for heavy tails (its own variance is governed by the
        # 8th moment, infinite for nu <= 8); this bounded proportion is the reproducible
        # primary tail statistic. 3 stationary std in standard-t units = 3 sqrt(nu/(nu-2)).
        _q = 3.0 * np.sqrt(self.nu / (self.nu - 2.0))
        _r = np.random.default_rng(12345)
        self._tail3_target = float((np.abs(_r.standard_t(self.nu, size=4_000_000)) > _q).mean())

    def describe(self) -> dict:
        return dict(a=self.a, b=self.b, scale=self.sigma, nu=self.nu, dt=self.dt,
                    substeps=self.substeps, excess_kurtosis=round(self.stat_exkurt, 4))

    def _cond_var(self, y0: np.ndarray) -> np.ndarray:
        """Exact conditional variance over one substep, given x_t = b + y0."""
        vc = y0**2 * (self._e2 - self._e**2) + self._Vstar * (1.0 - self._e2)
        return np.maximum(vc, 0.0)              # guard tiny roundoff negatives

    def exact_sample(self, N: int, L: int, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        x = np.empty((N, L), dtype=np.float64)
        cur = self.b + self.sigma * rng.standard_t(self.nu, size=N)       # exact t marginal
        x[:, 0] = cur
        for l in range(1, L):
            for _ in range(self.substeps):
                y0 = cur - self.b
                cur = self.b + y0 * self._e + np.sqrt(self._cond_var(y0)) * rng.standard_normal(N)
            x[:, l] = cur
        return x[:, None, :]

    def validate(self, traj: np.ndarray):
        # base report: mean, variance, autocorr, skewness(=0), excess kurtosis (noisy signature)
        rep = super().validate(traj)
        flat = as_ndl(traj, self.d)[:, 0, :].reshape(-1)
        std = float(np.sqrt(self.stat_var))
        tail = float((np.abs(flat - self.b) > 3.0 * std).mean())
        # robust, reproducible tail check; under-dispersion shows up as tail < target
        rep.checks.append(StatCheck("tail mass |x-b|>3std", tail, self._tail3_target, "rel"))
        return rep


if __name__ == "__main__":
    proc = StudentTProcess(theta=1.0, mu=0.0, sigma=1.0, nu=9.0, dt=0.05)
    print(f"StudentT {proc.describe()}  (stat var={proc.stat_var:.4f}, skew=0, "
          f"exkurt={proc.stat_exkurt:.3f}, rho1={proc.rho1:.4f}, "
          f"tail>3std target={proc._tail3_target:.4f} vs Gaussian 0.0027)")

    # two seeds: kurtosis is a noisy estimate (heavy tail), tail-mass is reproducible
    for seed in (0, 1):
        x = proc.exact_sample(N=40000, L=64, seed=seed)
        print(f"\n[sampler self-check vs exact t targets | seed={seed}]")
        print(proc.validate(x))

    # one-step conditional-variance check against the closed form
    x0 = 2.0
    vc = float(proc._cond_var(np.array([x0 - proc.b]))[0])
    rng = np.random.default_rng(1)
    y = proc.b + (x0 - proc.b) * proc._e + np.sqrt(vc) * rng.standard_normal(400_000)
    print(f"\n[one substep from x0={x0}]  Var target {vc:.5f}  sampler {y.var():.5f}")