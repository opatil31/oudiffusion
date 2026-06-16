"""
Fractional Pearson processes: non-Gaussian marginal AND long memory (polynomial autocorrelation decay)
Sample exact fGn Z with Hurst H (standard-normal marginal, long memory), then push it through the probability-integral transform
to the target Pearson marginal:
Z ~ fGn(H, sigma=1)   ->   U = Phi(Z) ~ Uniform(0,1)   ->   Y = F_Pearson^{-1}(U).
I did it for CIR, Jacobi, and Student-t.
"""
from __future__ import annotations
from math import sqrt, pi
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy import stats
from .base import StatCheck, as_ndl
from .pearson import PearsonDiffusion
from .fgn import FGNProcess

_CLIP = 1e-12
_GH_N = 64


class FractionalPearsonBase(PearsonDiffusion):
    d = 1
    is_gaussian = False

    def _finish_init(self, dist, hurst: float, support, params: dict):
        if not (0.5 < hurst < 1.0):
            raise ValueError(f"hurst H must be in (0.5,1) for long memory; got {hurst}.")
        self._dist = dist
        self._params = params
        self.H = float(hurst)
        self.support = support
        m, v, s, k = dist.stats(moments="mvsk")
        self.stat_mean = float(m)
        self.stat_var = float(v)
        self.stat_skew = float(s)
        self._dist_exkurt = float(k) if np.isfinite(k) else None
        self._fgn = FGNProcess(hurst=self.H, sigma=1.0)
        self._x_gh, self._w_gh = hermegauss(_GH_N)
        self._rho_cache: dict[float, float] = {}
        self.rho1 = self._rho_Y(self._fgn.rho(1))

    def _rho_Y(self, rho_z: float) -> float:
        key = round(float(rho_z), 10)
        if key in self._rho_cache:
            return self._rho_cache[key]
        x, w = self._x_gh, self._w_gh
        W = np.outer(w, w) / (2.0 * pi)
        u = x[:, None]
        v = x[None, :]
        g1 = self._dist.ppf(np.clip(stats.norm.cdf(u), _CLIP, 1 - _CLIP))
        g2 = self._dist.ppf(np.clip(stats.norm.cdf(rho_z * u + sqrt(1 - rho_z**2) * v), _CLIP, 1 - _CLIP))
        val = float(((W * g1 * g2).sum() - self.stat_mean**2) / self.stat_var)
        self._rho_cache[key] = val
        return val

    def rho(self, h: int = 1) -> float:
        return self._rho_Y(self._fgn.rho(h))

    def exact_sample(self, N: int, L: int, seed: int | None = None) -> np.ndarray:
        z = self._fgn.exact_sample(N, L, seed)[:, 0, :]                 # standard-normal, long memory
        y = self._dist.ppf(np.clip(stats.norm.cdf(z), _CLIP, 1 - _CLIP))  # exact Pearson marginal
        return y[:, None, :]

    @staticmethod
    def _agg_var_hurst(x: np.ndarray) -> float:
        N, L = x.shape
        logm, logv, m = [], [], 1
        while m <= L // 4:
            nb = L // m
            logm.append(np.log(m))
            logv.append(np.log(x[:, :nb * m].reshape(N, nb, m).mean(axis=2).var()))
            m *= 2
        return 0.5 * float(np.polyfit(np.array(logm), np.array(logv), 1)[0]) + 1.0

    def validate(self, traj: np.ndarray):
        rep = super().validate(traj)
        x = as_ndl(traj, self.d)[:, 0, :]
        L = x.shape[1]
        xc = x - x.mean()
        denom = float((xc * xc).mean())
        far = min(L // 4, 16)
        ac_far = float((xc[:, far:] * xc[:, :L - far]).mean() / denom)
        rep.checks.append(StatCheck(f"lag-{far} autocorr", ac_far, self._rho_Y(self._fgn.rho(far)), "abs"))
        rep.checks.append(StatCheck("effective Hurst (agg-var)", self._agg_var_hurst(x), self.H, "abs"))
        return rep


class FracCIRProcess(FractionalPearsonBase):
    name = "frac-CIR"

    def __init__(self, theta: float = 1.0, mu: float = 1.0, sigma: float = 1.0, hurst: float = 0.75):
        if theta <= 0 or mu <= 0 or sigma <= 0:
            raise ValueError("theta, mu, sigma must be positive.")
        shape = 2.0 * theta * mu / sigma**2
        scale = sigma**2 / (2.0 * theta)
        self._finish_init(stats.gamma(a=shape, scale=scale), hurst, (0.0, None),
                          dict(family="gamma", shape=round(shape, 4), scale=round(scale, 4)))

    def describe(self) -> dict:
        return dict(**self._params, hurst=self.H)


class FracJacobiProcess(FractionalPearsonBase):
    name = "frac-Jacobi"

    def __init__(self, theta: float = 2.0, mu: float = 0.3, sigma: float = 1.0, hurst: float = 0.75):
        if theta <= 0 or sigma <= 0 or not (0.0 < mu < 1.0):
            raise ValueError("theta, sigma must be positive and mu in (0,1).")
        alpha = 2.0 * theta * mu / sigma**2
        beta = 2.0 * theta * (1.0 - mu) / sigma**2
        self._finish_init(stats.beta(a=alpha, b=beta), hurst, (0.0, 1.0),
                          dict(family="beta", alpha=round(alpha, 4), beta=round(beta, 4)))

    def describe(self) -> dict:
        return dict(**self._params, hurst=self.H)


class FracStudentTProcess(FractionalPearsonBase):
    name = "frac-StudentT"

    def __init__(self, mu: float = 0.0, sigma: float = 1.0, nu: float = 9.0, hurst: float = 0.75):
        if sigma <= 0:
            raise ValueError("sigma must be positive.")
        if nu <= 4.0:
            raise ValueError(f"nu must be > 4 for a finite excess kurtosis target; got {nu}.")
        self._finish_init(stats.t(df=nu, loc=mu, scale=sigma), hurst, (None, None),
                          dict(family="studentt", nu=nu, loc=mu, scale=sigma))
        self.stat_exkurt = self._dist_exkurt

    def describe(self) -> dict:
        return dict(**self._params, hurst=self.H)


if __name__ == "__main__":
    L, N = 64, 20000
    for proc in (FracCIRProcess(hurst=0.75), FracJacobiProcess(hurst=0.75),
                 FracStudentTProcess(hurst=0.75)):
        print(f"\n=== {proc.name} {proc.describe()} ===")
        x = proc.exact_sample(N=N, L=L, seed=0)
        print(proc.validate(x))
        print(f"   transformed autocorr: lag-1 {proc._rho_Y(proc._fgn.rho(1)):.4f}  "
              f"lag-16 {proc._rho_Y(proc._fgn.rho(16)):.4f}   (fGn base: {proc._fgn.rho(1):.4f}/{proc._fgn.rho(16):.4f})")