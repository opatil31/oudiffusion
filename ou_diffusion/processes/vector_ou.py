from __future__ import annotations
import numpy as np
import torch
from .base import LinearTransition, Process, ProcessReport, StatCheck, as_ndl

def _expm(M: np.ndarray) -> np.ndarray:
    return torch.linalg.matrix_exp(torch.from_numpy(M)).numpy()

def _discretize(Theta: np.ndarray, BBt: np.ndarray, dt: float):
    d = Theta.shape[0]
    H = np.zeros((2 * d, 2 * d))
    H[:d, :d] = -Theta
    H[:d, d:] = BBt
    H[d:, d:] = Theta.T
    E = _expm(H * dt)
    A = E[:d, :d]
    Q = E[:d, d:] @ A.T
    return A, 0.5 * (Q + Q.T)

def _stationary(A: np.ndarray, Q: np.ndarray) -> np.ndarray:
    S, Ak = Q.copy(), A.copy()
    for _ in range(64):
        S = S + Ak @ S @ Ak.T
        Ak = Ak @ Ak
        if np.abs(Ak).max() < 1e-150:
            break
    return 0.5 * (S + S.T)


def _psd_sqrt(M: np.ndarray) -> np.ndarray:
    lam, V = np.linalg.eigh(M)
    return V * np.sqrt(np.clip(lam, 0.0, None))


class VectorOU(Process):
    name = "VOU"
    is_gaussian = True

    def __init__(self, Theta=None, B=None, theta: float = 1.0, omega: float = 1.5,
                 sigma: float = 1.0, dt: float = 0.05):
        if Theta is None:
            J = np.array([[0.0, -1.0], [1.0, 0.0]])
            Theta = theta * np.eye(2) + omega * J
            self._desc = dict(theta=theta, omega=omega, sigma=sigma, dt=dt)
        else:
            self._desc = dict(dt=dt)
        self.Theta = np.atleast_2d(np.asarray(Theta, dtype=np.float64))
        self.d = self.Theta.shape[0]
        if B is None:
            B = sigma * np.eye(self.d)
        self.B = np.atleast_2d(np.asarray(B, dtype=np.float64))
        if dt <= 0:
            raise ValueError("dt must be positive.")
        if np.linalg.eigvals(self.Theta).real.min() <= 0:
            raise ValueError("Theta must be stable (eigenvalues with positive real part).")
        self.dt = float(dt)
        self._desc["d"] = self.d

        self.A, self.Q = _discretize(self.Theta, self.B @ self.B.T, self.dt)
        self.Sinf = _stationary(self.A, self.Q)
        self._Q_sqrt = _psd_sqrt(self.Q)
        self._S_sqrt = _psd_sqrt(self.Sinf)
        self.lag_check = 5 # horizon for the lag-h validation target

    def describe(self) -> dict:
        return dict(self._desc)

    def exact_sample(self, N: int, L: int, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        x = np.empty((N, self.d, L), dtype=np.float64)
        x[:, :, 0] = rng.standard_normal((N, self.d)) @ self._S_sqrt.T
        for l in range(1, L):
            x[:, :, l] = (x[:, :, l - 1] @ self.A.T
                          + rng.standard_normal((N, self.d)) @ self._Q_sqrt.T)
        return x

    def mean(self, L: int) -> np.ndarray:
        return np.zeros(self.d * L)

    def covariance(self, L: int) -> np.ndarray:
        G = [self.Sinf]
        for _ in range(1, L):
            G.append(self.A @ G[-1])
        T = np.empty((L, L, self.d, self.d))
        for i in range(L):
            for j in range(L):
                T[i, j] = G[i - j] if i >= j else G[j - i].T
        Sigma = T.transpose(2, 0, 3, 1).reshape(self.d * L, self.d * L)
        return 0.5 * (Sigma + Sigma.T)

    def transition(self) -> LinearTransition:
        return LinearTransition(A=self.A.copy(), b=np.zeros(self.d), Q=self.Q.copy())

    def validate(self, traj: np.ndarray) -> ProcessReport:
        x = as_ndl(traj, self.d)                       # (N, d, L)
        N, d, L = x.shape
        xc = x - x.mean(axis=(0, 2))[None, :, None]

        def emp_lag(h: int) -> np.ndarray:
            a, b = xc[:, :, h:], xc[:, :, : L - h]
            return np.einsum("ndl,nel->de", a, b) / (N * (L - h))

        def rel_dev(emp: np.ndarray, tgt: np.ndarray) -> float:
            return float(np.linalg.norm(emp - tgt) / np.linalg.norm(tgt))

        h = max(2, min(self.lag_check, L - 2))
        checks = [
            StatCheck("mean (max abs)", float(np.abs(x.mean(axis=(0, 2))).max()),
                      0.0, "abs"),
            StatCheck("stationary cov rel dev", rel_dev(emp_lag(0), self.Sinf),
                      0.0, "abs"),
            StatCheck("lag-1 xcov rel dev", rel_dev(emp_lag(1), self.A @ self.Sinf),
                      0.0, "abs"),
            StatCheck(f"lag-{h} xcov rel dev",
                      rel_dev(emp_lag(h),
                              np.linalg.matrix_power(self.A, h) @ self.Sinf),
                      0.0, "abs"),
        ]
        return ProcessReport(self.name, checks)


class StochasticOscillator(VectorOU):
    name = "OSC"

    def __init__(self, omega: float = 2.0, zeta: float = 0.15,
                 sigma: float = 1.0, dt: float = 0.05):
        if not (0.0 < zeta < 1.0):
            raise ValueError("zeta in (0, 1) required (underdamped oscillator).")
        if omega <= 0:
            raise ValueError("omega must be positive.")
        Theta = np.array([[0.0, -1.0], [omega**2, 2.0 * zeta * omega]])
        B = np.array([[0.0], [sigma]])
        super().__init__(Theta=Theta, B=B, dt=dt)
        self.omega, self.zeta, self.sigma = float(omega), float(zeta), float(sigma)
        self._desc = dict(omega=self.omega, zeta=self.zeta, sigma=self.sigma,
                          dt=self.dt, d=2)
        omega_d = omega * np.sqrt(1.0 - zeta**2)
        self.lag_check = max(2, int(round((np.pi / 2.0) / (omega_d * dt))))
