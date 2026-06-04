from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class OUConstants:
    a: float          
    q: float          
    s2: float         
    theta: float
    sigma: float
    dt: float


def ou_constants(theta: float, sigma: float, dt: float) -> OUConstants:
    """Return the exact (a, q, s2) constants for the OU process (Eq. 5)."""
    if theta <= 0 or sigma <= 0 or dt <= 0:
        raise ValueError("theta, sigma, dt must all be positive.")
    a = float(np.exp(-theta * dt))
    s2 = sigma**2 / (2.0 * theta)
    q = s2 * (1.0 - np.exp(-2.0 * theta * dt))
    return OUConstants(a=a, q=q, s2=s2, theta=theta, sigma=sigma, dt=dt)


def generate_ou_dataset(
    theta: float = 1.0,
    sigma: float = 1.0,
    dt: float = 0.05,
    L: int = 64,
    N: int = 10_000,
    seed: int | None = 0,
) -> np.ndarray:
    c = ou_constants(theta, sigma, dt)
    rng = np.random.default_rng(seed)

    x = np.empty((N, L), dtype=np.float64)
    x[:, 0] = rng.standard_normal(N) * np.sqrt(c.s2)
    sqrt_q = np.sqrt(c.q)
    for l in range(1, L):
        x[:, l] = c.a * x[:, l - 1] + sqrt_q * rng.standard_normal(N)
    return x


def add_measurement_noise(
    x0: np.ndarray, R: float, seed: int | None = 1
) -> np.ndarray:
    if R < 0:
        raise ValueError("measurement variance R must be non-negative.")
    rng = np.random.default_rng(seed)
    return x0 + np.sqrt(R) * rng.standard_normal(x0.shape)
