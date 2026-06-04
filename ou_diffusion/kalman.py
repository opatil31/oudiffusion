from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .ou_process import OUConstants

@dataclass
class KalmanResult:
    x_filt: np.ndarray
    P_filt: np.ndarray
    gain: np.ndarray

def kalman_filter(y: np.ndarray, c: OUConstants, R: float) -> KalmanResult:
    if R <= 0:
        raise ValueError("measurement variance R must be positive for filtering.")
    y = np.atleast_2d(y).astype(np.float64)
    N, L = y.shape
    a, q, s2 = c.a, c.q, c.s2

    x_filt = np.empty((N, L), dtype=np.float64)
    P_filt = np.empty(L, dtype=np.float64)
    gain = np.empty(L, dtype=np.float64)

    K0 = s2 / (s2 + R)
    x_filt[:, 0] = K0 * y[:, 0]
    P_filt[0] = s2 * R / (s2 + R)          
    gain[0] = K0

    for l in range(1, L):
        P_pred = a**2 * P_filt[l - 1] + q              
        K = P_pred / (P_pred + R)                       
        x_pred = a * x_filt[:, l - 1]                   
        x_filt[:, l] = x_pred + K * (y[:, l] - x_pred)  
        P_filt[l] = (1.0 - K) * P_pred                  
        gain[l] = K

    return KalmanResult(x_filt=x_filt, P_filt=P_filt, gain=gain)


def steady_state_variance(c: OUConstants, R: float) -> float:
    """Fixed point P_inf of the scalar Riccati recursion P = (1-K)(a^2 P + q)."""
    a2 = c.a**2
    A = a2
    B = c.q + (1.0 - a2) * R
    C = -c.q * R
    disc = B**2 - 4.0 * A * C
    return float((-B + np.sqrt(disc)) / (2.0 * A))
