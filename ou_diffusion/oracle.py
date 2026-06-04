from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


def gaussian_loss_floor(Sigma_or_eigvals: np.ndarray, alphas_cumprod) -> float:
    S = np.asarray(Sigma_or_eigvals, dtype=np.float64)
    lam = np.linalg.eigvalsh(S) if S.ndim == 2 else S
    lam = np.clip(lam, 0.0, None)
    ab = np.asarray(alphas_cumprod, dtype=np.float64)
    tr_inv = np.mean(1.0 / (ab[:, None] * lam[None, :] + (1.0 - ab[:, None])), axis=1)
    floor_t = 1.0 - (1.0 - ab) * tr_inv
    return float(floor_t.mean())


class AnalyticGaussianDenoiser(nn.Module):

    def __init__(self, mean: np.ndarray, Sigma: np.ndarray, schedule):
        super().__init__()
        mu = np.asarray(mean, dtype=np.float64).reshape(-1)
        S = np.asarray(Sigma, dtype=np.float64)
        if S.shape != (mu.size, mu.size):
            raise ValueError(f"Sigma {S.shape} incompatible with mean ({mu.size},)")
        lam, V = np.linalg.eigh(S)
        lam = np.clip(lam, 0.0, None)
        self.register_buffer("mu", torch.from_numpy(mu))
        self.register_buffer("lam", torch.from_numpy(lam))
        self.register_buffer("V", torch.from_numpy(V))
        self.register_buffer("ab", schedule.alphas_cumprod.double())
        self.register_buffer("sqrt_omab", schedule.sqrt_omab.double())

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, shape = x.shape[0], x.shape
        k = int(t[0].item()) # lvl is shared across batch
        ab = self.ab[k]
        z = x.reshape(B, -1).double() - torch.sqrt(ab) * self.mu      # (B, D)
        w = z @ self.V #eigenbasis coords
        w = w / (ab * self.lam + (1.0 - ab))
        eps = self.sqrt_omab[k] * (w @ self.V.T)
        return eps.reshape(shape).to(x.dtype)
