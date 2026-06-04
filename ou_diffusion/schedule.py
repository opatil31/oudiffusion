from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass
class NoiseSchedule:
    T: int
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    sqrt_ab: torch.Tensor
    sqrt_omab: torch.Tensor
    posterior_var: torch.Tensor

    def to(self, device) -> "NoiseSchedule":
        return NoiseSchedule(
            T=self.T,
            betas=self.betas.to(device),
            alphas=self.alphas.to(device),
            alphas_cumprod=self.alphas_cumprod.to(device),
            sqrt_ab=self.sqrt_ab.to(device),
            sqrt_omab=self.sqrt_omab.to(device),
            posterior_var=self.posterior_var.to(device),
        )

def make_linear_schedule(
    T: int = 200, beta_start: float = 1e-4, beta_end: float = 0.02
) -> NoiseSchedule:
    betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float64)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat(
        [torch.ones(1, dtype=torch.float64), alphas_cumprod[:-1]]
    )
    posterior_var = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    return NoiseSchedule(
        T=T,
        betas=betas.float(),
        alphas=alphas.float(),
        alphas_cumprod=alphas_cumprod.float(),
        sqrt_ab=torch.sqrt(alphas_cumprod).float(),
        sqrt_omab=torch.sqrt(1.0 - alphas_cumprod).float(),
        posterior_var=posterior_var.float(),
    )

def _bcast(coef_t: torch.Tensor, ndim: int) -> torch.Tensor:
    return coef_t.view(-1, *([1] * (ndim - 1)))

def q_sample(
    x0: torch.Tensor,
    t: torch.Tensor,
    schedule: NoiseSchedule,
    noise: torch.Tensor | None = None,
):
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_ab = _bcast(schedule.sqrt_ab[t], x0.ndim)
    sqrt_omab = _bcast(schedule.sqrt_omab[t], x0.ndim)
    return sqrt_ab * x0 + sqrt_omab * noise, noise