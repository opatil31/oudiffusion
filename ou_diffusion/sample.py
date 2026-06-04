from __future__ import annotations
import torch
import torch.nn as nn
from .schedule import NoiseSchedule

@torch.no_grad()
def ddpm_sample(
    model: nn.Module,
    schedule: NoiseSchedule,
    n: int,
    L: int,
    d: int = 1,
    device: str = "cpu",
    seed: int | None = None,
) -> torch.Tensor:
    device = torch.device(device)
    schedule = schedule.to(device)
    model = model.to(device)
    model.eval()

    gen = None
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)

    def randn(shape):
        return torch.randn(shape, generator=gen, device=device)

    x = randn((n, d, L))
    for k in reversed(range(schedule.T)):
        t = torch.full((n,), k, device=device, dtype=torch.long)
        eps = model(x, t)
        alpha = schedule.alphas[k]
        alpha_bar = schedule.alphas_cumprod[k]
        beta = schedule.betas[k]
        coef = beta / torch.sqrt(1.0 - alpha_bar)
        mean = (x - coef * eps) / torch.sqrt(alpha)
        if k > 0:
            x = mean + torch.sqrt(beta) * randn(x.shape)
        else:
            x = mean
    return x.squeeze(1).cpu() if d == 1 else x.cpu()