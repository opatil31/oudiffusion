from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ema import EMA
from .schedule import NoiseSchedule, q_sample
from .unet1d import UNet1D

@dataclass
class TrainConfig:
    steps: int = 20_000
    batch_size: int = 256
    lr: float = 2e-4
    grad_clip: float | None = 1.0
    ema_decay: float = 0.999
    base_channels: int = 32
    log_every: int = 1_000
    seed: int = 0


def train_denoiser(
    x0: np.ndarray,
    schedule: NoiseSchedule,
    cfg: TrainConfig = TrainConfig(),
    device: str = "cpu",
    verbose: bool = True,
):
    torch.manual_seed(cfg.seed)
    device = torch.device(device)
    schedule = schedule.to(device)

    data = torch.as_tensor(x0, dtype=torch.float32, device=device).unsqueeze(1)  # (N,1,L)
    N = data.shape[0]

    model = UNet1D(in_ch=1, base=cfg.base_channels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    ema = EMA(model, decay=cfg.ema_decay)
    gen = torch.Generator(device=device).manual_seed(cfg.seed)

    losses = []
    model.train()
    for step in range(1, cfg.steps + 1):
        idx = torch.randint(0, N, (cfg.batch_size,), generator=gen, device=device)
        xb = data[idx]                                              # (B,1,L)
        t = torch.randint(0, schedule.T, (cfg.batch_size,), generator=gen, device=device)
        xt, eps = q_sample(xb, t, schedule)
        eps_pred = model(xt, t)
        loss = F.mse_loss(eps_pred, eps)

        opt.zero_grad()
        loss.backward()
        if cfg.grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        ema.update(model)

        if step % cfg.log_every == 0 or step == 1:
            losses.append((step, loss.item()))
            if verbose:
                print(f"step {step:>6d}/{cfg.steps}   loss {loss.item():.5f}")

    return model, ema, losses
