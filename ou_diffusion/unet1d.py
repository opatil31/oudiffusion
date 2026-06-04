from __future__ import annotations
import math
import torch
import torch.nn as nn

class SinusoidalTimeEmbedding(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "embedding dim must be even"
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


def _groups(ch: int, max_groups: int = 8) -> int:
    return math.gcd(max_groups, ch)


class ResBlock1d(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, t_hidden: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_ch), in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(t_hidden, out_ch)
        self.norm2 = nn.GroupNorm(_groups(out_ch), out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch
            else nn.Identity()
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_proj(temb).unsqueeze(-1)   # (B, out_ch, 1) broadcast
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class UNet1D(nn.Module):

    def __init__(
        self,
        in_ch: int = 1,
        base: int = 32,
        t_dim: int = 64,
        t_hidden: int = 128,
    ):
        super().__init__()
        C = base
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(t_dim),
            nn.Linear(t_dim, t_hidden),
            nn.SiLU(),
            nn.Linear(t_hidden, t_hidden),
        )

        self.conv_in = nn.Conv1d(in_ch, C, kernel_size=3, padding=1)
        self.res1 = ResBlock1d(C, C, t_hidden) #skip 1                  
        self.down1 = nn.Conv1d(C, 2 * C, kernel_size=3, stride=2, padding=1)
        self.res2 = ResBlock1d(2 * C, 2 * C, t_hidden) #skip 2          
        self.down2 = nn.Conv1d(2 * C, 4 * C, kernel_size=3, stride=2, padding=1)

        self.mid = ResBlock1d(4 * C, 4 * C, t_hidden)           

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(4 * C, 2 * C, kernel_size=3, padding=1),
        )
        self.res_up2 = ResBlock1d(4 * C, 2 * C, t_hidden)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(2 * C, C, kernel_size=3, padding=1),
        )
        self.res_up1 = ResBlock1d(2 * C, C, t_hidden)

        self.norm_out = nn.GroupNorm(_groups(C), C)
        self.conv_out = nn.Conv1d(C, in_ch, kernel_size=3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % 4 != 0:
            raise ValueError(f"trajectory length L={x.shape[-1]} must be divisible by 4.")
        temb = self.time_mlp(t)

        x = self.conv_in(x)
        s1 = self.res1(x, temb)
        x = self.down1(s1)
        s2 = self.res2(x, temb)
        x = self.down2(s2)

        x = self.mid(x, temb)

        x = self.up2(x)
        x = self.res_up2(torch.cat([x, s2], dim=1), temb)
        x = self.up1(x)
        x = self.res_up1(torch.cat([x, s1], dim=1), temb)

        return self.conv_out(self.act(self.norm_out(x)))
