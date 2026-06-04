from __future__ import annotations
import copy
import torch
import torch.nn as nn

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def clone_into(self, model: nn.Module) -> nn.Module:
        """Return a deep copy of `model` with the EMA weights loaded."""
        out = copy.deepcopy(model)
        with torch.no_grad():
            for n, p in out.named_parameters():
                if p.requires_grad and n in self.shadow:
                    p.data.copy_(self.shadow[n])
        return out
