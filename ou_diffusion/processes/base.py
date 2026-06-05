"""
I decided to make a Process abstraction to help us implement various Processes.
Here, I define a Process as an exact trajectory sampler plus the closed forms needed to
verify a generative model trained on it:
  - exact_sample(N, L, seed) -- trajectories with no discretization error,
    shaped (N, d, L): d state dimensions (= network channels), L time steps.
  - validate(traj)           -- process-specific statistic checks against
    exact targets
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np

@dataclass(frozen=True)
class LinearTransition:
    A: np.ndarray
    b: np.ndarray
    Q: np.ndarray

@dataclass
class StatCheck:
    name: str
    generated: float
    target: float
    kind: str = "abs"  # "abs" or "rel"

    @property
    def error(self) -> float:
        if self.kind == "rel":
            return abs(self.generated - self.target) / max(abs(self.target), 1e-12)
        return abs(self.generated - self.target)

@dataclass
class ProcessReport:
    process: str
    checks: list[StatCheck] = field(default_factory=list)
    def __str__(self) -> str:
        head = f"{self.process} statistic            generated      target      error"
        rows = []
        for c in self.checks:
            unit = "% (rel)" if c.kind == "rel" else "  (abs)"
            err = c.error * 100 if c.kind == "rel" else c.error
            rows.append(
                f"{c.name:<22s} {c.generated:11.4f} {c.target:11.4f} "
                f"{err:10.4f}{unit}"
            )
        return "\n".join([head] + rows)

    def max_error(self, kind: str | None = None) -> float:
        sel = [c for c in self.checks if kind is None or c.kind == kind]
        return max(c.error for c in sel) if sel else 0.0

def as_ndl(traj: np.ndarray, d: int) -> np.ndarray:
    a = np.asarray(traj, dtype=np.float64)
    if a.ndim == 2:
        if d != 1:
            raise ValueError(f"got 2-D trajectories but process has d={d}")
        a = a[:, None, :]
    if a.ndim != 3 or a.shape[1] != d:
        raise ValueError(f"expected (N, {d}, L) trajectories, got {a.shape}")
    return a

class Process(ABC):
    name: str = "process"
    d: int = 1
    is_gaussian: bool = False # True -> mean()/covariance() -> oracle + floor
    @abstractmethod
    def exact_sample(self, N: int, L: int, seed: int | None = None) -> np.ndarray:
        """Exact trajectories, shape (N, d, L)."""

    @abstractmethod
    def validate(self, traj: np.ndarray) -> ProcessReport:
        """Compare trajectory statistics against exact targets."""
    def describe(self) -> dict:
        """Constructor parameters, for logging."""
        return {}
    # closed forms overrides
    def mean(self, L: int) -> np.ndarray:
        """Mean of the flattened trajectory, shape (d*L,)."""
        raise NotImplementedError(f"{self.name} does not provide a Gaussian mean")

    def covariance(self, L: int) -> np.ndarray:
        """Covariance of the flattened trajectory, shape (d*L, d*L)."""
        raise NotImplementedError(f"{self.name} does not provide a Gaussian covariance")

    def transition(self) -> LinearTransition:
        """Exact one-step kernel (linear-Gaussian processes)."""
        raise NotImplementedError(f"{self.name} does not provide a linear transition")
