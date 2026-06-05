from __future__ import annotations
import inspect
from .base import LinearTransition, Process, ProcessReport, StatCheck, as_ndl
from .brownian import BrownianMotion
from .ou import OUProcess
from .gbm import GeometricBrownianMotion
from .vector_ou import VectorOU, StochasticOscillator

PROCESSES: dict[str, type[Process]] = {
    "ou": OUProcess,
    "bm": BrownianMotion,
    "gbm": GeometricBrownianMotion,
    "vou": VectorOU,
    "osc": StochasticOscillator,
}

def get_process(name: str, **kwargs) -> Process:
    """Construct a process by registry name (e.g. get_process('bm', mu=0.3))."""
    key = name.lower()
    if key not in PROCESSES:
        raise KeyError(f"unknown process '{name}'; available: {sorted(PROCESSES)}")
    return PROCESSES[key](**kwargs)

def make_process(name: str, **maybe_kwargs) -> Process:
    cls = PROCESSES[name.lower()]
    sig = inspect.signature(cls.__init__)
    use = {k: v for k, v in maybe_kwargs.items()
           if v is not None and k in sig.parameters}
    return cls(**use)


__all__ = [
    "Process",
    "ProcessReport",
    "StatCheck",
    "LinearTransition",
    "as_ndl",
    "OUProcess",
    "BrownianMotion",
    "VectorOU",
    "StochasticOscillator",
    "PROCESSES",
    "get_process",
]