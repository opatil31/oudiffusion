from __future__ import annotations
import inspect
from .base import LinearTransition, Process, ProcessReport, StatCheck, as_ndl
from .brownian import BrownianMotion
from .gbm import GeometricBrownianMotion
from .ou import OUProcess
from .heat import StochasticHeat
from .vector_ou import StochasticOscillator, VectorOU

PROCESSES: dict[str, type[Process]] = {
    "ou": OUProcess,
    "bm": BrownianMotion,
    "gbm": GeometricBrownianMotion,
    "vou": VectorOU,
    "osc": StochasticOscillator,
    "heat": StochasticHeat,
}


def get_process(name: str, **kwargs) -> Process:
    """Construct a process by registry name (e.g. get_process('bm', mu=0.3))."""
    key = name.lower()
    if key not in PROCESSES:
        raise KeyError(f"unknown process '{name}'; available: {sorted(PROCESSES)}")
    return PROCESSES[key](**kwargs)


def make_process(name: str, **maybe_kwargs) -> Process:
    """Like get_process, but drops None values and parameters the class does
    not accept -- convenient for CLI scripts with shared flags whose defaults
    should fall through to each process's own defaults."""
    key = name.lower()
    if key not in PROCESSES:
        raise KeyError(f"unknown process '{name}'; available: {sorted(PROCESSES)}")
    cls = PROCESSES[key]
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
    "GeometricBrownianMotion",
    "VectorOU",
    "StochasticOscillator",
    "StochasticHeat",
    "PROCESSES",
    "get_process",
    "make_process",
]