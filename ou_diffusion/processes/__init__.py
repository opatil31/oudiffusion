from __future__ import annotations
import inspect
from .base import LinearTransition, Process, ProcessReport, StatCheck, as_ndl
from .brownian import BrownianMotion
from .gbm import GeometricBrownianMotion
from .ou import OUProcess
from .cir import CIRProcess
from .jacobi import JacobiProcess
from .pearson import PearsonDiffusion
from .heat import StochasticHeat
from .vector_ou import StochasticOscillator, VectorOU

PROCESSES: dict[str, type[Process]] = {
    "ou": OUProcess,
    "cir": CIRProcess,
    "jacobi": JacobiProcess,
    "bm": BrownianMotion,
    "gbm": GeometricBrownianMotion,
    "vou": VectorOU,
    "osc": StochasticOscillator,
    "heat": StochasticHeat,
}


def get_process(name: str, **kwargs) -> Process:
    key = name.lower()
    if key not in PROCESSES:
        raise KeyError(f"unknown process '{name}'; available: {sorted(PROCESSES)}")
    return PROCESSES[key](**kwargs)


def make_process(name: str, **maybe_kwargs) -> Process:
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
    "CIRProcess",
    "JacobiProcess",
    "PearsonDiffusion",
    "BrownianMotion",
    "GeometricBrownianMotion",
    "VectorOU",
    "StochasticOscillator",
    "StochasticHeat",
    "PROCESSES",
    "get_process",
    "make_process",
]