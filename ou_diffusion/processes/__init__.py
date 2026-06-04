from __future__ import annotations
from .base import LinearTransition, Process, ProcessReport, StatCheck, as_ndl
from .brownian import BrownianMotion
from .ou import OUProcess

PROCESSES: dict[str, type[Process]] = {
    "ou": OUProcess,
    "bm": BrownianMotion,
}

def get_process(name: str, **kwargs) -> Process:
    """Construct a process by registry name (e.g. get_process('bm', mu=0.3))."""
    key = name.lower()
    if key not in PROCESSES:
        raise KeyError(f"unknown process '{name}'; available: {sorted(PROCESSES)}")
    return PROCESSES[key](**kwargs)


__all__ = [
    "Process",
    "ProcessReport",
    "StatCheck",
    "LinearTransition",
    "as_ndl",
    "OUProcess",
    "BrownianMotion",
    "PROCESSES",
    "get_process",
]