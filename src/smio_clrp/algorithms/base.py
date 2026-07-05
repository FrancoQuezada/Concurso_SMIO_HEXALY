from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from smio_clrp.core.instance import Instance
from smio_clrp.core.solution import Solution


@dataclass
class SolverConfig:
    seed: int = 1
    time_limit_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SolverResult:
    solution: Solution | None
    cost: float | None
    feasible: bool
    runtime_seconds: float
    seed: int
    algorithm_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


class Solver(ABC):
    algorithm_name = "abstract_solver"

    def __init__(self, config: SolverConfig | None = None) -> None:
        self.config = config or SolverConfig()

    @abstractmethod
    def solve(self, instance: Instance) -> SolverResult:
        raise NotImplementedError
