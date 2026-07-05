from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from smio_clrp.core.solution import Solution


@dataclass
class DestroyResult:
    partial_solution: Solution
    removed_customer_ids: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RepairResult:
    solution: Solution | None
    success: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ALNSState:
    current_solution: Solution
    current_cost: float
    best_solution: Solution
    best_cost: float
    initial_cost: float
    temperature: float
    iterations: int = 0
    accepted_moves: int = 0
    improving_moves: int = 0
    best_improvements: int = 0
    failed_repairs: int = 0
    no_improve_iterations: int = 0


@dataclass
class ALNSOperatorStats:
    calls: int = 0
    successes: int = 0
    improvements: int = 0
    best_improvements: int = 0
    cumulative_runtime: float = 0.0

    def as_dict(self) -> dict[str, int | float]:
        return {
            "calls": self.calls,
            "successes": self.successes,
            "improvements": self.improvements,
            "best_improvements": self.best_improvements,
            "cumulative_runtime": self.cumulative_runtime,
        }
