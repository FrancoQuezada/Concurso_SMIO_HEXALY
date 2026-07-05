from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Route:
    depot_id: int
    customer_ids: list[int]

    def demand(self, customer_demands: dict[int, float]) -> float:
        return sum(customer_demands[customer_id] for customer_id in self.customer_ids)


@dataclass
class Solution:
    instance_name: str | None
    routes: list[Route]
    reported_cost: float | None = None

    @property
    def opened_depot_ids(self) -> set[int]:
        return {route.depot_id for route in self.routes}


@dataclass
class ValidationResult:
    is_feasible: bool
    cost: float
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
