from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VNSConfig:
    seed: int = 1
    max_iterations: int = 50
    time_limit_seconds: float | None = None
    local_search_iterations: int = 10
    shake_fractions: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30)
    local_search_operators: tuple[str, ...] = (
        "two_opt_star",
        "route_depot_reassignment",
        "two_opt",
        "route_reinsertion",
    )

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError("vns_iterations must be positive")
        if self.local_search_iterations <= 0:
            raise ValueError("vns_local_search_iterations must be positive")
        if not self.shake_fractions or any(not 0 < value <= 1 for value in self.shake_fractions):
            raise ValueError("vns_shake_fractions must contain values in (0, 1]")
