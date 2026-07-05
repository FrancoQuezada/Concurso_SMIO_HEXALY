from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FixOptConfig:
    seed: int = 1
    max_iterations: int = 50
    time_limit_seconds: float | None = None
    neighborhood_types: list[str] | None = None
    max_customers_per_subproblem: int = 12
    max_routes_per_subproblem: int = 3
    backend: str = "auto"
    mip_time_limit_seconds: float | None = 5.0
    accept_worsening: bool = False
    local_search_after_subproblem: bool = True
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.max_customers_per_subproblem <= 0:
            raise ValueError("max_customers_per_subproblem must be positive")
        if self.max_routes_per_subproblem <= 0:
            raise ValueError("max_routes_per_subproblem must be positive")
        if self.backend not in {"auto", "heuristic", "mip"}:
            raise ValueError("backend must be auto, heuristic, or mip")
