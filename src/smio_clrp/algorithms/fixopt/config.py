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
    # Default mirrors mip_backend.MAX_EXTRA_CANDIDATE_ROUTES (the historical hardcoded cap);
    # exposed here so a small, tractable instance can be given a much larger candidate
    # budget than the 12-25 customer default subproblem windows ever needed.
    max_extra_candidate_routes: int = 2000
    # Slack added on top of the capacity/avg-demand "typical route size" when capping
    # candidate subset sizes. Default (2) mirrors the historical hardcoded value; too small
    # a cap can make the model unable to even represent the incumbent's own longer routes as
    # a candidate, which showed up as spurious "infeasible" MIP subproblems.
    max_subset_size_slack: int = 2

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.max_customers_per_subproblem <= 0:
            raise ValueError("max_customers_per_subproblem must be positive")
        if self.max_routes_per_subproblem <= 0:
            raise ValueError("max_routes_per_subproblem must be positive")
        if self.backend not in {"auto", "heuristic", "mip"}:
            raise ValueError("backend must be auto, heuristic, or mip")
        if self.max_extra_candidate_routes <= 0:
            raise ValueError("max_extra_candidate_routes must be positive")
