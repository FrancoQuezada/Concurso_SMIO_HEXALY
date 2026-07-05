from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ALNSConfig:
    seed: int = 1
    max_iterations: int = 500
    time_limit_seconds: float | None = None
    num_starts: int = 20
    destroy_fraction_min: float = 0.15
    destroy_fraction_max: float = 0.35
    initial_temperature: float = 10.0
    cooling_rate: float = 0.995
    local_search_frequency: str = "best"
    max_no_improve: int = 150
    repair_method: str = "mixed"
    acceptance_method: str = "simulated_annealing"
    verbose: bool = False
    segment_length: int = 25
    local_search_iterations: int = 10

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.num_starts <= 0:
            raise ValueError("num_starts must be positive")
        if not 0 < self.destroy_fraction_min <= self.destroy_fraction_max <= 1:
            raise ValueError("destroy fractions must satisfy 0 < min <= max <= 1")
        if self.initial_temperature <= 0:
            raise ValueError("initial_temperature must be positive")
        if not 0 < self.cooling_rate <= 1:
            raise ValueError("cooling_rate must be in (0, 1]")
        if self.segment_length <= 0:
            raise ValueError("segment_length must be positive")
