from __future__ import annotations

import math
import random
from dataclasses import dataclass


def accept_if_better(candidate_cost: float, current_cost: float, tolerance: float = 1e-9) -> bool:
    return candidate_cost <= current_cost + tolerance


@dataclass
class SimulatedAnnealingAcceptance:
    temperature: float = 10.0
    cooling_rate: float = 0.995

    def accept(self, candidate_cost: float, current_cost: float, rng: random.Random) -> bool:
        if candidate_cost <= current_cost:
            return True
        probability = math.exp(-(candidate_cost - current_cost) / max(self.temperature, 1e-12))
        return rng.random() < probability

    def cool(self) -> float:
        self.temperature *= self.cooling_rate
        return self.temperature


@dataclass
class RecordToRecordAcceptance:
    tolerance: float = 0.01

    def accept(self, candidate_cost: float, reference_cost: float) -> bool:
        return candidate_cost <= reference_cost * (1.0 + self.tolerance)
