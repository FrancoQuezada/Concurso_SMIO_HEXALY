from __future__ import annotations

import random
from dataclasses import dataclass, field

from smio_clrp.algorithms.alns.operators import ALNSOperatorStats


@dataclass
class AdaptiveRouletteWheel:
    operator_names: list[str]
    reaction_factor: float = 0.25
    weights: dict[str, float] = field(init=False)
    stats: dict[str, ALNSOperatorStats] = field(init=False)
    segment_scores: dict[str, float] = field(init=False)
    segment_counts: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.weights = {name: 1.0 for name in self.operator_names}
        self.stats = {name: ALNSOperatorStats() for name in self.operator_names}
        self.segment_scores = {name: 0.0 for name in self.operator_names}
        self.segment_counts = {name: 0 for name in self.operator_names}

    def select(self, rng: random.Random) -> str:
        total = sum(self.weights.values())
        draw = rng.random() * total
        cumulative = 0.0
        for name in self.operator_names:
            cumulative += self.weights[name]
            if draw <= cumulative:
                return name
        return self.operator_names[-1]

    def record_call(self, name: str, runtime: float) -> None:
        self.stats[name].calls += 1
        self.stats[name].cumulative_runtime += runtime
        self.segment_counts[name] += 1

    def reward(self, name: str, score: float, accepted: bool, improved: bool, best: bool) -> None:
        self.segment_scores[name] += score
        if accepted:
            self.stats[name].successes += 1
        if improved:
            self.stats[name].improvements += 1
        if best:
            self.stats[name].best_improvements += 1

    def update_weights(self) -> None:
        for name in self.operator_names:
            count = self.segment_counts[name]
            if count == 0:
                continue
            average_score = self.segment_scores[name] / count
            self.weights[name] = (
                (1.0 - self.reaction_factor) * self.weights[name]
                + self.reaction_factor * max(0.1, average_score)
            )
            self.segment_scores[name] = 0.0
            self.segment_counts[name] = 0

    def stats_as_dict(self) -> dict[str, dict[str, int | float]]:
        return {name: stats.as_dict() for name, stats in self.stats.items()}
