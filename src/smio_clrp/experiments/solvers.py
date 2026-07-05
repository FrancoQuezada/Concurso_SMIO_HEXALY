from __future__ import annotations

from typing import Any

from smio_clrp.algorithms.alns import ALNSSolver
from smio_clrp.algorithms.base import Solver, SolverConfig
from smio_clrp.algorithms.constructive.greedy import GreedyNearestDepotSolver
from smio_clrp.algorithms.constructive.multistart import MultiStartConstructiveSolver
from smio_clrp.algorithms.constructive.regret import RegretInsertionSolver
from smio_clrp.algorithms.constructive.savings import SavingsConstructiveSolver
from smio_clrp.algorithms.fixopt import FixOptimizeSolver, HybridALNSFixOptSolver
from smio_clrp.algorithms.local_search.solver import ConstructiveLocalSearchSolver


SUPPORTED_ALGORITHMS = {
    "greedy_nearest_depot",
    "savings",
    "regret",
    "multistart",
    "constructive_ls",
    "alns",
    "fixopt",
    "hybrid",
}


def make_solver(
    algorithm: str,
    seed: int = 1,
    time_limit: float | None = None,
    parameters: dict[str, Any] | None = None,
) -> Solver:
    metadata = dict(parameters or {})
    config = SolverConfig(seed=seed, time_limit_seconds=time_limit, metadata=metadata)
    if algorithm == "greedy_nearest_depot":
        return GreedyNearestDepotSolver(config)
    if algorithm == "savings":
        return SavingsConstructiveSolver(config)
    if algorithm == "regret":
        return RegretInsertionSolver(config)
    if algorithm == "multistart":
        return MultiStartConstructiveSolver(config)
    if algorithm == "constructive_ls":
        return ConstructiveLocalSearchSolver(config)
    if algorithm == "alns":
        return ALNSSolver(config)
    if algorithm == "fixopt":
        return FixOptimizeSolver(config=config)
    if algorithm == "hybrid":
        return HybridALNSFixOptSolver(config)
    raise ValueError(f"Unsupported algorithm '{algorithm}'. Available: {sorted(SUPPORTED_ALGORITHMS)}")
