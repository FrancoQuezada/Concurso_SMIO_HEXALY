"""Algorithm interfaces and solver implementations."""

from smio_clrp.algorithms.base import Solver, SolverConfig, SolverResult
from smio_clrp.algorithms.alns import ALNSSolver
from smio_clrp.algorithms.constructive import (
    GreedyNearestDepotSolver,
    MultiStartConstructiveSolver,
    RegretInsertionSolver,
    SavingsConstructiveSolver,
)
from smio_clrp.algorithms.fixopt import FixOptimizeSolver, HybridALNSFixOptSolver
from smio_clrp.algorithms.local_search import ConstructiveLocalSearchSolver, LocalSearchSolver

__all__ = [
    "ConstructiveLocalSearchSolver",
    "ALNSSolver",
    "GreedyNearestDepotSolver",
    "LocalSearchSolver",
    "FixOptimizeSolver",
    "HybridALNSFixOptSolver",
    "MultiStartConstructiveSolver",
    "RegretInsertionSolver",
    "SavingsConstructiveSolver",
    "Solver",
    "SolverConfig",
    "SolverResult",
]
