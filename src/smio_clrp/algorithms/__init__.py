"""Algorithm interfaces and solver implementations."""

from smio_clrp.algorithms.base import Solver, SolverConfig, SolverResult
from smio_clrp.algorithms.constructive import (
    GreedyNearestDepotSolver,
    MultiStartConstructiveSolver,
    RegretInsertionSolver,
    SavingsConstructiveSolver,
)
from smio_clrp.algorithms.local_search import ConstructiveLocalSearchSolver, LocalSearchSolver

__all__ = [
    "ConstructiveLocalSearchSolver",
    "GreedyNearestDepotSolver",
    "LocalSearchSolver",
    "MultiStartConstructiveSolver",
    "RegretInsertionSolver",
    "SavingsConstructiveSolver",
    "Solver",
    "SolverConfig",
    "SolverResult",
]
