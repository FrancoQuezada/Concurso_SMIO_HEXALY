"""Algorithm interfaces and solver implementations."""

from smio_clrp.algorithms.base import Solver, SolverConfig, SolverResult
from smio_clrp.algorithms.alns import ALNSSolver
from smio_clrp.algorithms.constructive import (
    GreedyNearestDepotSolver,
    MultiStartConstructiveSolver,
    RegretInsertionSolver,
    SavingsConstructiveSolver,
)
from smio_clrp.algorithms.clustering import ClusteredConstructiveSolver, ClusteredHybridSolver
from smio_clrp.algorithms.fixopt import FixOptimizeSolver, HybridALNSFixOptSolver
from smio_clrp.algorithms.local_search import ConstructiveLocalSearchSolver, LocalSearchSolver
from smio_clrp.algorithms.vns import ClusteredHybridALNSVNSSolver, HybridALNSVNSSolver, VNSSolver

__all__ = [
    "ConstructiveLocalSearchSolver",
    "ClusteredConstructiveSolver",
    "ClusteredHybridSolver",
    "ClusteredHybridALNSVNSSolver",
    "ALNSSolver",
    "GreedyNearestDepotSolver",
    "LocalSearchSolver",
    "FixOptimizeSolver",
    "HybridALNSFixOptSolver",
    "HybridALNSVNSSolver",
    "MultiStartConstructiveSolver",
    "RegretInsertionSolver",
    "SavingsConstructiveSolver",
    "Solver",
    "SolverConfig",
    "SolverResult",
    "VNSSolver",
]
