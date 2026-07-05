from __future__ import annotations

from smio_clrp.algorithms.base import Solver, SolverResult
from smio_clrp.core.instance import Instance


class FixOptimizeSolver(Solver):
    algorithm_name = "fixopt_skeleton"

    def solve(self, instance: Instance) -> SolverResult:
        raise NotImplementedError("Fix-and-optimize restricted subproblems are planned later")


class HybridSolver(Solver):
    algorithm_name = "hybrid_skeleton"

    def solve(self, instance: Instance) -> SolverResult:
        raise NotImplementedError("Hybrid constructive/ALNS/fixopt orchestration is planned later")
