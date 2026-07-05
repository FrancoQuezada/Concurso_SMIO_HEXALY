from __future__ import annotations

from smio_clrp.algorithms.base import Solver, SolverResult
from smio_clrp.core.instance import Instance


class ALNSSolver(Solver):
    algorithm_name = "alns_skeleton"

    def solve(self, instance: Instance) -> SolverResult:
        raise NotImplementedError("ALNS will be implemented after destroy/repair operators mature")
