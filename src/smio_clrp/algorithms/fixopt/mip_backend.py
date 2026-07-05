from __future__ import annotations

from smio_clrp.algorithms.fixopt.backend import BackendUnavailable, FixOptResult
from smio_clrp.algorithms.fixopt.config import FixOptConfig
from smio_clrp.algorithms.fixopt.heuristic_backend import HeuristicFixOptBackend
from smio_clrp.algorithms.fixopt.neighborhoods import FixOptNeighborhood
from smio_clrp.core.instance import Instance
from smio_clrp.core.solution import Solution


class MIPFixOptBackend:
    backend_name = "mip"

    def __init__(self, seed: int = 1) -> None:
        try:
            import gurobipy  # noqa: F401
        except ImportError as exc:
            raise BackendUnavailable("gurobipy is not installed") from exc
        self.seed = seed

    def reoptimize(
        self,
        instance: Instance,
        solution: Solution,
        neighborhood: FixOptNeighborhood,
        config: FixOptConfig,
    ) -> FixOptResult:
        if len(neighborhood.released_customer_ids) > config.max_customers_per_subproblem:
            return FixOptResult(
                None,
                False,
                self.backend_name,
                {"error": "restricted MIP skipped because subproblem is too large"},
            )
        # Conservative first version: keep Gurobi optional and delegate route reconstruction
        # to the validated restricted heuristic. This is a matheuristic hook, not a full CLRP MIP.
        heuristic = HeuristicFixOptBackend(seed=self.seed)
        result = heuristic.reoptimize(instance, solution, neighborhood, config)
        result.backend_used = self.backend_name
        result.metadata["note"] = "MIP backend hook used heuristic reconstruction in this conservative version"
        return result


def gurobi_available() -> bool:
    try:
        import gurobipy  # noqa: F401
    except ImportError:
        return False
    return True
