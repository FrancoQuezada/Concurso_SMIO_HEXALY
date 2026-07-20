from __future__ import annotations

import time

from smio_clrp.algorithms.alns import ALNSSolver
from smio_clrp.algorithms.base import Solver, SolverConfig, SolverResult
from smio_clrp.algorithms.clustering import ClusterFirstSolver
from smio_clrp.algorithms.vns.vns_solver import VNSSolver
from smio_clrp.core.instance import Instance
from smio_clrp.core.solution import Solution


class HybridALNSVNSSolver(Solver):
    algorithm_name = "hybrid"

    def __init__(self, config: SolverConfig | None = None, initial_solution: Solution | None = None) -> None:
        super().__init__(config)
        self.initial_solution = initial_solution

    def solve(self, instance: Instance) -> SolverResult:
        start = time.perf_counter()
        alns_budget, vns_budget = _hybrid_time_budgets(self.config)
        alns_metadata = {**self.config.metadata, "vns_on_stagnation": True}
        alns = ALNSSolver(
            SolverConfig(self.config.seed, alns_budget, alns_metadata),
            initial_solution=self.initial_solution,
        ).solve(instance)
        if alns.solution is None:
            alns.algorithm_name = self.algorithm_name
            return alns

        if self.config.time_limit_seconds is not None:
            remaining = max(0.0, self.config.time_limit_seconds - (time.perf_counter() - start))
            vns_budget = remaining if vns_budget is None else min(vns_budget, remaining)
        vns = VNSSolver(
            alns.solution,
            SolverConfig(self.config.seed, vns_budget, dict(self.config.metadata)),
        ).solve(instance)
        use_vns = vns.solution is not None and vns.cost is not None and (alns.cost is None or vns.cost <= alns.cost)
        final_solution = vns.solution if use_vns else alns.solution
        final_cost = vns.cost if use_vns else alns.cost
        return SolverResult(
            final_solution,
            final_cost,
            final_solution is not None,
            time.perf_counter() - start,
            self.config.seed,
            self.algorithm_name,
            {
                "alns_initial_cost": alns.metadata.get("initial_cost"),
                "alns_best_cost": alns.cost,
                "vns_initial_cost": vns.metadata.get("initial_cost") if vns.metadata else alns.cost,
                "vns_best_cost": vns.cost,
                "final_cost": final_cost,
                "alns_runtime_seconds": alns.runtime_seconds,
                "vns_runtime_seconds": vns.runtime_seconds,
                "total_runtime_seconds": time.perf_counter() - start,
                "alns_time_budget_seconds": alns_budget,
                "vns_time_budget_seconds": vns_budget,
                "vns_iterations": vns.metadata.get("iterations", 0),
                "vns_improvements": vns.metadata.get("improvements", 0),
                "vns_stagnation_kicks": alns.metadata.get("vns_stagnation_kicks", 0),
                "vns_stagnation_improvements": alns.metadata.get("vns_stagnation_improvements", 0),
                "vns_stagnation_gain": alns.metadata.get("vns_stagnation_gain", 0.0),
            },
        )


class ClusteredHybridALNSVNSSolver(Solver):
    algorithm_name = "clustered_hybrid"

    def solve(self, instance: Instance) -> SolverResult:
        start = time.perf_counter()
        clustered = ClusterFirstSolver(self.config).solve(instance)
        if clustered.solution is None:
            clustered.algorithm_name = self.algorithm_name
            return clustered
        remaining = None
        if self.config.time_limit_seconds is not None:
            remaining = max(0.0, self.config.time_limit_seconds - (time.perf_counter() - start))
        result = HybridALNSVNSSolver(
            SolverConfig(self.config.seed, remaining, dict(self.config.metadata)),
            initial_solution=clustered.solution,
        ).solve(instance)
        result.algorithm_name = self.algorithm_name
        result.runtime_seconds = time.perf_counter() - start
        result.metadata = {
            "cluster_initial_cost": clustered.cost,
            "cluster_runtime_seconds": clustered.runtime_seconds,
            **result.metadata,
            "total_runtime_seconds": time.perf_counter() - start,
        }
        return result


def _hybrid_time_budgets(config: SolverConfig) -> tuple[float | None, float | None]:
    if config.time_limit_seconds is None:
        return None, None
    fraction = float(config.metadata.get("hybrid_alns_fraction", 0.75))
    if not 0 < fraction < 1:
        raise ValueError("hybrid_alns_fraction must be between 0 and 1")
    alns_budget = config.time_limit_seconds * fraction
    return alns_budget, config.time_limit_seconds - alns_budget
