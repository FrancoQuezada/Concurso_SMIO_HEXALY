from __future__ import annotations

import time

from smio_clrp.algorithms.base import Solver, SolverConfig, SolverResult
from smio_clrp.algorithms.constructive.multistart import MultiStartConfig, multi_start_constructive
from smio_clrp.algorithms.local_search.driver import improve_solution
from smio_clrp.core.instance import Instance
from smio_clrp.core.solution import Solution
from smio_clrp.evaluation.cost import objective_cost
from smio_clrp.evaluation.validator import validate_solution


class LocalSearchSolver(Solver):
    algorithm_name = "local_search"

    def __init__(self, initial_solution: Solution, config: SolverConfig | None = None) -> None:
        super().__init__(config)
        self.initial_solution = initial_solution

    def solve(self, instance: Instance) -> SolverResult:
        start = time.perf_counter()
        try:
            solution = improve_solution(
                instance,
                self.initial_solution,
                operators=self.config.metadata.get("operators"),
                max_iterations=int(self.config.metadata.get("max_iterations", 50)),
                time_limit_seconds=self.config.time_limit_seconds,
            )
            validation = validate_solution(instance, solution)
            return SolverResult(
                solution=solution if validation.is_feasible else None,
                cost=validation.cost if validation.is_feasible else None,
                feasible=validation.is_feasible,
                runtime_seconds=time.perf_counter() - start,
                seed=self.config.seed,
                algorithm_name=self.algorithm_name,
                metadata={"validation_errors": validation.errors},
            )
        except ValueError as exc:
            return SolverResult(None, None, False, time.perf_counter() - start, self.config.seed, self.algorithm_name, {"error": str(exc)})


class ConstructiveLocalSearchSolver(Solver):
    algorithm_name = "constructive_ls"

    def solve(self, instance: Instance) -> SolverResult:
        start = time.perf_counter()
        multi = multi_start_constructive(
            instance,
            MultiStartConfig(
                num_starts=int(self.config.metadata.get("num_starts", 10)),
                seed=self.config.seed,
                constructive_methods=self.config.metadata.get("constructive_methods"),
                time_limit_seconds=self.config.time_limit_seconds,
            ),
        )
        if multi.solution is None:
            multi.algorithm_name = self.algorithm_name
            multi.runtime_seconds = time.perf_counter() - start
            return multi
        remaining_time = None
        if self.config.time_limit_seconds is not None:
            remaining_time = max(0.0, self.config.time_limit_seconds - (time.perf_counter() - start))
        try:
            improved = improve_solution(
                instance,
                multi.solution,
                operators=self.config.metadata.get("operators"),
                max_iterations=int(self.config.metadata.get("max_iterations", 50)),
                time_limit_seconds=remaining_time,
            )
            validation = validate_solution(instance, improved)
            return SolverResult(
                solution=improved if validation.is_feasible else None,
                cost=objective_cost(instance, improved) if validation.is_feasible else None,
                feasible=validation.is_feasible,
                runtime_seconds=time.perf_counter() - start,
                seed=self.config.seed,
                algorithm_name=self.algorithm_name,
                metadata={"initial_cost": multi.cost, "validation_errors": validation.errors},
            )
        except ValueError as exc:
            return SolverResult(None, None, False, time.perf_counter() - start, self.config.seed, self.algorithm_name, {"error": str(exc)})
