from __future__ import annotations

import time
from dataclasses import dataclass

from smio_clrp.algorithms.base import Solver, SolverConfig, SolverResult
from smio_clrp.algorithms.constructive.greedy import solve_greedy_nearest_depot
from smio_clrp.algorithms.constructive.regret import regret_insertion
from smio_clrp.algorithms.constructive.savings import savings_routes_per_depot
from smio_clrp.core.instance import Instance
from smio_clrp.core.solution import Solution
from smio_clrp.evaluation.cost import objective_cost
from smio_clrp.evaluation.validator import validate_solution


DEFAULT_METHODS = ["greedy_nearest_depot", "savings", "regret2", "regret3"]


@dataclass
class MultiStartConfig:
    num_starts: int = 10
    seed: int = 1
    constructive_methods: list[str] | None = None
    time_limit_seconds: float | None = None


class MultiStartConstructiveSolver(Solver):
    algorithm_name = "multistart"

    def solve(self, instance: Instance) -> SolverResult:
        config = MultiStartConfig(
            num_starts=int(self.config.metadata.get("num_starts", 10)),
            seed=self.config.seed,
            constructive_methods=self.config.metadata.get("constructive_methods"),
            time_limit_seconds=self.config.time_limit_seconds,
        )
        start = time.perf_counter()
        result = multi_start_constructive(instance, config)
        result.runtime_seconds = time.perf_counter() - start
        return result


def multi_start_constructive(instance: Instance, config: MultiStartConfig) -> SolverResult:
    start = time.perf_counter()
    methods = config.constructive_methods or DEFAULT_METHODS
    best_solution: Solution | None = None
    best_cost = float("inf")
    attempts: list[dict[str, object]] = []

    for start_index in range(max(1, config.num_starts)):
        if config.time_limit_seconds is not None and time.perf_counter() - start >= config.time_limit_seconds:
            break
        method = methods[start_index % len(methods)]
        try:
            solution = _run_method(instance, method)
            validation = validate_solution(instance, solution)
            cost = validation.cost if validation.is_feasible else float("inf")
            attempts.append({"method": method, "feasible": validation.is_feasible, "cost": cost})
            if validation.is_feasible and (cost, method) < (best_cost, ""):
                best_solution = solution
                best_cost = cost
        except ValueError as exc:
            attempts.append({"method": method, "feasible": False, "error": str(exc)})

    if best_solution is None:
        return SolverResult(
            solution=None,
            cost=None,
            feasible=False,
            runtime_seconds=time.perf_counter() - start,
            seed=config.seed,
            algorithm_name="multistart",
            metadata={"attempts": attempts, "error": "No feasible constructive solution found"},
        )

    return SolverResult(
        solution=best_solution,
        cost=objective_cost(instance, best_solution),
        feasible=True,
        runtime_seconds=time.perf_counter() - start,
        seed=config.seed,
        algorithm_name="multistart",
        metadata={"attempts": attempts},
    )


def _run_method(instance: Instance, method: str) -> Solution:
    if method == "greedy_nearest_depot":
        result = solve_greedy_nearest_depot(instance)
        if result.solution is None:
            raise ValueError(str(result.metadata.get("error", "greedy failed")))
        return result.solution
    if method == "savings":
        return savings_routes_per_depot(instance)
    if method == "regret2":
        return regret_insertion(instance, regret_k=2)
    if method == "regret3":
        return regret_insertion(instance, regret_k=3)
    raise ValueError(f"Unknown constructive method: {method}")
