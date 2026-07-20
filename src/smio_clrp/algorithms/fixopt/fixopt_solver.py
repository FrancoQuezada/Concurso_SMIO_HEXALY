from __future__ import annotations

import random
import time
from collections import Counter

from smio_clrp.algorithms.alns import ALNSSolver
from smio_clrp.algorithms.base import Solver, SolverConfig, SolverResult
from smio_clrp.algorithms.common import clone_solution
from smio_clrp.algorithms.fixopt.backend import BackendUnavailable, FixOptBackend, FixOptResult
from smio_clrp.algorithms.fixopt.config import FixOptConfig
from smio_clrp.algorithms.fixopt.heuristic_backend import HeuristicFixOptBackend
from smio_clrp.algorithms.fixopt.mip_backend import MIPFixOptBackend
from smio_clrp.algorithms.fixopt.neighborhoods import DEFAULT_NEIGHBORHOODS, build_neighborhood
from smio_clrp.algorithms.local_search.solver import ConstructiveLocalSearchSolver
from smio_clrp.core.instance import Instance
from smio_clrp.core.solution import Solution
from smio_clrp.evaluation.cost import objective_cost
from smio_clrp.evaluation.validator import validate_solution


class FixOptimizeSolver(Solver):
    algorithm_name = "fixopt"

    def __init__(self, initial_solution: Solution | None = None, config: SolverConfig | None = None) -> None:
        super().__init__(config)
        self.initial_solution = initial_solution

    def solve(self, instance: Instance) -> SolverResult:
        start = time.perf_counter()
        try:
            config = _config_from_solver_config(self.config)
            initial = self.initial_solution or _build_initial_solution(instance, config)
            validation = validate_solution(instance, initial)
            if not validation.is_feasible:
                return SolverResult(None, None, False, time.perf_counter() - start, config.seed, self.algorithm_name, {"error": "; ".join(validation.errors)})
            backend, backend_used = _select_backend(config)
            result = _run_fixopt(instance, initial, config, backend, backend_used, start)
            result.algorithm_name = self.algorithm_name
            return result
        except ValueError as exc:
            return SolverResult(None, None, False, time.perf_counter() - start, self.config.seed, self.algorithm_name, {"error": str(exc)})


class HybridALNSFixOptSolver(Solver):
    algorithm_name = "hybrid"

    def solve(self, instance: Instance) -> SolverResult:
        start = time.perf_counter()
        alns_budget, configured_fixopt_budget = _hybrid_time_budgets(self.config)
        alns_config = SolverConfig(
            seed=self.config.seed,
            time_limit_seconds=alns_budget,
            metadata=dict(self.config.metadata),
        )
        alns = ALNSSolver(alns_config).solve(instance)
        if alns.solution is None:
            alns.algorithm_name = self.algorithm_name
            return alns
        elapsed = time.perf_counter() - start
        remaining_budget = None
        if self.config.time_limit_seconds is not None:
            remaining_budget = max(0.0, self.config.time_limit_seconds - elapsed)
        fixopt_budget = configured_fixopt_budget
        if remaining_budget is not None:
            fixopt_budget = remaining_budget if fixopt_budget is None else min(fixopt_budget, remaining_budget)
        fix_config = SolverConfig(
            seed=self.config.seed,
            time_limit_seconds=fixopt_budget,
            metadata={**self.config.metadata, "fixopt_time_limit": fixopt_budget},
        )
        fix_start = time.perf_counter()
        fix = FixOptimizeSolver(alns.solution, fix_config).solve(instance)
        if fix.solution is None or (alns.cost is not None and fix.cost is not None and fix.cost > alns.cost):
            final_solution = alns.solution
            final_cost = alns.cost
        else:
            final_solution = fix.solution
            final_cost = fix.cost
        return SolverResult(
            solution=final_solution,
            cost=final_cost,
            feasible=final_solution is not None,
            runtime_seconds=time.perf_counter() - start,
            seed=self.config.seed,
            algorithm_name=self.algorithm_name,
            metadata={
                "alns_initial_cost": alns.metadata.get("initial_cost"),
                "alns_best_cost": alns.cost,
                "fixopt_initial_cost": fix.metadata.get("initial_cost") if fix.metadata else alns.cost,
                "final_cost": final_cost,
                "alns_runtime_seconds": alns.runtime_seconds,
                "fixopt_runtime_seconds": time.perf_counter() - fix_start,
                "total_runtime_seconds": time.perf_counter() - start,
                "alns_time_budget_seconds": alns_budget,
                "fixopt_time_budget_seconds": fixopt_budget,
            },
        )


def _hybrid_time_budgets(config: SolverConfig) -> tuple[float | None, float | None]:
    """Split the global hybrid budget without allowing either phase to add extra time."""
    explicit_fixopt_budget = config.metadata.get("fixopt_time_limit")
    if explicit_fixopt_budget is not None:
        explicit_fixopt_budget = float(explicit_fixopt_budget)
        if explicit_fixopt_budget < 0:
            raise ValueError("fixopt_time_limit must be non-negative")
    if config.time_limit_seconds is None:
        return None, explicit_fixopt_budget

    fraction = float(config.metadata.get("hybrid_alns_fraction", 0.75))
    if not 0 < fraction < 1:
        raise ValueError("hybrid_alns_fraction must be between 0 and 1")
    alns_budget = config.time_limit_seconds * fraction
    reserved_fixopt_budget = config.time_limit_seconds - alns_budget
    if explicit_fixopt_budget is not None:
        reserved_fixopt_budget = min(reserved_fixopt_budget, explicit_fixopt_budget)
    return alns_budget, reserved_fixopt_budget


def _run_fixopt(
    instance: Instance,
    initial: Solution,
    config: FixOptConfig,
    backend: FixOptBackend,
    backend_used: str,
    start: float,
) -> SolverResult:
    rng = random.Random(config.seed)
    current = clone_solution(initial)
    best = clone_solution(initial)
    current_cost = objective_cost(instance, current)
    best_cost = current_cost
    initial_cost = current_cost
    neighborhood_types = config.neighborhood_types or DEFAULT_NEIGHBORHOODS
    improved_subproblems = 0
    failed_subproblems = 0
    skipped_subproblems = 0
    neighborhood_stats: Counter[str] = Counter()
    iterations = 0

    for iteration in range(1, config.max_iterations + 1):
        if config.time_limit_seconds is not None and time.perf_counter() - start >= config.time_limit_seconds:
            break
        iterations = iteration
        neighborhood_type = neighborhood_types[(iteration - 1) % len(neighborhood_types)]
        neighborhood = build_neighborhood(
            instance,
            current,
            neighborhood_type,
            rng,
            config.max_customers_per_subproblem,
            config.max_routes_per_subproblem,
        )
        neighborhood_stats[neighborhood_type] += 1
        if not neighborhood.released_customer_ids:
            skipped_subproblems += 1
            continue
        result = backend.reoptimize(instance, current, neighborhood, config)
        if not result.success or result.solution is None:
            failed_subproblems += 1
            continue
        validation = validate_solution(instance, result.solution)
        if not validation.is_feasible:
            failed_subproblems += 1
            continue
        candidate_cost = objective_cost(instance, result.solution)
        if config.accept_worsening or candidate_cost <= current_cost + 1e-9:
            current = clone_solution(result.solution)
            current_cost = candidate_cost
            if candidate_cost + 1e-9 < best_cost:
                best = clone_solution(result.solution)
                best_cost = candidate_cost
                improved_subproblems += 1

    final_validation = validate_solution(instance, best)
    if not final_validation.is_feasible:
        return SolverResult(None, None, False, time.perf_counter() - start, config.seed, "fixopt", {"error": "; ".join(final_validation.errors)})
    return SolverResult(
        solution=best,
        cost=best_cost,
        feasible=True,
        runtime_seconds=time.perf_counter() - start,
        seed=config.seed,
        algorithm_name="fixopt",
        metadata={
            "initial_cost": initial_cost,
            "best_cost": best_cost,
            "iterations": iterations,
            "improved_subproblems": improved_subproblems,
            "failed_subproblems": failed_subproblems,
            "skipped_subproblems": skipped_subproblems,
            "backend_used": backend_used,
            "neighborhood_stats": dict(neighborhood_stats),
            "runtime_seconds": time.perf_counter() - start,
        },
    )


def _build_initial_solution(instance: Instance, config: FixOptConfig) -> Solution:
    result = ConstructiveLocalSearchSolver(
        SolverConfig(
            seed=config.seed,
            time_limit_seconds=config.time_limit_seconds,
            metadata={"num_starts": 10, "max_iterations": 25},
        )
    ).solve(instance)
    if result.solution is None:
        raise ValueError("Could not build initial solution for fixopt")
    return result.solution


def _select_backend(config: FixOptConfig) -> tuple[FixOptBackend, str]:
    if config.backend == "heuristic":
        return HeuristicFixOptBackend(seed=config.seed), "heuristic"
    if config.backend == "mip":
        return MIPFixOptBackend(seed=config.seed), "mip"
    try:
        return MIPFixOptBackend(seed=config.seed), "mip"
    except BackendUnavailable:
        return HeuristicFixOptBackend(seed=config.seed), "heuristic"


def _config_from_solver_config(config: SolverConfig) -> FixOptConfig:
    metadata = config.metadata
    raw_neighborhoods = metadata.get("neighborhood_types")
    if isinstance(raw_neighborhoods, str):
        neighborhood_types = [item.strip() for item in raw_neighborhoods.split(",") if item.strip()]
    else:
        neighborhood_types = raw_neighborhoods
    return FixOptConfig(
        seed=config.seed,
        max_iterations=int(metadata.get("fixopt_iterations", metadata.get("max_iterations", 50))),
        time_limit_seconds=metadata.get("fixopt_time_limit", config.time_limit_seconds),
        neighborhood_types=neighborhood_types,
        max_customers_per_subproblem=int(metadata.get("max_customers_per_subproblem", 12)),
        max_routes_per_subproblem=int(metadata.get("max_routes_per_subproblem", 3)),
        backend=str(metadata.get("fixopt_backend", metadata.get("backend", "auto"))),
        mip_time_limit_seconds=metadata.get("mip_time_limit"),
        accept_worsening=bool(metadata.get("accept_worsening", False)),
        local_search_after_subproblem=bool(metadata.get("local_search_after_subproblem", True)),
        verbose=bool(metadata.get("verbose", False)),
    )
