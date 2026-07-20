from __future__ import annotations

import random
import time
from dataclasses import replace

from smio_clrp.algorithms.alns.config import ALNSConfig
from smio_clrp.algorithms.alns.destroy import random_customer_removal
from smio_clrp.algorithms.alns.repair import regret2_repair
from smio_clrp.algorithms.base import Solver, SolverConfig, SolverResult
from smio_clrp.algorithms.common import clone_solution
from smio_clrp.algorithms.local_search.driver import improve_solution
from smio_clrp.algorithms.vns.config import VNSConfig
from smio_clrp.core.instance import Instance
from smio_clrp.core.solution import Solution
from smio_clrp.evaluation.cost import objective_cost
from smio_clrp.evaluation.validator import validate_solution


class VNSSolver(Solver):
    algorithm_name = "vns"

    def __init__(self, initial_solution: Solution, config: SolverConfig | None = None) -> None:
        super().__init__(config)
        self.initial_solution = initial_solution

    def solve(self, instance: Instance) -> SolverResult:
        start = time.perf_counter()
        try:
            config = _config_from_solver_config(self.config)
            validation = validate_solution(instance, self.initial_solution)
            if not validation.is_feasible:
                return SolverResult(None, None, False, 0.0, config.seed, self.algorithm_name, {"error": "; ".join(validation.errors)})

            rng = random.Random(config.seed)
            best = clone_solution(self.initial_solution)
            initial_cost = objective_cost(instance, best)
            best_cost = initial_cost
            neighborhood_index = 0
            iterations = 0
            improvements = 0
            failed_shakes = 0
            neighborhood_calls = [0] * len(config.shake_fractions)

            while iterations < config.max_iterations:
                if config.time_limit_seconds is not None and time.perf_counter() - start >= config.time_limit_seconds:
                    break
                iterations += 1
                neighborhood_calls[neighborhood_index] += 1
                fraction = config.shake_fractions[neighborhood_index]
                shake_config = replace(
                    ALNSConfig(seed=config.seed),
                    destroy_fraction_min=fraction,
                    destroy_fraction_max=fraction,
                )
                destroyed = random_customer_removal(instance, best, rng, shake_config)
                repaired = regret2_repair(
                    instance,
                    destroyed.partial_solution,
                    destroyed.removed_customer_ids,
                    rng,
                )
                if not repaired.success or repaired.solution is None:
                    failed_shakes += 1
                    neighborhood_index = (neighborhood_index + 1) % len(config.shake_fractions)
                    continue

                remaining = None
                if config.time_limit_seconds is not None:
                    remaining = max(0.0, config.time_limit_seconds - (time.perf_counter() - start))
                candidate = improve_solution(
                    instance,
                    repaired.solution,
                    operators=list(config.local_search_operators),
                    max_iterations=config.local_search_iterations,
                    time_limit_seconds=remaining,
                )
                candidate_validation = validate_solution(instance, candidate)
                if not candidate_validation.is_feasible:
                    failed_shakes += 1
                    neighborhood_index = (neighborhood_index + 1) % len(config.shake_fractions)
                    continue
                candidate_cost = objective_cost(instance, candidate)
                if candidate_cost + 1e-9 < best_cost:
                    best = clone_solution(candidate)
                    best_cost = candidate_cost
                    improvements += 1
                    neighborhood_index = 0
                else:
                    neighborhood_index = (neighborhood_index + 1) % len(config.shake_fractions)

            return SolverResult(
                best,
                best_cost,
                True,
                time.perf_counter() - start,
                config.seed,
                self.algorithm_name,
                {
                    "initial_cost": initial_cost,
                    "best_cost": best_cost,
                    "iterations": iterations,
                    "improvements": improvements,
                    "failed_shakes": failed_shakes,
                    "shake_fractions": list(config.shake_fractions),
                    "neighborhood_calls": neighborhood_calls,
                },
            )
        except ValueError as exc:
            return SolverResult(None, None, False, time.perf_counter() - start, self.config.seed, self.algorithm_name, {"error": str(exc)})


def _config_from_solver_config(config: SolverConfig) -> VNSConfig:
    raw_fractions = config.metadata.get("vns_shake_fractions", (0.05, 0.10, 0.20, 0.30))
    if isinstance(raw_fractions, str):
        fractions = tuple(float(item.strip()) for item in raw_fractions.split(",") if item.strip())
    else:
        fractions = tuple(float(item) for item in raw_fractions)
    raw_operators = config.metadata.get(
        "vns_operators",
        ("two_opt_star", "route_depot_reassignment", "two_opt", "route_reinsertion"),
    )
    operators = tuple(item.strip() for item in raw_operators.split(",") if item.strip()) if isinstance(raw_operators, str) else tuple(raw_operators)
    return VNSConfig(
        seed=config.seed,
        max_iterations=int(config.metadata.get("vns_iterations", 50)),
        time_limit_seconds=config.time_limit_seconds,
        local_search_iterations=int(config.metadata.get("vns_local_search_iterations", 10)),
        shake_fractions=fractions,
        local_search_operators=operators,
    )
