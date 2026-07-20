from __future__ import annotations

import random
import time

from smio_clrp.algorithms.alns.acceptance import (
    RecordToRecordAcceptance,
    SimulatedAnnealingAcceptance,
    accept_if_better,
)
from smio_clrp.algorithms.alns.config import ALNSConfig
from smio_clrp.algorithms.alns.destroy import DESTROY_OPERATORS
from smio_clrp.algorithms.alns.operator_selection import AdaptiveRouletteWheel
from smio_clrp.algorithms.alns.operators import ALNSState
from smio_clrp.algorithms.alns.repair import REPAIR_OPERATORS
from smio_clrp.algorithms.base import Solver, SolverConfig, SolverResult
from smio_clrp.algorithms.common import clone_solution
from smio_clrp.algorithms.local_search.driver import improve_solution
from smio_clrp.algorithms.local_search.solver import ConstructiveLocalSearchSolver
from smio_clrp.core.instance import Instance
from smio_clrp.core.solution import Solution
from smio_clrp.evaluation.cost import objective_cost
from smio_clrp.evaluation.validator import validate_solution


class ALNSSolver(Solver):
    algorithm_name = "alns"

    def __init__(self, config: SolverConfig | None = None, initial_solution: Solution | None = None) -> None:
        super().__init__(config)
        self.initial_solution = initial_solution

    def solve(self, instance: Instance) -> SolverResult:
        start = time.perf_counter()
        try:
            config = _config_from_solver_config(self.config)
            rng = random.Random(config.seed)
            if self.initial_solution is None:
                initial = _build_initial_solution(instance, config)
            else:
                validation = validate_solution(instance, self.initial_solution)
                initial = SolverResult(
                    solution=clone_solution(self.initial_solution) if validation.is_feasible else None,
                    cost=objective_cost(instance, self.initial_solution) if validation.is_feasible else None,
                    feasible=validation.is_feasible,
                    runtime_seconds=0.0,
                    seed=config.seed,
                    algorithm_name="provided_initial_solution",
                    metadata={"validation_errors": validation.errors},
                )
            if initial.solution is None:
                return SolverResult(
                    None,
                    None,
                    False,
                    time.perf_counter() - start,
                    config.seed,
                    self.algorithm_name,
                    {"error": "Could not build initial solution", "initial_metadata": initial.metadata},
                )

            current = clone_solution(initial.solution)
            best = clone_solution(initial.solution)
            initial_cost = objective_cost(instance, initial.solution)
            state = ALNSState(
                current_solution=current,
                current_cost=initial_cost,
                best_solution=best,
                best_cost=initial_cost,
                initial_cost=initial_cost,
                temperature=config.initial_temperature,
            )
            destroy_selector = AdaptiveRouletteWheel(list(DESTROY_OPERATORS))
            repair_names = _repair_names(config)
            repair_selector = AdaptiveRouletteWheel(repair_names)
            annealing = SimulatedAnnealingAcceptance(config.initial_temperature, config.cooling_rate)
            record_acceptance = RecordToRecordAcceptance()
            vns_kicks = 0
            vns_improvements = 0
            vns_gain = 0.0
            vns_threshold = int(self.config.metadata.get("vns_stagnation_threshold", 50))
            vns_on_stagnation = bool(self.config.metadata.get("vns_on_stagnation", False))
            if vns_threshold <= 0:
                raise ValueError("vns_stagnation_threshold must be positive")

            for iteration in range(1, config.max_iterations + 1):
                if config.time_limit_seconds is not None and time.perf_counter() - start >= config.time_limit_seconds:
                    break
                if state.no_improve_iterations >= config.max_no_improve:
                    break
                if (
                    vns_on_stagnation
                    and state.no_improve_iterations > 0
                    and state.no_improve_iterations % vns_threshold == 0
                ):
                    previous_cost = state.best_cost
                    intensified = _intensify_with_vns(instance, state.best_solution, self.config, start)
                    vns_kicks += 1
                    if intensified.solution is not None and intensified.cost is not None and intensified.cost + 1e-9 < state.best_cost:
                        state.best_solution = clone_solution(intensified.solution)
                        state.best_cost = intensified.cost
                        state.current_solution = clone_solution(intensified.solution)
                        state.current_cost = intensified.cost
                        state.no_improve_iterations = 0
                        state.best_improvements += 1
                        vns_improvements += 1
                        vns_gain += previous_cost - intensified.cost

                state.iterations = iteration
                destroy_name = destroy_selector.select(rng)
                repair_name = repair_selector.select(rng)

                destroy_start = time.perf_counter()
                destroy_result = DESTROY_OPERATORS[destroy_name](instance, state.current_solution, rng, config)
                destroy_selector.record_call(destroy_name, time.perf_counter() - destroy_start)

                repair_start = time.perf_counter()
                repair_result = REPAIR_OPERATORS[repair_name](
                    instance,
                    destroy_result.partial_solution,
                    destroy_result.removed_customer_ids,
                    rng,
                )
                repair_selector.record_call(repair_name, time.perf_counter() - repair_start)

                if not repair_result.success or repair_result.solution is None:
                    state.failed_repairs += 1
                    destroy_selector.reward(destroy_name, 0.0, False, False, False)
                    repair_selector.reward(repair_name, 0.0, False, False, False)
                    _maybe_update_weights(iteration, config, destroy_selector, repair_selector)
                    annealing.cool()
                    state.temperature = annealing.temperature
                    continue

                candidate = repair_result.solution
                validation = validate_solution(instance, candidate)
                if not validation.is_feasible:
                    state.failed_repairs += 1
                    destroy_selector.reward(destroy_name, 0.0, False, False, False)
                    repair_selector.reward(repair_name, 0.0, False, False, False)
                    _maybe_update_weights(iteration, config, destroy_selector, repair_selector)
                    annealing.cool()
                    state.temperature = annealing.temperature
                    continue

                candidate_cost = objective_cost(instance, candidate)
                accepted = _accept_candidate(
                    config,
                    candidate_cost,
                    state.current_cost,
                    state.best_cost,
                    rng,
                    annealing,
                    record_acceptance,
                )
                improved = candidate_cost + 1e-9 < state.current_cost
                new_best = candidate_cost + 1e-9 < state.best_cost

                if accepted:
                    state.accepted_moves += 1
                    if improved:
                        state.improving_moves += 1
                    state.current_solution = clone_solution(candidate)
                    state.current_cost = candidate_cost
                    if new_best:
                        candidate = _maybe_local_search_new_best(instance, candidate, config)
                        candidate_cost = objective_cost(instance, candidate)
                        if candidate_cost + 1e-9 < state.best_cost:
                            state.best_improvements += 1
                            state.best_solution = clone_solution(candidate)
                            state.best_cost = candidate_cost
                            state.current_solution = clone_solution(candidate)
                            state.current_cost = candidate_cost
                            state.no_improve_iterations = 0
                        else:
                            state.no_improve_iterations += 1
                    else:
                        state.no_improve_iterations += 1

                    if _should_local_search_on_frequency(config, state.accepted_moves):
                        improved_solution = improve_solution(
                            instance,
                            state.current_solution,
                            max_iterations=config.local_search_iterations,
                        )
                        improved_cost = objective_cost(instance, improved_solution)
                        if improved_cost + 1e-9 < state.current_cost:
                            state.current_solution = clone_solution(improved_solution)
                            state.current_cost = improved_cost
                            if improved_cost + 1e-9 < state.best_cost:
                                state.best_solution = clone_solution(improved_solution)
                                state.best_cost = improved_cost
                                state.best_improvements += 1
                                new_best = True
                else:
                    state.no_improve_iterations += 1

                score = _reward_score(accepted, improved, new_best)
                destroy_selector.reward(destroy_name, score, accepted, improved, new_best)
                repair_selector.reward(repair_name, score, accepted, improved, new_best)
                _maybe_update_weights(iteration, config, destroy_selector, repair_selector)
                annealing.cool()
                state.temperature = annealing.temperature

            validation = validate_solution(instance, state.best_solution)
            if not validation.is_feasible:
                return SolverResult(
                    None,
                    None,
                    False,
                    time.perf_counter() - start,
                    config.seed,
                    self.algorithm_name,
                    {"error": "; ".join(validation.errors)},
                )
            return SolverResult(
                solution=state.best_solution,
                cost=state.best_cost,
                feasible=True,
                runtime_seconds=time.perf_counter() - start,
                seed=config.seed,
                algorithm_name=self.algorithm_name,
                metadata=_metadata(
                    state,
                    destroy_selector,
                    repair_selector,
                    time.perf_counter() - start,
                    vns_kicks,
                    vns_improvements,
                    vns_gain,
                ),
            )
        except ValueError as exc:
            return SolverResult(
                None,
                None,
                False,
                time.perf_counter() - start,
                self.config.seed,
                self.algorithm_name,
                {"error": str(exc)},
            )


def _config_from_solver_config(config: SolverConfig) -> ALNSConfig:
    metadata = config.metadata
    return ALNSConfig(
        seed=config.seed,
        max_iterations=int(metadata.get("max_iterations", 500)),
        time_limit_seconds=config.time_limit_seconds,
        num_starts=int(metadata.get("num_starts", 20)),
        destroy_fraction_min=float(metadata.get("destroy_fraction_min", 0.15)),
        destroy_fraction_max=float(metadata.get("destroy_fraction_max", 0.35)),
        initial_temperature=float(metadata.get("initial_temperature", 10.0)),
        cooling_rate=float(metadata.get("cooling_rate", 0.995)),
        local_search_frequency=str(metadata.get("local_search_frequency", "best")),
        max_no_improve=int(metadata.get("max_no_improve", 150)),
        repair_method=str(metadata.get("repair_method", "mixed")),
        acceptance_method=str(metadata.get("acceptance_method", "simulated_annealing")),
        verbose=bool(metadata.get("verbose", False)),
    )


def _build_initial_solution(instance: Instance, config: ALNSConfig) -> SolverResult:
    solver = ConstructiveLocalSearchSolver(
        SolverConfig(
            seed=config.seed,
            time_limit_seconds=config.time_limit_seconds,
            metadata={"num_starts": config.num_starts, "max_iterations": 25},
        )
    )
    return solver.solve(instance)


def _initial_result(instance: Instance, config: ALNSConfig, initial_solution: Solution | None) -> SolverResult:
    if initial_solution is None:
        return _build_initial_solution(instance, config)
    validation = validate_solution(instance, initial_solution)
    if not validation.is_feasible:
        return SolverResult(None, None, False, 0.0, config.seed, "initial", {"error": "; ".join(validation.errors)})
    return SolverResult(initial_solution, validation.cost, True, 0.0, config.seed, "initial", {"source": "provided"})


def _repair_names(config: ALNSConfig) -> list[str]:
    if config.repair_method == "mixed":
        return list(REPAIR_OPERATORS)
    if config.repair_method not in REPAIR_OPERATORS:
        raise ValueError(f"Unknown repair_method: {config.repair_method}")
    return [config.repair_method]


def _accept_candidate(
    config: ALNSConfig,
    candidate_cost: float,
    current_cost: float,
    best_cost: float,
    rng: random.Random,
    annealing: SimulatedAnnealingAcceptance,
    record_acceptance: RecordToRecordAcceptance,
) -> bool:
    if config.acceptance_method == "accept_if_better":
        return accept_if_better(candidate_cost, current_cost)
    if config.acceptance_method == "simulated_annealing":
        return annealing.accept(candidate_cost, current_cost, rng)
    if config.acceptance_method == "record_to_record":
        return record_acceptance.accept(candidate_cost, best_cost)
    raise ValueError(f"Unknown acceptance_method: {config.acceptance_method}")


def _maybe_local_search_new_best(instance: Instance, candidate: Solution, config: ALNSConfig) -> Solution:
    if config.local_search_frequency != "best":
        return candidate
    return improve_solution(instance, candidate, max_iterations=config.local_search_iterations)


def _should_local_search_on_frequency(config: ALNSConfig, accepted_moves: int) -> bool:
    if config.local_search_frequency in {"best", "never", "0"}:
        return False
    try:
        frequency = int(config.local_search_frequency)
    except ValueError:
        return False
    return frequency > 0 and accepted_moves % frequency == 0


def _reward_score(accepted: bool, improved: bool, best: bool) -> float:
    if best:
        return 8.0
    if improved:
        return 4.0
    if accepted:
        return 1.0
    return 0.0


def _maybe_update_weights(
    iteration: int,
    config: ALNSConfig,
    destroy_selector: AdaptiveRouletteWheel,
    repair_selector: AdaptiveRouletteWheel,
) -> None:
    if iteration % config.segment_length == 0:
        destroy_selector.update_weights()
        repair_selector.update_weights()


def _metadata(
    state: ALNSState,
    destroy_selector: AdaptiveRouletteWheel,
    repair_selector: AdaptiveRouletteWheel,
    runtime_seconds: float,
    vns_kicks: int = 0,
    vns_improvements: int = 0,
    vns_gain: float = 0.0,
) -> dict[str, object]:
    return {
        "initial_cost": state.initial_cost,
        "best_cost": state.best_cost,
        "iterations": state.iterations,
        "accepted_moves": state.accepted_moves,
        "improving_moves": state.improving_moves,
        "best_improvements": state.best_improvements,
        "failed_repairs": state.failed_repairs,
        "destroy_operator_stats": destroy_selector.stats_as_dict(),
        "repair_operator_stats": repair_selector.stats_as_dict(),
        "runtime_seconds": runtime_seconds,
        "vns_stagnation_kicks": vns_kicks,
        "vns_stagnation_improvements": vns_improvements,
        "vns_stagnation_gain": vns_gain,
    }


def _intensify_with_vns(
    instance: Instance,
    solution: Solution,
    solver_config: SolverConfig,
    alns_start: float,
) -> SolverResult:
    # Local import avoids coupling the standalone ALNS module to the hybrid at import time.
    from smio_clrp.algorithms.vns.vns_solver import VNSSolver

    remaining = None
    if solver_config.time_limit_seconds is not None:
        remaining = max(0.0, solver_config.time_limit_seconds - (time.perf_counter() - alns_start))
        kick_limit = float(solver_config.metadata.get("vns_stagnation_time_limit", 2.0))
        remaining = min(remaining, kick_limit)
    metadata = dict(solver_config.metadata)
    metadata["vns_iterations"] = int(metadata.get("vns_stagnation_iterations", 3))
    return VNSSolver(
        solution,
        SolverConfig(seed=solver_config.seed, time_limit_seconds=remaining, metadata=metadata),
    ).solve(instance)
