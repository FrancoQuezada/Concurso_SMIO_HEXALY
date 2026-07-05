import random

from smio_clrp.algorithms.alns.acceptance import SimulatedAnnealingAcceptance, accept_if_better
from smio_clrp.algorithms.alns.alns_solver import ALNSSolver
from smio_clrp.algorithms.alns.config import ALNSConfig
from smio_clrp.algorithms.alns.destroy import (
    depot_removal,
    random_customer_removal,
    route_removal,
    shaw_related_removal,
    worst_customer_removal,
)
from smio_clrp.algorithms.alns.operator_selection import AdaptiveRouletteWheel
from smio_clrp.algorithms.alns.repair import greedy_repair, noise_repair, regret2_repair, regret3_repair
from smio_clrp.algorithms.base import SolverConfig
from smio_clrp.algorithms.local_search.solver import ConstructiveLocalSearchSolver
from smio_clrp.evaluation.cost import objective_cost
from smio_clrp.evaluation.validator import validate_solution
from smio_clrp.io.instance_reader import read_instance


def _alns_solver(seed=1, max_iterations=80):
    return ALNSSolver(
        SolverConfig(
            seed=seed,
            metadata={
                "num_starts": 8,
                "max_iterations": max_iterations,
                "destroy_fraction_min": 0.2,
                "destroy_fraction_max": 0.4,
                "initial_temperature": 5.0,
                "cooling_rate": 0.99,
            },
        )
    )


def _base_solution(path="data/samples/tiny_coords.txt"):
    instance = read_instance(path)
    result = ConstructiveLocalSearchSolver(
        SolverConfig(seed=1, metadata={"num_starts": 8, "max_iterations": 20})
    ).solve(instance)
    assert result.solution is not None
    return instance, result.solution


def _flatten(solution):
    return [customer_id for route in solution.routes for customer_id in route.customer_ids]


def _assert_partial_has_no_duplicates(result):
    customers = _flatten(result.partial_solution)
    assert len(customers) == len(set(customers))
    assert set(result.removed_customer_ids).isdisjoint(customers)


def test_alns_returns_feasible_solution_on_tiny_coords():
    instance = read_instance("data/samples/tiny_coords.txt")

    result = _alns_solver().solve(instance)

    assert result.solution is not None
    assert result.feasible
    assert validate_solution(instance, result.solution).is_feasible


def test_alns_returns_feasible_solution_on_tiny_full_matrix():
    instance = read_instance("data/samples/tiny_full_matrix.txt")

    result = _alns_solver().solve(instance)

    assert result.solution is not None
    assert result.feasible
    assert validate_solution(instance, result.solution).is_feasible


def test_alns_is_deterministic_for_fixed_seed():
    instance = read_instance("data/samples/tiny_coords.txt")

    first = _alns_solver(seed=5).solve(instance)
    second = _alns_solver(seed=5).solve(instance)

    assert first.cost == second.cost
    assert first.solution == second.solution


def test_alns_supports_asymmetric_full_matrix_without_symmetry():
    instance = read_instance("data/samples/tiny_full_matrix.txt")

    result = _alns_solver(seed=3).solve(instance)

    assert result.solution is not None
    assert validate_solution(instance, result.solution).is_feasible


def test_alns_best_cost_is_not_worse_than_constructive_ls_on_coords():
    instance = read_instance("data/samples/tiny_coords.txt")
    initial = ConstructiveLocalSearchSolver(
        SolverConfig(seed=1, metadata={"num_starts": 8, "max_iterations": 20})
    ).solve(instance)
    result = _alns_solver(seed=1, max_iterations=100).solve(instance)

    assert initial.cost is not None
    assert result.cost is not None
    assert result.cost <= initial.cost


def test_random_removal_removes_customers_and_keeps_no_duplicates():
    instance, solution = _base_solution()

    result = random_customer_removal(instance, solution, random.Random(1), ALNSConfig(seed=1))

    assert result.removed_customer_ids
    _assert_partial_has_no_duplicates(result)


def test_worst_removal_removes_customers_and_keeps_no_duplicates():
    instance, solution = _base_solution()

    result = worst_customer_removal(instance, solution, random.Random(1), ALNSConfig(seed=1))

    assert result.removed_customer_ids
    _assert_partial_has_no_duplicates(result)


def test_shaw_removal_removes_customers_and_keeps_no_duplicates():
    instance, solution = _base_solution("data/samples/tiny_full_matrix.txt")

    result = shaw_related_removal(instance, solution, random.Random(1), ALNSConfig(seed=1))

    assert result.removed_customer_ids
    _assert_partial_has_no_duplicates(result)


def test_route_removal_removes_complete_routes():
    instance, solution = _base_solution()

    result = route_removal(instance, solution, random.Random(1), ALNSConfig(seed=1))

    assert result.removed_customer_ids
    assert len(result.partial_solution.routes) < len(solution.routes)
    _assert_partial_has_no_duplicates(result)


def test_depot_removal_removes_customers_from_one_depot():
    instance, solution = _base_solution()

    result = depot_removal(instance, solution, random.Random(1), ALNSConfig(seed=1))

    assert result.removed_customer_ids
    assert len({route.depot_id for route in result.partial_solution.routes}) <= len(
        {route.depot_id for route in solution.routes}
    )
    _assert_partial_has_no_duplicates(result)


def test_greedy_repair_reconstructs_feasible_solution_after_random_removal():
    instance, solution = _base_solution()
    destroyed = random_customer_removal(instance, solution, random.Random(2), ALNSConfig(seed=1))

    repaired = greedy_repair(instance, destroyed.partial_solution, destroyed.removed_customer_ids, random.Random(3))

    assert repaired.solution is not None
    assert validate_solution(instance, repaired.solution).is_feasible


def test_regret_repairs_reconstruct_feasible_solution_after_random_removal():
    instance, solution = _base_solution()
    destroyed = random_customer_removal(instance, solution, random.Random(2), ALNSConfig(seed=1))

    repaired2 = regret2_repair(instance, destroyed.partial_solution, destroyed.removed_customer_ids, random.Random(3))
    repaired3 = regret3_repair(instance, destroyed.partial_solution, destroyed.removed_customer_ids, random.Random(3))

    assert repaired2.solution is not None
    assert repaired3.solution is not None
    assert validate_solution(instance, repaired2.solution).is_feasible
    assert validate_solution(instance, repaired3.solution).is_feasible


def test_noise_repair_is_deterministic_for_fixed_seed():
    instance, solution = _base_solution("data/samples/tiny_full_matrix.txt")
    destroyed = random_customer_removal(instance, solution, random.Random(2), ALNSConfig(seed=1))

    first = noise_repair(instance, destroyed.partial_solution, destroyed.removed_customer_ids, random.Random(9))
    second = noise_repair(instance, destroyed.partial_solution, destroyed.removed_customer_ids, random.Random(9))

    assert first.solution == second.solution


def test_accept_if_better_rejects_worse_moves():
    assert not accept_if_better(candidate_cost=11.0, current_cost=10.0)


def test_simulated_annealing_can_accept_worse_move_and_cools():
    acceptance = SimulatedAnnealingAcceptance(temperature=100.0, cooling_rate=0.5)

    assert acceptance.accept(candidate_cost=11.0, current_cost=10.0, rng=random.Random(1))
    assert acceptance.cool() == 50.0


def test_operator_weights_update_after_rewards():
    selector = AdaptiveRouletteWheel(["a", "b"])

    selector.reward("a", score=8.0, accepted=True, improved=True, best=True)
    selector.segment_counts["a"] = 1
    selector.update_weights()

    assert selector.weights["a"] > selector.weights["b"]


def test_roulette_selection_is_deterministic_for_fixed_seed():
    first = AdaptiveRouletteWheel(["a", "b", "c"])
    second = AdaptiveRouletteWheel(["a", "b", "c"])

    first_draws = [first.select(random.Random(4)) for _ in range(3)]
    second_draws = [second.select(random.Random(4)) for _ in range(3)]

    assert first_draws == second_draws
