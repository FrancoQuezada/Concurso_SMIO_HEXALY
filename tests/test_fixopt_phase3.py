import random

import pytest

from smio_clrp.algorithms.base import SolverConfig
from smio_clrp.algorithms.fixopt.backend import BackendUnavailable
from smio_clrp.algorithms.fixopt.config import FixOptConfig
from smio_clrp.algorithms.fixopt.fixopt_solver import (
    FixOptimizeSolver,
    HybridALNSFixOptSolver,
    _hybrid_time_budgets,
)
from smio_clrp.algorithms.fixopt.heuristic_backend import HeuristicFixOptBackend
from smio_clrp.algorithms.fixopt.mip_backend import MIPFixOptBackend, gurobi_available
from smio_clrp.algorithms.fixopt.neighborhoods import (
    boundary_customer_neighborhood,
    depot_neighborhood,
    expensive_customer_neighborhood,
    route_neighborhood,
    route_pair_neighborhood,
)
from smio_clrp.algorithms.local_search.solver import ConstructiveLocalSearchSolver
from smio_clrp.evaluation.cost import objective_cost
from smio_clrp.evaluation.validator import validate_solution
from smio_clrp.io.instance_reader import read_instance


def _initial_solution(path="data/samples/tiny_coords.txt"):
    instance = read_instance(path)
    result = ConstructiveLocalSearchSolver(
        SolverConfig(seed=1, metadata={"num_starts": 8, "max_iterations": 20})
    ).solve(instance)
    assert result.solution is not None
    return instance, result.solution


def _config(**overrides):
    values = {"seed": 1, "max_iterations": 20, "backend": "heuristic"}
    values.update(overrides)
    return FixOptConfig(**values)


def _assert_neighborhood_valid(solution, neighborhood):
    assert neighborhood.released_customer_ids
    assert len(neighborhood.released_customer_ids) == len(set(neighborhood.released_customer_ids))
    fixed_customers = [customer_id for route in neighborhood.fixed_routes for customer_id in route.customer_ids]
    assert set(neighborhood.released_customer_ids).isdisjoint(fixed_customers)
    assert len(fixed_customers) == len(set(fixed_customers))


def test_depot_neighborhood_releases_valid_customers_and_keeps_no_duplicates():
    instance, solution = _initial_solution()

    neighborhood = depot_neighborhood(instance, solution, random.Random(1), max_customers=8)

    _assert_neighborhood_valid(solution, neighborhood)
    assert neighborhood.candidate_depot_ids


def test_route_neighborhood_releases_route_customers():
    instance, solution = _initial_solution()

    neighborhood = route_neighborhood(instance, solution, random.Random(1), max_customers=8, max_routes=1)

    _assert_neighborhood_valid(solution, neighborhood)
    assert neighborhood.metadata["type"] == "route"


def test_boundary_customer_neighborhood_returns_valid_candidate_depots():
    instance, solution = _initial_solution()

    neighborhood = boundary_customer_neighborhood(instance, solution, random.Random(1), max_customers=3)

    _assert_neighborhood_valid(solution, neighborhood)
    assert set(neighborhood.candidate_depot_ids).issubset(set(instance.depots_by_id))


def test_expensive_customer_neighborhood_returns_valid_customers():
    instance, solution = _initial_solution()

    neighborhood = expensive_customer_neighborhood(instance, solution, max_customers=3)

    _assert_neighborhood_valid(solution, neighborhood)


def test_route_pair_neighborhood_works_with_routes_from_different_depots():
    instance, solution = _initial_solution()

    neighborhood = route_pair_neighborhood(instance, solution, random.Random(2), max_customers=10)

    _assert_neighborhood_valid(solution, neighborhood)
    assert len(neighborhood.candidate_depot_ids) >= 1


def test_heuristic_backend_reconstructs_feasible_solution_after_depot_neighborhood():
    instance, solution = _initial_solution()
    neighborhood = depot_neighborhood(instance, solution, random.Random(1), max_customers=8)

    result = HeuristicFixOptBackend(seed=1).reoptimize(instance, solution, neighborhood, _config())

    assert result.solution is not None
    assert validate_solution(instance, result.solution).is_feasible


def test_heuristic_backend_reconstructs_feasible_solution_after_route_neighborhood():
    instance, solution = _initial_solution()
    neighborhood = route_neighborhood(instance, solution, random.Random(1), max_customers=8, max_routes=1)

    result = HeuristicFixOptBackend(seed=1).reoptimize(instance, solution, neighborhood, _config())

    assert result.solution is not None
    assert validate_solution(instance, result.solution).is_feasible


def test_heuristic_backend_supports_asymmetric_full_matrix():
    instance, solution = _initial_solution("data/samples/tiny_full_matrix.txt")
    neighborhood = route_pair_neighborhood(instance, solution, random.Random(1), max_customers=10)

    result = HeuristicFixOptBackend(seed=1).reoptimize(instance, solution, neighborhood, _config())

    assert result.solution is not None
    assert validate_solution(instance, result.solution).is_feasible


def test_fixopt_non_worsening_default_keeps_cost_no_worse():
    instance, solution = _initial_solution()
    solver = FixOptimizeSolver(
        solution,
        SolverConfig(seed=1, metadata={"fixopt_iterations": 20, "fixopt_backend": "heuristic"}),
    )

    result = solver.solve(instance)

    assert result.cost is not None
    assert result.cost <= objective_cost(instance, solution)


def test_heuristic_backend_is_deterministic_for_fixed_seed():
    instance, solution = _initial_solution("data/samples/tiny_full_matrix.txt")
    neighborhood = route_neighborhood(instance, solution, random.Random(1), max_customers=8, max_routes=1)

    first = HeuristicFixOptBackend(seed=3).reoptimize(instance, solution, neighborhood, _config())
    second = HeuristicFixOptBackend(seed=3).reoptimize(instance, solution, neighborhood, _config())

    assert first.solution == second.solution


def test_fixopt_returns_feasible_solution_on_tiny_coords():
    instance = read_instance("data/samples/tiny_coords.txt")
    solver = FixOptimizeSolver(config=SolverConfig(seed=1, metadata={"fixopt_iterations": 20}))

    result = solver.solve(instance)

    assert result.solution is not None
    assert validate_solution(instance, result.solution).is_feasible


def test_fixopt_returns_feasible_solution_on_tiny_full_matrix():
    instance = read_instance("data/samples/tiny_full_matrix.txt")
    solver = FixOptimizeSolver(config=SolverConfig(seed=1, metadata={"fixopt_iterations": 20}))

    result = solver.solve(instance)

    assert result.solution is not None
    assert validate_solution(instance, result.solution).is_feasible


def test_fixopt_is_deterministic_for_fixed_seed():
    instance = read_instance("data/samples/tiny_coords.txt")
    config = SolverConfig(seed=4, metadata={"fixopt_iterations": 20, "fixopt_backend": "heuristic"})

    first = FixOptimizeSolver(config=config).solve(instance)
    second = FixOptimizeSolver(config=config).solve(instance)

    assert first.cost == second.cost
    assert first.solution == second.solution


def test_fixopt_best_cost_is_not_worse_than_initial_solution():
    instance, solution = _initial_solution()

    result = FixOptimizeSolver(
        solution,
        SolverConfig(seed=1, metadata={"fixopt_iterations": 30, "fixopt_backend": "heuristic"}),
    ).solve(instance)

    assert result.cost is not None
    assert result.cost <= objective_cost(instance, solution)


def test_mip_backend_lazy_import_behavior():
    if gurobi_available():
        backend = MIPFixOptBackend(seed=1)
        assert backend.backend_name == "mip"
    else:
        with pytest.raises(BackendUnavailable):
            MIPFixOptBackend(seed=1)


def test_auto_backend_falls_back_to_heuristic_without_requiring_gurobi():
    instance = read_instance("data/samples/tiny_coords.txt")

    result = FixOptimizeSolver(config=SolverConfig(seed=1, metadata={"fixopt_backend": "auto"})).solve(instance)

    assert result.solution is not None
    assert validate_solution(instance, result.solution).is_feasible
    assert result.metadata["backend_used"] in {"heuristic", "mip"}


def test_hybrid_returns_feasible_solution_on_tiny_coords():
    instance = read_instance("data/samples/tiny_coords.txt")
    solver = HybridALNSFixOptSolver(
        SolverConfig(seed=1, metadata={"num_starts": 8, "max_iterations": 40, "fixopt_iterations": 20})
    )

    result = solver.solve(instance)

    assert result.solution is not None
    assert validate_solution(instance, result.solution).is_feasible


def test_hybrid_returns_feasible_solution_on_tiny_full_matrix():
    instance = read_instance("data/samples/tiny_full_matrix.txt")
    solver = HybridALNSFixOptSolver(
        SolverConfig(seed=1, metadata={"num_starts": 8, "max_iterations": 40, "fixopt_iterations": 20})
    )

    result = solver.solve(instance)

    assert result.solution is not None
    assert validate_solution(instance, result.solution).is_feasible


def test_hybrid_cost_is_not_worse_than_constructive_ls_on_coords():
    instance = read_instance("data/samples/tiny_coords.txt")
    initial = ConstructiveLocalSearchSolver(
        SolverConfig(seed=1, metadata={"num_starts": 8, "max_iterations": 20})
    ).solve(instance)

    result = HybridALNSFixOptSolver(
        SolverConfig(seed=1, metadata={"num_starts": 8, "max_iterations": 50, "fixopt_iterations": 20})
    ).solve(instance)

    assert initial.cost is not None
    assert result.cost is not None
    assert result.cost <= initial.cost


def test_hybrid_splits_global_time_budget_between_both_phases():
    config = SolverConfig(seed=1, time_limit_seconds=20, metadata={"hybrid_alns_fraction": 0.75})

    assert _hybrid_time_budgets(config) == (15.0, 5.0)


def test_hybrid_explicit_fixopt_budget_is_capped_by_reserved_time():
    config = SolverConfig(
        seed=1,
        time_limit_seconds=20,
        metadata={"hybrid_alns_fraction": 0.6, "fixopt_time_limit": 3},
    )

    assert _hybrid_time_budgets(config) == (12.0, 3.0)


def test_hybrid_rejects_invalid_alns_fraction():
    config = SolverConfig(seed=1, time_limit_seconds=20, metadata={"hybrid_alns_fraction": 1.0})

    with pytest.raises(ValueError, match="hybrid_alns_fraction"):
        _hybrid_time_budgets(config)
