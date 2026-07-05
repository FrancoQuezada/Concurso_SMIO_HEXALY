from smio_clrp.algorithms.constructive.greedy import solve_greedy_nearest_depot
from smio_clrp.algorithms.constructive.multistart import MultiStartConfig, multi_start_constructive
from smio_clrp.algorithms.constructive.regret import regret_insertion
from smio_clrp.algorithms.constructive.savings import savings_routes_per_depot
from smio_clrp.algorithms.local_search.driver import improve_solution
from smio_clrp.algorithms.local_search.relocate import relocate_customer
from smio_clrp.algorithms.local_search.route_reinsertion import route_reinsertion
from smio_clrp.algorithms.local_search.solver import ConstructiveLocalSearchSolver
from smio_clrp.algorithms.local_search.swap import swap_customers
from smio_clrp.algorithms.local_search.two_opt import intra_route_two_opt
from smio_clrp.algorithms.base import SolverConfig
from smio_clrp.core.solution import Route, Solution
from smio_clrp.evaluation.cost import objective_cost, route_distance
from smio_clrp.evaluation.validator import validate_solution
from smio_clrp.io.instance_reader import read_instance


def test_savings_produces_feasible_solution_on_coords():
    instance = read_instance("data/samples/tiny_coords.txt")

    solution = savings_routes_per_depot(instance)

    assert validate_solution(instance, solution).is_feasible


def test_savings_produces_feasible_solution_on_full_matrix():
    instance = read_instance("data/samples/tiny_full_matrix.txt")

    solution = savings_routes_per_depot(instance)

    assert validate_solution(instance, solution).is_feasible


def test_regret_insertion_produces_feasible_solution_on_both_samples():
    for path in ["data/samples/tiny_coords.txt", "data/samples/tiny_full_matrix.txt"]:
        instance = read_instance(path)

        assert validate_solution(instance, regret_insertion(instance, regret_k=2)).is_feasible
        assert validate_solution(instance, regret_insertion(instance, regret_k=3)).is_feasible


def test_multistart_is_deterministic_for_fixed_seed():
    instance = read_instance("data/samples/tiny_coords.txt")
    config = MultiStartConfig(num_starts=12, seed=7)

    first = multi_start_constructive(instance, config)
    second = multi_start_constructive(instance, config)

    assert first.feasible
    assert second.feasible
    assert first.cost == second.cost
    assert first.solution == second.solution


def test_two_opt_never_worsens_when_improvement_exists():
    instance = read_instance("data/samples/tiny_coords.txt")
    route = Route(1, [105, 101, 102])

    improved = intra_route_two_opt(instance, route)

    assert route_distance(instance, improved) <= route_distance(instance, route)


def test_relocate_preserves_feasibility():
    instance = read_instance("data/samples/tiny_coords.txt")
    solution = savings_routes_per_depot(instance)

    improved = relocate_customer(instance, solution)

    assert validate_solution(instance, improved).is_feasible


def test_swap_preserves_feasibility():
    instance = read_instance("data/samples/tiny_coords.txt")
    solution = savings_routes_per_depot(instance)

    improved = swap_customers(instance, solution)

    assert validate_solution(instance, improved).is_feasible


def test_route_reinsertion_preserves_feasibility():
    instance = read_instance("data/samples/tiny_coords.txt")
    solution = savings_routes_per_depot(instance)

    improved = route_reinsertion(instance, solution)

    assert validate_solution(instance, improved).is_feasible


def test_constructive_ls_produces_feasible_solution():
    instance = read_instance("data/samples/tiny_full_matrix.txt")
    solver = ConstructiveLocalSearchSolver(
        SolverConfig(seed=1, time_limit_seconds=10, metadata={"num_starts": 8, "max_iterations": 20})
    )

    result = solver.solve(instance)

    assert result.solution is not None
    assert result.feasible
    assert validate_solution(instance, result.solution).is_feasible


def test_constructive_ls_cost_is_not_worse_than_greedy_on_coords():
    instance = read_instance("data/samples/tiny_coords.txt")
    greedy = solve_greedy_nearest_depot(instance, seed=1)
    solver = ConstructiveLocalSearchSolver(
        SolverConfig(seed=1, time_limit_seconds=10, metadata={"num_starts": 20, "max_iterations": 50})
    )

    result = solver.solve(instance)

    assert result.solution is not None
    assert greedy.cost is not None
    assert result.cost <= greedy.cost


def test_local_search_supports_asymmetric_full_matrix():
    instance = read_instance("data/samples/tiny_full_matrix.txt")
    solution = regret_insertion(instance, regret_k=3)

    improved = improve_solution(instance, solution, max_iterations=10)

    assert validate_solution(instance, improved).is_feasible
    assert objective_cost(instance, improved) <= objective_cost(instance, solution)
