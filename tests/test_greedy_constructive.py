from smio_clrp.algorithms.constructive.greedy import solve_greedy_nearest_depot
from smio_clrp.evaluation.validator import validate_solution
from smio_clrp.io.instance_reader import read_instance


def test_greedy_nearest_depot_returns_feasible_solution_for_coords_sample():
    instance = read_instance("data/samples/tiny_coords.txt")

    result = solve_greedy_nearest_depot(instance, seed=1)

    assert result.solution is not None
    assert result.feasible
    assert validate_solution(instance, result.solution).is_feasible


def test_greedy_nearest_depot_returns_feasible_solution_for_full_matrix_sample():
    instance = read_instance("data/samples/tiny_full_matrix.txt")

    result = solve_greedy_nearest_depot(instance, seed=1)

    assert result.solution is not None
    assert result.feasible
    assert validate_solution(instance, result.solution).is_feasible
