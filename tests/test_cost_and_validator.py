from smio_clrp.core.solution import Route, Solution
from smio_clrp.evaluation.cost import objective_cost, route_distance
from smio_clrp.evaluation.validator import validate_solution
from smio_clrp.io.instance_reader import read_instance


def _feasible_solution(instance_name="tiny_coords", reported_cost=None):
    return Solution(
        instance_name=instance_name,
        routes=[
            Route(1, [101, 102]),
            Route(2, [103, 104]),
            Route(2, [105]),
        ],
        reported_cost=reported_cost,
    )


def test_cost_computation_for_coords_solution():
    instance = read_instance("data/samples/tiny_coords.txt")
    solution = _feasible_solution()

    assert route_distance(instance, solution.routes[0]) == 4.8
    assert objective_cost(instance, solution) == 263.2


def test_validator_accepts_feasible_solution():
    instance = read_instance("data/samples/tiny_coords.txt")
    solution = _feasible_solution(reported_cost=263.2)

    result = validate_solution(instance, solution)

    assert result.is_feasible
    assert result.errors == []
    assert result.cost == 263.2


def test_validator_rejects_duplicate_customer():
    instance = read_instance("data/samples/tiny_coords.txt")
    solution = Solution(
        "tiny_coords",
        [Route(1, [101, 102]), Route(2, [101, 103, 104]), Route(2, [105])],
    )

    result = validate_solution(instance, solution)

    assert not result.is_feasible
    assert any("Customer 101 appears 2 times" in error for error in result.errors)


def test_validator_rejects_missing_customer():
    instance = read_instance("data/samples/tiny_coords.txt")
    solution = Solution("tiny_coords", [Route(1, [101, 102]), Route(2, [103, 104])])

    result = validate_solution(instance, solution)

    assert not result.is_feasible
    assert any("Missing customer 105" in error for error in result.errors)


def test_validator_rejects_capacity_violation():
    instance = read_instance("data/samples/tiny_coords.txt")
    solution = Solution("tiny_coords", [Route(1, [101, 102, 103]), Route(2, [104, 105])])

    result = validate_solution(instance, solution)

    assert not result.is_feasible
    assert any("exceeds vehicle capacity" in error for error in result.errors)


def test_validator_rejects_wrong_reported_cost():
    instance = read_instance("data/samples/tiny_coords.txt")
    solution = _feasible_solution(reported_cost=1.0)

    result = validate_solution(instance, solution)

    assert not result.is_feasible
    assert any("Reported cost" in error for error in result.errors)
