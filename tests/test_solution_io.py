from smio_clrp.core.solution import Route, Solution
from smio_clrp.evaluation.validator import validate_solution
from smio_clrp.io.instance_reader import read_instance
from smio_clrp.io.solution_reader import parse_solution_text, read_solution
from smio_clrp.io.solution_writer import format_solution, write_solution


def test_solution_writer_then_reader_preserves_feasibility(tmp_path):
    instance = read_instance("data/samples/tiny_coords.txt")
    solution = Solution(
        instance_name=instance.name,
        routes=[Route(1, [101, 102]), Route(2, [103, 104]), Route(2, [105])],
    )
    path = tmp_path / "solution.sol"

    write_solution(solution, path, instance=instance)
    parsed = read_solution(path)
    result = validate_solution(instance, parsed)

    assert result.is_feasible
    assert result.cost == 263.2


def test_solution_parser_accepts_empty_route_for_validator_to_reject():
    text = """# instance = tiny_coords
COST : 0
DEPOTS_OPENED : 1
ROUTES : 1
DEPOT 1
ROUTE :
EOF
"""
    solution = parse_solution_text(text)

    assert solution.routes[0].customer_ids == []


def test_solution_writer_omits_depot_blocks_with_no_routes():
    instance = read_instance("data/samples/tiny_coords.txt")
    solution = Solution(instance.name, [Route(1, [101, 102]), Route(2, [])])

    text = format_solution(solution, instance=instance)

    assert "DEPOT 1" in text
    assert "DEPOT 2" not in text
