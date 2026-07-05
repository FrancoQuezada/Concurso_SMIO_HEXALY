from __future__ import annotations

from smio_clrp.core.instance import Instance
from smio_clrp.core.solution import Route, Solution
from smio_clrp.evaluation.cost import objective_cost
from smio_clrp.evaluation.validator import validate_solution


def swap_customers(instance: Instance, solution: Solution) -> Solution:
    best = solution
    best_cost = objective_cost(instance, solution)
    positions = [
        (route_index, customer_index)
        for route_index, route in enumerate(solution.routes)
        for customer_index, _ in enumerate(route.customer_ids)
    ]
    for first_pos_index, first in enumerate(positions):
        for second in positions[first_pos_index + 1 :]:
            candidate_routes = _swap(solution.routes, first, second)
            candidate = Solution(solution.instance_name, candidate_routes)
            validation = validate_solution(instance, candidate)
            if not validation.is_feasible:
                continue
            cost = objective_cost(instance, candidate)
            if cost + 1e-9 < best_cost:
                best = candidate
                best_cost = cost
    return best


def _swap(
    routes: list[Route],
    first: tuple[int, int],
    second: tuple[int, int],
) -> list[Route]:
    updated = [Route(route.depot_id, list(route.customer_ids)) for route in routes]
    first_route, first_customer = first
    second_route, second_customer = second
    updated[first_route].customer_ids[first_customer], updated[second_route].customer_ids[second_customer] = (
        updated[second_route].customer_ids[second_customer],
        updated[first_route].customer_ids[first_customer],
    )
    return updated
