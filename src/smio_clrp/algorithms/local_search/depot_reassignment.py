from __future__ import annotations

from smio_clrp.core.instance import Instance
from smio_clrp.core.solution import Route, Solution
from smio_clrp.evaluation.cost import objective_cost
from smio_clrp.evaluation.validator import validate_solution


def reassign_route_depot(instance: Instance, solution: Solution) -> Solution:
    """Move a complete route to another depot, including open/close cost effects."""
    best = solution
    best_cost = objective_cost(instance, solution)
    for route_index, route in enumerate(solution.routes):
        for depot in instance.depots:
            if depot.id == route.depot_id:
                continue
            candidate_routes = [Route(item.depot_id, list(item.customer_ids)) for item in solution.routes]
            candidate_routes[route_index] = Route(depot.id, list(route.customer_ids))
            candidate = Solution(solution.instance_name, candidate_routes)
            validation = validate_solution(instance, candidate)
            if not validation.is_feasible:
                continue
            cost = objective_cost(instance, candidate)
            if cost + 1e-9 < best_cost:
                best = candidate
                best_cost = cost
    return best
