from __future__ import annotations

from smio_clrp.core.instance import Instance
from smio_clrp.core.solution import Route, Solution
from smio_clrp.evaluation.cost import objective_cost, route_distance
from smio_clrp.evaluation.validator import validate_solution


def intra_route_two_opt(instance: Instance, route: Route) -> Route:
    """Improve one route by reversing customer subsequences using directed route costs."""
    best = route
    best_cost = route_distance(instance, route)
    improved = True
    while improved:
        improved = False
        customers = best.customer_ids
        for start in range(len(customers) - 1):
            for end in range(start + 1, len(customers)):
                candidate_customers = (
                    customers[:start]
                    + list(reversed(customers[start : end + 1]))
                    + customers[end + 1 :]
                )
                candidate = Route(best.depot_id, candidate_customers)
                candidate_cost = route_distance(instance, candidate)
                if candidate_cost + 1e-9 < best_cost:
                    best = candidate
                    best_cost = candidate_cost
                    improved = True
                    break
            if improved:
                break
    return best


def improve_solution_two_opt(instance: Instance, solution: Solution) -> Solution:
    routes = [intra_route_two_opt(instance, route) for route in solution.routes]
    candidate = Solution(solution.instance_name, routes)
    validation = validate_solution(instance, candidate)
    if validation.is_feasible and objective_cost(instance, candidate) <= objective_cost(instance, solution) + 1e-9:
        return candidate
    return solution


def improve_route_two_opt(instance: Instance, route: Route) -> Route:
    return intra_route_two_opt(instance, route)
