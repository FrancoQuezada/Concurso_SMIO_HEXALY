from __future__ import annotations

from smio_clrp.core.instance import Instance
from smio_clrp.core.solution import Route, Solution
from smio_clrp.evaluation.cost import objective_cost
from smio_clrp.evaluation.validator import validate_solution


def inter_route_two_opt_star(instance: Instance, solution: Solution, max_evaluations: int = 50) -> Solution:
    """Exchange route tails using bounded first improvement for predictable runtime."""
    base_cost = objective_cost(instance, solution)
    routes = solution.routes
    evaluations = 0
    for first_index, first in enumerate(routes):
        for second_index in range(first_index + 1, len(routes)):
            second = routes[second_index]
            for first_cut in range(1, len(first.customer_ids) + 1):
                for second_cut in range(1, len(second.customer_ids) + 1):
                    first_customers = first.customer_ids[:first_cut] + second.customer_ids[second_cut:]
                    second_customers = second.customer_ids[:second_cut] + first.customer_ids[first_cut:]
                    if not first_customers or not second_customers:
                        continue
                    evaluations += 1
                    if evaluations > max_evaluations:
                        return solution
                    candidate_routes = [Route(route.depot_id, list(route.customer_ids)) for route in routes]
                    candidate_routes[first_index] = Route(first.depot_id, first_customers)
                    candidate_routes[second_index] = Route(second.depot_id, second_customers)
                    candidate = Solution(solution.instance_name, candidate_routes)
                    validation = validate_solution(instance, candidate)
                    if not validation.is_feasible:
                        continue
                    cost = objective_cost(instance, candidate)
                    if cost + 1e-9 < base_cost:
                        return candidate
    return solution
