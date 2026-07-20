from __future__ import annotations

from smio_clrp.algorithms.common import EPS, depot_loads, route_load
from smio_clrp.core.instance import Instance
from smio_clrp.core.solution import Route, Solution
from smio_clrp.evaluation.cost import route_distance
from smio_clrp.evaluation.validator import validate_solution


def swap_customers(instance: Instance, solution: Solution) -> Solution:
    """Find the single best "exchange two customers' positions" move.

    Evaluated via route-local distance deltas instead of full-solution
    objective_cost/validate_solution per pair (that was O(pairs) full
    recomputes, i.e. O(n^2) per call). The chosen candidate is still
    validated for real before being returned.
    """
    routes = solution.routes
    loads = depot_loads(instance, routes)
    # Precomputed once, not re-summed inside the O(positions^2) pair loop below.
    route_loads = [route_load(instance, route) for route in routes]
    positions = [
        (route_index, customer_index)
        for route_index, route in enumerate(routes)
        for customer_index, _ in enumerate(route.customer_ids)
    ]

    best_delta = -EPS
    best_pair: tuple[tuple[int, int], tuple[int, int]] | None = None

    for first_pos_index, first in enumerate(positions):
        first_route_index, first_customer_index = first
        first_route = routes[first_route_index]
        first_customer = first_route.customer_ids[first_customer_index]
        for second in positions[first_pos_index + 1 :]:
            second_route_index, second_customer_index = second
            second_route = routes[second_route_index]
            second_customer = second_route.customer_ids[second_customer_index]

            if first_route_index == second_route_index:
                customers = list(first_route.customer_ids)
                customers[first_customer_index], customers[second_customer_index] = (
                    customers[second_customer_index],
                    customers[first_customer_index],
                )
                delta = route_distance(instance, Route(first_route.depot_id, customers)) - route_distance(
                    instance, first_route
                )
            else:
                demand_first = instance.customers_by_id[first_customer].demand
                demand_second = instance.customers_by_id[second_customer].demand
                if route_loads[first_route_index] - demand_first + demand_second > instance.vehicle_capacity + EPS:
                    continue
                if route_loads[second_route_index] - demand_second + demand_first > instance.vehicle_capacity + EPS:
                    continue
                if first_route.depot_id != second_route.depot_id:
                    first_depot = instance.depots_by_id[first_route.depot_id]
                    second_depot = instance.depots_by_id[second_route.depot_id]
                    if loads[first_route.depot_id] - demand_first + demand_second > first_depot.capacity + EPS:
                        continue
                    if loads[second_route.depot_id] - demand_second + demand_first > second_depot.capacity + EPS:
                        continue
                first_customers = list(first_route.customer_ids)
                first_customers[first_customer_index] = second_customer
                second_customers = list(second_route.customer_ids)
                second_customers[second_customer_index] = first_customer
                delta = (
                    route_distance(instance, Route(first_route.depot_id, first_customers))
                    - route_distance(instance, first_route)
                    + route_distance(instance, Route(second_route.depot_id, second_customers))
                    - route_distance(instance, second_route)
                )
            if delta < best_delta:
                best_delta = delta
                best_pair = (first, second)

    if best_pair is None:
        return solution
    candidate_routes = _swap(routes, best_pair[0], best_pair[1])
    candidate = Solution(solution.instance_name, candidate_routes)
    validation = validate_solution(instance, candidate)
    if not validation.is_feasible:
        return solution
    return candidate


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
