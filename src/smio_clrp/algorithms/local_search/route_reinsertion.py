from __future__ import annotations

from smio_clrp.algorithms.common import EPS, depot_loads, depot_route_counts, insertion_delta, route_load
from smio_clrp.core.instance import Instance
from smio_clrp.core.solution import Route, Solution
from smio_clrp.evaluation.cost import objective_cost, route_distance
from smio_clrp.evaluation.validator import validate_solution


def route_reinsertion(instance: Instance, solution: Solution) -> Solution:
    best = solution
    best_cost = objective_cost(instance, solution)
    for route_index, route in enumerate(solution.routes):
        remaining_routes = [candidate for index, candidate in enumerate(solution.routes) if index != route_index]
        try:
            candidate_routes = _reinsert_customers(instance, remaining_routes, route.customer_ids)
        except ValueError:
            continue
        candidate = Solution(solution.instance_name, candidate_routes)
        validation = validate_solution(instance, candidate)
        if not validation.is_feasible:
            continue
        cost = objective_cost(instance, candidate)
        if cost + 1e-9 < best_cost:
            best = candidate
            best_cost = cost
    return best


def _reinsert_customers(
    instance: Instance,
    routes: list[Route],
    customer_ids: list[int],
) -> list[Route]:
    current = [Route(route.depot_id, list(route.customer_ids)) for route in routes]
    for customer_id in sorted(customer_ids, key=lambda item: (-instance.customers_by_id[item].demand, item)):
        current = _best_partial_insertion(instance, current, customer_id)
    return current


def _best_partial_insertion(instance: Instance, routes: list[Route], customer_id: int) -> list[Route]:
    """Insert one customer at its cheapest feasible spot, evaluated via route-local
    distance deltas instead of a full-solution objective_cost per candidate (that was
    O(n) per candidate -- intractable when called once per released customer above
    ~100-150 customers)."""
    demand = instance.customers_by_id[customer_id].demand
    opened_depot_ids = {route.depot_id for route in routes}
    loads = depot_loads(instance, routes)
    counts = depot_route_counts(routes)
    route_loads = [route_load(instance, route) for route in routes]

    best_cost = float("inf")
    best_move: tuple[int, int] | None = None  # (route_index, position); route_index=-1 means new route

    for route_index, route in enumerate(routes):
        depot = instance.depots_by_id[route.depot_id]
        if route_loads[route_index] + demand > instance.vehicle_capacity + EPS:
            continue
        if loads.get(route.depot_id, 0.0) + demand > depot.capacity + EPS:
            continue
        for position in range(len(route.customer_ids) + 1):
            cost = insertion_delta(instance, route, customer_id, position)
            if cost < best_cost:
                best_cost = cost
                best_move = (route_index, position)

    if demand <= instance.vehicle_capacity + EPS:
        for depot in instance.depots:
            if loads.get(depot.id, 0.0) + demand > depot.capacity + EPS:
                continue
            if counts.get(depot.id, 0) + 1 > depot.vehicle_limit:
                continue
            opening_cost = 0.0 if depot.id in opened_depot_ids else depot.opening_cost
            cost = route_distance(instance, Route(depot.id, [customer_id])) + instance.route_fixed_cost + opening_cost
            if cost < best_cost:
                best_cost = cost
                best_move = (-1, depot.id)

    if best_move is None:
        raise ValueError(f"No feasible reinsertion for customer {customer_id}")

    route_index, position_or_depot = best_move
    if route_index == -1:
        return routes + [Route(position_or_depot, [customer_id])]
    updated = [Route(item.depot_id, list(item.customer_ids)) for item in routes]
    customers = list(updated[route_index].customer_ids)
    customers.insert(position_or_depot, customer_id)
    updated[route_index] = Route(routes[route_index].depot_id, customers)
    return updated
