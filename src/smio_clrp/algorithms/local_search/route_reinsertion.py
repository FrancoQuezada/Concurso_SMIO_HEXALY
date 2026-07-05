from __future__ import annotations

from smio_clrp.algorithms.common import EPS, route_load
from smio_clrp.core.instance import Instance
from smio_clrp.core.solution import Route, Solution
from smio_clrp.evaluation.cost import objective_cost
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
    best_routes: list[Route] | None = None
    best_cost = float("inf")
    for route_index, route in enumerate(routes):
        for position in range(len(route.customer_ids) + 1):
            candidate = [Route(item.depot_id, list(item.customer_ids)) for item in routes]
            customers = list(candidate[route_index].customer_ids)
            customers.insert(position, customer_id)
            candidate[route_index] = Route(route.depot_id, customers)
            if _partial_constraints_ok(instance, candidate):
                cost = objective_cost(instance, Solution(instance.name, candidate))
                if cost < best_cost:
                    best_routes = candidate
                    best_cost = cost
    for depot in instance.depots:
        candidate = [Route(item.depot_id, list(item.customer_ids)) for item in routes]
        candidate.append(Route(depot.id, [customer_id]))
        if _partial_constraints_ok(instance, candidate):
            cost = objective_cost(instance, Solution(instance.name, candidate))
            if cost < best_cost:
                best_routes = candidate
                best_cost = cost
    if best_routes is None:
        raise ValueError(f"No feasible reinsertion for customer {customer_id}")
    return best_routes


def _partial_constraints_ok(instance: Instance, routes: list[Route]) -> bool:
    depot_loads: dict[int, float] = {}
    depot_counts: dict[int, int] = {}
    for route in routes:
        load = route_load(instance, route)
        if load > instance.vehicle_capacity + EPS:
            return False
        depot = instance.depots_by_id[route.depot_id]
        depot_loads[route.depot_id] = depot_loads.get(route.depot_id, 0.0) + load
        depot_counts[route.depot_id] = depot_counts.get(route.depot_id, 0) + 1
        if depot_loads[route.depot_id] > depot.capacity + EPS:
            return False
        if depot_counts[route.depot_id] > depot.vehicle_limit:
            return False
    return True
