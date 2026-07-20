from __future__ import annotations

from smio_clrp.algorithms.common import EPS, depot_loads, depot_route_counts, route_load
from smio_clrp.core.instance import Instance
from smio_clrp.core.solution import Route, Solution
from smio_clrp.evaluation.cost import route_distance
from smio_clrp.evaluation.validator import validate_solution


def relocate_customers(instance: Instance, solution: Solution) -> Solution:
    return relocate_customer(instance, solution)


def relocate_customer(instance: Instance, solution: Solution) -> Solution:
    """Find the single best "move one customer to a different route/position" move.

    Evaluated via route-local distance deltas instead of full-solution
    objective_cost/validate_solution per candidate (that was O(n) per
    candidate, i.e. O(n^2) or worse per call -- intractable above ~100-150
    customers). The final chosen candidate is still validated for real
    before being returned, so this is an exact search over an approximate
    (but correct where it matters -- capacity/vehicle_limit checks are exact,
    only the rare "opening a route at the same depot the customer left"
    option is pruned as never-beneficial) neighborhood.
    """
    routes = solution.routes
    opened_depot_ids = {route.depot_id for route in routes}
    loads = depot_loads(instance, routes)
    counts = depot_route_counts(routes)

    best_delta = -EPS
    best_move: tuple[str, int, int, int, int] | None = None
    # ("existing", source_index, source_position, target_index, target_position)
    # ("new", source_index, source_position, depot_id, 0)

    for source_index, source in enumerate(routes):
        for source_position, customer_id in enumerate(source.customer_ids):
            demand = instance.customers_by_id[customer_id].demand
            remaining = source.customer_ids[:source_position] + source.customer_ids[source_position + 1 :]
            if remaining:
                reduced_source = Route(source.depot_id, remaining)
                removal_gain = route_distance(instance, source) - route_distance(instance, reduced_source)
            else:
                removal_gain = route_distance(instance, source) + instance.route_fixed_cost

            for target_index, target in enumerate(routes):
                if target_index == source_index:
                    continue
                if route_load(instance, target) + demand > instance.vehicle_capacity + EPS:
                    continue
                if target.depot_id != source.depot_id:
                    target_depot = instance.depots_by_id[target.depot_id]
                    if loads[target.depot_id] + demand > target_depot.capacity + EPS:
                        continue
                for target_position in range(len(target.customer_ids) + 1):
                    customers = list(target.customer_ids)
                    customers.insert(target_position, customer_id)
                    insert_cost = route_distance(instance, Route(target.depot_id, customers)) - route_distance(
                        instance, target
                    )
                    delta = insert_cost - removal_gain
                    if delta < best_delta:
                        best_delta = delta
                        best_move = ("existing", source_index, source_position, target_index, target_position)

            if demand <= instance.vehicle_capacity + EPS:
                for depot in instance.depots:
                    if depot.id == source.depot_id:
                        # Opening a fresh route at the depot the customer is leaving is never
                        # better than the options already considered above; skip for speed.
                        continue
                    if counts.get(depot.id, 0) + 1 > depot.vehicle_limit:
                        continue
                    if loads.get(depot.id, 0.0) + demand > depot.capacity + EPS:
                        continue
                    opening_cost = 0.0 if depot.id in opened_depot_ids else depot.opening_cost
                    insert_cost = (
                        route_distance(instance, Route(depot.id, [customer_id]))
                        + instance.route_fixed_cost
                        + opening_cost
                    )
                    delta = insert_cost - removal_gain
                    if delta < best_delta:
                        best_delta = delta
                        best_move = ("new", source_index, source_position, depot.id, 0)

    if best_move is None:
        return solution

    kind, source_index, source_position, target_index_or_depot, target_position = best_move
    if kind == "existing":
        updated = _relocate_between_existing(routes, source_index, source_position, target_index_or_depot, target_position)
    else:
        updated = _relocate_to_new_route(routes, source_index, source_position, target_index_or_depot)

    candidate = Solution(solution.instance_name, updated)
    validation = validate_solution(instance, candidate)
    if not validation.is_feasible:
        return solution
    return candidate


def _relocate_between_existing(
    routes: list[Route],
    source_index: int,
    source_position: int,
    target_index: int,
    target_position: int,
) -> list[Route]:
    updated: list[Route] = []
    customer_id = routes[source_index].customer_ids[source_position]
    for index, route in enumerate(routes):
        customers = list(route.customer_ids)
        if index == source_index:
            customers.pop(source_position)
        if index == target_index:
            customers.insert(target_position, customer_id)
        if customers:
            updated.append(Route(route.depot_id, customers))
    return updated


def _relocate_to_new_route(
    routes: list[Route],
    source_index: int,
    source_position: int,
    depot_id: int,
) -> list[Route]:
    updated: list[Route] = []
    customer_id = routes[source_index].customer_ids[source_position]
    for index, route in enumerate(routes):
        customers = list(route.customer_ids)
        if index == source_index:
            customers.pop(source_position)
        if customers:
            updated.append(Route(route.depot_id, customers))
    updated.append(Route(depot_id, [customer_id]))
    return updated
