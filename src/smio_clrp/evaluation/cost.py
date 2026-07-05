from __future__ import annotations

from smio_clrp.core.distance import distance
from smio_clrp.core.instance import Instance
from smio_clrp.core.solution import Route, Solution


def route_distance(instance: Instance, route: Route) -> float:
    if not route.customer_ids:
        return 0.0
    depot = ("depot", route.depot_id)
    total = distance(instance, depot, ("customer", route.customer_ids[0]))
    for current_id, next_id in zip(route.customer_ids, route.customer_ids[1:]):
        total += distance(instance, ("customer", current_id), ("customer", next_id))
    total += distance(instance, ("customer", route.customer_ids[-1]), depot)
    return total


def objective_cost(instance: Instance, solution: Solution) -> float:
    opening = sum(instance.depots_by_id[depot_id].opening_cost for depot_id in solution.opened_depot_ids)
    dispatch = instance.route_fixed_cost * len(solution.routes)
    routing = sum(route_distance(instance, route) for route in solution.routes)
    return float(opening + dispatch + routing)
