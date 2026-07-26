from __future__ import annotations

from smio_clrp.algorithms.common import EPS, depot_loads, route_load
from smio_clrp.core.instance import Instance
from smio_clrp.core.solution import Route, Solution
from smio_clrp.evaluation.cost import route_distance
from smio_clrp.evaluation.validator import validate_solution


def inter_route_two_opt_star(instance: Instance, solution: Solution) -> Solution:
    """Find the single best "exchange the tails of two routes" move (2-opt* for VRP):
    for routes (depot_a -> ... -> a_i -> a_i+1 -> ...) and (depot_b -> ... -> b_j -> b_j+1
    -> ...), reconnect as (depot_a -> ... -> a_i) + (b_j+1 -> ... -> depot_b) and
    (depot_b -> ... -> b_j) + (a_i+1 -> ... -> depot_a).

    Complements relocate/swap/or_opt, none of which reconnect two routes at a cut point:
    relocate and or_opt move a bounded-size chunk of customers, and swap exchanges single
    customers, but none can restructure two long routes' shared boundary in one move --
    exactly the kind of move needed to fix two routes that each detour toward the other's
    territory. Depots may differ: a customer's depot assignment changes if it ends up in
    a tail reattached to the other route's depot.
    """
    routes = solution.routes
    loads = depot_loads(instance, routes)
    route_loads = [route_load(instance, route) for route in routes]

    best_delta = -EPS
    best_move: tuple[int, int, int, int] | None = None
    # (first_route_index, cut_i, second_route_index, cut_j)

    for first_index, first in enumerate(routes):
        first_customers = first.customer_ids
        for second_index in range(first_index + 1, len(routes)):
            second = routes[second_index]
            second_customers = second.customer_ids
            if first.depot_id == second.depot_id:
                same_depot_capacity_ok = True
            else:
                same_depot_capacity_ok = False

            base_cost = route_distance(instance, first) + route_distance(instance, second)

            for cut_i in range(len(first_customers) + 1):
                first_head = first_customers[:cut_i]
                first_tail = first_customers[cut_i:]
                head_demand = sum(instance.customers_by_id[c].demand for c in first_head)
                tail_demand = sum(instance.customers_by_id[c].demand for c in first_tail)
                for cut_j in range(len(second_customers) + 1):
                    if cut_i == 0 and cut_j == 0:
                        continue
                    if cut_i == len(first_customers) and cut_j == len(second_customers):
                        continue
                    second_head = second_customers[:cut_j]
                    second_tail = second_customers[cut_j:]
                    second_head_demand = sum(instance.customers_by_id[c].demand for c in second_head)
                    second_tail_demand = sum(instance.customers_by_id[c].demand for c in second_tail)

                    new_first = first_head + second_tail
                    new_second = second_head + first_tail
                    new_first_demand = head_demand + second_tail_demand
                    new_second_demand = second_head_demand + tail_demand
                    if new_first_demand > instance.vehicle_capacity + EPS:
                        continue
                    if new_second_demand > instance.vehicle_capacity + EPS:
                        continue

                    if not same_depot_capacity_ok:
                        first_depot = instance.depots_by_id[first.depot_id]
                        second_depot = instance.depots_by_id[second.depot_id]
                        # Depot loads shift by the demand that crosses over (the swapped tails).
                        first_depot_new_load = loads[first.depot_id] - tail_demand + second_tail_demand
                        second_depot_new_load = loads[second.depot_id] - second_tail_demand + tail_demand
                        if first_depot_new_load > first_depot.capacity + EPS:
                            continue
                        if second_depot_new_load > second_depot.capacity + EPS:
                            continue

                    if not new_first and not new_second:
                        continue
                    new_cost = (
                        (route_distance(instance, Route(first.depot_id, new_first)) if new_first else 0.0)
                        + (route_distance(instance, Route(second.depot_id, new_second)) if new_second else 0.0)
                    )
                    delta = new_cost - base_cost
                    if delta < best_delta:
                        best_delta = delta
                        best_move = (first_index, cut_i, second_index, cut_j)

    if best_move is None:
        return solution

    first_index, cut_i, second_index, cut_j = best_move
    first = routes[first_index]
    second = routes[second_index]
    new_first = first.customer_ids[:cut_i] + second.customer_ids[cut_j:]
    new_second = second.customer_ids[:cut_j] + first.customer_ids[cut_i:]

    updated: list[Route] = []
    for index, route in enumerate(routes):
        if index == first_index:
            if new_first:
                updated.append(Route(route.depot_id, list(new_first)))
        elif index == second_index:
            if new_second:
                updated.append(Route(route.depot_id, list(new_second)))
        else:
            updated.append(Route(route.depot_id, list(route.customer_ids)))

    candidate = Solution(solution.instance_name, updated)
    validation = validate_solution(instance, candidate)
    if not validation.is_feasible:
        return solution
    return candidate
