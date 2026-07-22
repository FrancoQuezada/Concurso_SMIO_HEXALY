from __future__ import annotations

from smio_clrp.algorithms.common import EPS, depot_loads, depot_route_counts, route_load
from smio_clrp.core.distance import distance
from smio_clrp.core.instance import Instance
from smio_clrp.core.solution import Route, Solution
from smio_clrp.evaluation.cost import route_distance
from smio_clrp.evaluation.validator import validate_solution


MAX_SEGMENT_LENGTH = 3


def or_opt(instance: Instance, solution: Solution) -> Solution:
    """Find the single best "move a short consecutive segment (1-3 customers) to a
    different route" move. Complements relocate_customer (segment length 1 only) and
    intra_route_two_opt (reordering within one route only): neither can move a
    multi-customer chunk between routes in one step, which is a common source of
    stuck local optima in VRP-style neighborhoods.

    Only considers moves into a *different* existing route (never back into the
    source route, never opening a brand-new route) to keep index bookkeeping simple
    and unambiguous; relocate_customer already covers single-customer moves into new
    routes, and this operator focuses on the multi-customer case relocate can't reach.
    """
    routes = solution.routes
    loads = depot_loads(instance, routes)
    counts = depot_route_counts(routes)
    route_loads = [route_load(instance, route) for route in routes]

    best_delta = -EPS
    best_move: tuple[int, int, int, int, int] | None = None
    # (source_index, start, length, target_index, target_position)

    for source_index, source in enumerate(routes):
        customers = source.customer_ids
        n = len(customers)
        for length in range(1, min(MAX_SEGMENT_LENGTH, n) + 1):
            for start in range(0, n - length + 1):
                segment = customers[start : start + length]
                demand = sum(instance.customers_by_id[c].demand for c in segment)

                if length == n:
                    # Emptying the whole route also closes it; credit the fixed route
                    # cost and (if this was the depot's only route) the opening cost too.
                    removal_gain = route_distance(instance, source) + instance.route_fixed_cost
                    if counts.get(source.depot_id, 0) == 1:
                        removal_gain += instance.depots_by_id[source.depot_id].opening_cost
                else:
                    removal_gain = -_segment_removal_delta(instance, source, start, length)

                for target_index, target in enumerate(routes):
                    if target_index == source_index:
                        continue
                    if route_loads[target_index] + demand > instance.vehicle_capacity + EPS:
                        continue
                    if target.depot_id != source.depot_id:
                        target_depot = instance.depots_by_id[target.depot_id]
                        if loads.get(target.depot_id, 0.0) + demand > target_depot.capacity + EPS:
                            continue
                    for position in range(len(target.customer_ids) + 1):
                        insert_cost = _segment_insertion_delta(instance, target, segment, position)
                        delta = insert_cost - removal_gain
                        if delta < best_delta:
                            best_delta = delta
                            best_move = (source_index, start, length, target_index, position)

    if best_move is None:
        return solution

    updated = _apply_segment_move(routes, *best_move)
    candidate = Solution(solution.instance_name, updated)
    validation = validate_solution(instance, candidate)
    if not validation.is_feasible:
        return solution
    return candidate


def _segment_removal_delta(instance: Instance, route: Route, start: int, length: int) -> float:
    customers = route.customer_ids
    size = len(customers)
    prev_node = ("customer", customers[start - 1]) if start > 0 else ("depot", route.depot_id)
    next_node = ("customer", customers[start + length]) if start + length < size else ("depot", route.depot_id)
    seg_first = ("customer", customers[start])
    seg_last = ("customer", customers[start + length - 1])
    removed_edges = distance(instance, prev_node, seg_first) + distance(instance, seg_last, next_node)
    added_edge = distance(instance, prev_node, next_node)
    return added_edge - removed_edges


def _segment_insertion_delta(instance: Instance, route: Route, segment: list[int], position: int) -> float:
    customers = route.customer_ids
    size = len(customers)
    prev_node = ("customer", customers[position - 1]) if position > 0 else ("depot", route.depot_id)
    next_node = ("customer", customers[position]) if position < size else ("depot", route.depot_id)
    seg_first = ("customer", segment[0])
    seg_last = ("customer", segment[-1])
    removed_edge = distance(instance, prev_node, next_node)
    added_edges = distance(instance, prev_node, seg_first) + distance(instance, seg_last, next_node)
    return added_edges - removed_edge


def _apply_segment_move(
    routes: list[Route],
    source_index: int,
    start: int,
    length: int,
    target_index: int,
    position: int,
) -> list[Route]:
    segment = routes[source_index].customer_ids[start : start + length]
    updated: list[Route] = []
    for index, route in enumerate(routes):
        customers = list(route.customer_ids)
        if index == source_index:
            del customers[start : start + length]
        if index == target_index:
            customers[position:position] = segment
        if customers:
            updated.append(Route(route.depot_id, customers))
    return updated
