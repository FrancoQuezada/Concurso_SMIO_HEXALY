from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from smio_clrp.algorithms.common import route_load
from smio_clrp.core.distance import distance
from smio_clrp.core.instance import Instance
from smio_clrp.core.solution import Route, Solution
from smio_clrp.evaluation.cost import route_distance


@dataclass(frozen=True)
class FixOptNeighborhood:
    released_customer_ids: list[int]
    candidate_depot_ids: list[int]
    fixed_routes: list[Route]
    metadata: dict[str, Any] = field(default_factory=dict)


DEFAULT_NEIGHBORHOODS = [
    "depot",
    "route",
    "boundary",
    "expensive",
    "route_pair",
]


def _candidate_depot_count(instance: Instance) -> int:
    """Scale the candidate-depot window with instance size. A fixed window of 2-3
    (the previous hardcoded value everywhere in this module) is reasonable on the
    5-14-depot small instances, but on the 20-40-depot medium/large instances it
    means FixOpt can only ever consider the 2-3 nearest depots as an alternative --
    a genuinely better depot for a released customer/cluster is often outside that
    narrow window, so a beneficial "close depot A, open depot B" swap is never even
    offered to the backend to evaluate. Capped at 10 to keep subproblems bounded on
    instances with very many depots.
    """
    return min(10, max(3, len(instance.depots) // 3))


def build_neighborhood(
    instance: Instance,
    solution: Solution,
    neighborhood_type: str,
    rng: random.Random,
    max_customers: int,
    max_routes: int,
) -> FixOptNeighborhood:
    if neighborhood_type == "depot":
        return depot_neighborhood(instance, solution, rng, max_customers)
    if neighborhood_type == "route":
        return route_neighborhood(instance, solution, rng, max_customers, max_routes)
    if neighborhood_type == "boundary":
        return boundary_customer_neighborhood(instance, solution, rng, max_customers)
    if neighborhood_type == "expensive":
        return expensive_customer_neighborhood(instance, solution, max_customers)
    if neighborhood_type == "route_pair":
        return route_pair_neighborhood(instance, solution, rng, max_customers)
    raise ValueError(f"Unknown fixopt neighborhood: {neighborhood_type}")


def depot_neighborhood(
    instance: Instance,
    solution: Solution,
    rng: random.Random,
    max_customers: int,
) -> FixOptNeighborhood:
    depot_ids = sorted({route.depot_id for route in solution.routes})
    depot_id = rng.choice(depot_ids)
    released = [
        customer_id
        for route in solution.routes
        if route.depot_id == depot_id
        for customer_id in route.customer_ids
    ][:max_customers]
    candidate_depots = _nearby_depots(instance, depot_id, count=_candidate_depot_count(instance))
    return _make_neighborhood(solution, released, candidate_depots, {"type": "depot", "depot_id": depot_id})


def route_neighborhood(
    instance: Instance,
    solution: Solution,
    rng: random.Random,
    max_customers: int,
    max_routes: int,
) -> FixOptNeighborhood:
    route_indexes = list(range(len(solution.routes)))
    rng.shuffle(route_indexes)
    selected = route_indexes[: max(1, min(max_routes, len(route_indexes)))]
    released: list[int] = []
    candidate_depots: set[int] = set()
    for route_index in selected:
        route = solution.routes[route_index]
        released.extend(route.customer_ids)
        candidate_depots.add(route.depot_id)
        if len(released) >= max_customers:
            break
    for customer_id in released[:max_customers]:
        candidate_depots.update(_nearest_depots_for_customer(instance, customer_id, count=_candidate_depot_count(instance)))
    return _make_neighborhood(
        solution,
        released[:max_customers],
        sorted(candidate_depots),
        {"type": "route", "route_indexes": selected},
    )


def boundary_customer_neighborhood(
    instance: Instance,
    solution: Solution,
    rng: random.Random,
    max_customers: int,
) -> FixOptNeighborhood:
    assigned_depot = _assigned_depots(solution)
    scored: list[tuple[float, float, int]] = []
    for customer_id, current_depot in assigned_depot.items():
        current_cost = distance(instance, ("depot", current_depot), ("customer", customer_id))
        alternatives = [
            distance(instance, ("depot", depot.id), ("customer", customer_id))
            for depot in instance.depots
            if depot.id != current_depot
        ]
        if alternatives:
            scored.append((min(alternatives) - current_cost, rng.random(), customer_id))
    released = [customer_id for _, _, customer_id in sorted(scored)[:max_customers]]
    candidate_depots = sorted(
        {
            depot_id
            for customer_id in released
            for depot_id in [assigned_depot[customer_id], *_nearest_depots_for_customer(instance, customer_id, _candidate_depot_count(instance))]
        }
    )
    return _make_neighborhood(solution, released, candidate_depots, {"type": "boundary"})


def expensive_customer_neighborhood(
    instance: Instance,
    solution: Solution,
    max_customers: int,
) -> FixOptNeighborhood:
    # Rank customers by their own route's distance delta if removed, not by recomputing
    # the whole solution's objective_cost per customer (that was O(n) per customer, i.e.
    # O(n^2) overall -- confirmed to take ~21s at 3000 customers vs ~0.001s for the other
    # neighborhood builders). Removing one customer never changes route_fixed_cost or depot
    # opening cost (the route stays open), so the route-distance delta is the exact cost
    # delta, not just an approximation.
    scored: list[tuple[float, int]] = []
    for route in solution.routes:
        base_route_cost = route_distance(instance, route)
        for position, customer_id in enumerate(route.customer_ids):
            reduced_customers = route.customer_ids[:position] + route.customer_ids[position + 1 :]
            reduced_cost = route_distance(instance, Route(route.depot_id, reduced_customers))
            scored.append((base_route_cost - reduced_cost, customer_id))
    released = [customer_id for _, customer_id in sorted(scored, reverse=True)[:max_customers]]
    assigned = _assigned_depots(solution)
    candidate_depots = sorted(
        {
            depot_id
            for customer_id in released
            for depot_id in [assigned[customer_id], *_nearest_depots_for_customer(instance, customer_id, _candidate_depot_count(instance))]
        }
    )
    return _make_neighborhood(solution, released, candidate_depots, {"type": "expensive"})


def route_pair_neighborhood(
    instance: Instance,
    solution: Solution,
    rng: random.Random,
    max_customers: int,
) -> FixOptNeighborhood:
    if len(solution.routes) < 2:
        return route_neighborhood(instance, solution, rng, max_customers, 1)

    # Prefer a pair of routes from two different depots, same preference as before, but
    # without materializing and sorting the full O(routes^2) pair list (that took ~0.26s
    # at 579 routes and grows quadratically -- would dominate at instances with more routes).
    indices_by_depot: dict[int, list[int]] = {}
    for index, route in enumerate(solution.routes):
        indices_by_depot.setdefault(route.depot_id, []).append(index)

    if len(indices_by_depot) >= 2:
        first_depot, second_depot = rng.sample(sorted(indices_by_depot), 2)
        first = rng.choice(indices_by_depot[first_depot])
        second = rng.choice(indices_by_depot[second_depot])
    else:
        first, second = rng.sample(range(len(solution.routes)), 2)
    first, second = sorted((first, second))

    released = (solution.routes[first].customer_ids + solution.routes[second].customer_ids)[:max_customers]
    candidate_depots = sorted({solution.routes[first].depot_id, solution.routes[second].depot_id})
    for customer_id in released:
        candidate_depots = sorted(set(candidate_depots) | set(_nearest_depots_for_customer(instance, customer_id, _candidate_depot_count(instance))))
    return _make_neighborhood(
        solution,
        released,
        candidate_depots,
        {"type": "route_pair", "route_indexes": [first, second]},
    )


def _make_neighborhood(
    solution: Solution,
    released: list[int],
    candidate_depots: list[int],
    metadata: dict[str, Any],
) -> FixOptNeighborhood:
    unique_released = sorted(dict.fromkeys(released))
    fixed_routes = [
        Route(route.depot_id, [customer_id for customer_id in route.customer_ids if customer_id not in unique_released])
        for route in solution.routes
    ]
    fixed_routes = [route for route in fixed_routes if route.customer_ids]
    return FixOptNeighborhood(unique_released, sorted(dict.fromkeys(candidate_depots)), fixed_routes, metadata)


def _nearby_depots(instance: Instance, depot_id: int, count: int) -> list[int]:
    depot = instance.depots_by_id[depot_id]
    if depot.x is None or depot.y is None:
        return [item.id for item in sorted(instance.depots, key=lambda item: (item.id != depot_id, item.id))[:count]]
    return [
        item.id
        for item in sorted(
            instance.depots,
            key=lambda item: (
                distance(instance, ("depot", depot_id), ("depot", item.id)),
                item.id,
            ),
        )[:count]
    ]


def _nearest_depots_for_customer(instance: Instance, customer_id: int, count: int) -> list[int]:
    return [
        depot.id
        for depot in sorted(
            instance.depots,
            key=lambda depot: (
                distance(instance, ("depot", depot.id), ("customer", customer_id)),
                depot.id,
            ),
        )[:count]
    ]


def _assigned_depots(solution: Solution) -> dict[int, int]:
    return {
        customer_id: route.depot_id
        for route in solution.routes
        for customer_id in route.customer_ids
    }
