from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from smio_clrp.algorithms.common import route_load
from smio_clrp.core.distance import distance
from smio_clrp.core.instance import Instance
from smio_clrp.core.solution import Route, Solution
from smio_clrp.evaluation.cost import objective_cost


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
    candidate_depots = _nearby_depots(instance, depot_id, count=3)
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
        candidate_depots.update(_nearest_depots_for_customer(instance, customer_id, count=2))
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
            for depot_id in [assigned_depot[customer_id], *_nearest_depots_for_customer(instance, customer_id, 2)]
        }
    )
    return _make_neighborhood(solution, released, candidate_depots, {"type": "boundary"})


def expensive_customer_neighborhood(
    instance: Instance,
    solution: Solution,
    max_customers: int,
) -> FixOptNeighborhood:
    base_cost = objective_cost(instance, solution)
    scored: list[tuple[float, int]] = []
    for route in solution.routes:
        for customer_id in route.customer_ids:
            reduced = _remove_customers(solution, {customer_id})
            scored.append((base_cost - objective_cost(instance, reduced), customer_id))
    released = [customer_id for _, customer_id in sorted(scored, reverse=True)[:max_customers]]
    assigned = _assigned_depots(solution)
    candidate_depots = sorted(
        {
            depot_id
            for customer_id in released
            for depot_id in [assigned[customer_id], *_nearest_depots_for_customer(instance, customer_id, 2)]
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
    pairs = [
        (first, second)
        for first in range(len(solution.routes))
        for second in range(first + 1, len(solution.routes))
    ]
    pairs.sort(
        key=lambda pair: (
            solution.routes[pair[0]].depot_id == solution.routes[pair[1]].depot_id,
            rng.random(),
        )
    )
    first, second = pairs[0]
    released = (solution.routes[first].customer_ids + solution.routes[second].customer_ids)[:max_customers]
    candidate_depots = sorted({solution.routes[first].depot_id, solution.routes[second].depot_id})
    for customer_id in released:
        candidate_depots = sorted(set(candidate_depots) | set(_nearest_depots_for_customer(instance, customer_id, 2)))
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


def _remove_customers(solution: Solution, removed: set[int]) -> Solution:
    return Solution(
        solution.instance_name,
        [
            Route(route.depot_id, [customer_id for customer_id in route.customer_ids if customer_id not in removed])
            for route in solution.routes
        ],
    )
