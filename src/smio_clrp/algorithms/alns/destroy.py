from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Callable

from smio_clrp.algorithms.alns.config import ALNSConfig
from smio_clrp.algorithms.alns.operators import DestroyResult
from smio_clrp.algorithms.common import EPS, route_load
from smio_clrp.core.distance import distance
from smio_clrp.core.instance import Instance
from smio_clrp.core.solution import Route, Solution
from smio_clrp.evaluation.cost import route_distance


DestroyOperator = Callable[[Instance, Solution, random.Random, ALNSConfig], DestroyResult]


def random_customer_removal(
    instance: Instance,
    solution: Solution,
    rng: random.Random,
    config: ALNSConfig,
) -> DestroyResult:
    customers = _all_customer_ids(solution)
    remove_count = _removal_count(len(customers), rng, config)
    removed = sorted(rng.sample(customers, remove_count))
    partial = _remove_customers(solution, set(removed))
    _assert_partial_valid(instance, partial)
    return DestroyResult(partial, removed, {"operator": "random_customer_removal"})


def worst_customer_removal(
    instance: Instance,
    solution: Solution,
    rng: random.Random,
    config: ALNSConfig,
) -> DestroyResult:
    # Score each customer by its own route's distance delta if removed, not by recomputing
    # the whole solution's objective_cost per customer (that was O(n) per customer, i.e.
    # O(n^2) overall, and intractable above ~100-150 customers).
    customers = _all_customer_ids(solution)
    remove_count = _removal_count(len(customers), rng, config)
    scored = []
    for route in solution.routes:
        base_route_cost = route_distance(instance, route)
        for position, customer_id in enumerate(route.customer_ids):
            reduced_customers = route.customer_ids[:position] + route.customer_ids[position + 1 :]
            reduced_cost = route_distance(instance, Route(route.depot_id, reduced_customers))
            saving = base_route_cost - reduced_cost
            scored.append((saving, rng.random(), customer_id))
    removed = [customer_id for _, _, customer_id in sorted(scored, reverse=True)[:remove_count]]
    partial = _remove_customers(solution, set(removed))
    _assert_partial_valid(instance, partial)
    return DestroyResult(partial, sorted(removed), {"operator": "worst_customer_removal"})


def shaw_related_removal(
    instance: Instance,
    solution: Solution,
    rng: random.Random,
    config: ALNSConfig,
) -> DestroyResult:
    customers = _all_customer_ids(solution)
    remove_count = _removal_count(len(customers), rng, config)
    seed_customer = rng.choice(customers)
    route_by_customer, depot_by_customer = _customer_locations(solution)
    scored = []
    for customer_id in customers:
        if customer_id == seed_customer:
            score = -1.0
        else:
            # Directed-compatible relation: use min(d_ij, d_ji) as closeness without assuming symmetry.
            distance_score = min(
                distance(instance, ("customer", seed_customer), ("customer", customer_id)),
                distance(instance, ("customer", customer_id), ("customer", seed_customer)),
            )
            demand_score = abs(
                instance.customers_by_id[seed_customer].demand
                - instance.customers_by_id[customer_id].demand
            )
            same_depot_bonus = -2.0 if depot_by_customer[seed_customer] == depot_by_customer[customer_id] else 0.0
            same_route_bonus = -3.0 if route_by_customer[seed_customer] == route_by_customer[customer_id] else 0.0
            score = distance_score + demand_score + same_depot_bonus + same_route_bonus
        scored.append((score, rng.random(), customer_id))
    removed = [customer_id for _, _, customer_id in sorted(scored)[:remove_count]]
    partial = _remove_customers(solution, set(removed))
    _assert_partial_valid(instance, partial)
    return DestroyResult(partial, sorted(removed), {"operator": "shaw_related_removal"})


def route_removal(
    instance: Instance,
    solution: Solution,
    rng: random.Random,
    config: ALNSConfig,
) -> DestroyResult:
    if not solution.routes:
        return DestroyResult(solution, [], {"operator": "route_removal"})
    remove_routes = max(1, min(len(solution.routes), math.ceil(len(solution.routes) * config.destroy_fraction_min)))
    indexes = set(rng.sample(range(len(solution.routes)), remove_routes))
    removed: list[int] = []
    routes: list[Route] = []
    for index, route in enumerate(solution.routes):
        if index in indexes:
            removed.extend(route.customer_ids)
        else:
            routes.append(Route(route.depot_id, list(route.customer_ids)))
    partial = Solution(solution.instance_name, routes)
    _assert_partial_valid(instance, partial)
    return DestroyResult(partial, sorted(removed), {"operator": "route_removal", "routes_removed": remove_routes})


def depot_removal(
    instance: Instance,
    solution: Solution,
    rng: random.Random,
    config: ALNSConfig,
) -> DestroyResult:
    depots = sorted({route.depot_id for route in solution.routes})
    if not depots:
        return DestroyResult(solution, [], {"operator": "depot_removal"})
    depot_id = rng.choice(depots)
    removed: list[int] = []
    routes: list[Route] = []
    for route in solution.routes:
        if route.depot_id == depot_id:
            removed.extend(route.customer_ids)
        else:
            routes.append(Route(route.depot_id, list(route.customer_ids)))
    partial = Solution(solution.instance_name, routes)
    _assert_partial_valid(instance, partial)
    return DestroyResult(partial, sorted(removed), {"operator": "depot_removal", "depot_id": depot_id})


DESTROY_OPERATORS = {
    "random": random_customer_removal,
    "worst": worst_customer_removal,
    "shaw": shaw_related_removal,
    "route": route_removal,
    "depot": depot_removal,
}


def _removal_count(customer_count: int, rng: random.Random, config: ALNSConfig) -> int:
    fraction = rng.uniform(config.destroy_fraction_min, config.destroy_fraction_max)
    return max(1, min(customer_count, math.ceil(customer_count * fraction)))


def _all_customer_ids(solution: Solution) -> list[int]:
    return [customer_id for route in solution.routes for customer_id in route.customer_ids]


def _remove_customers(solution: Solution, removed: set[int]) -> Solution:
    routes = []
    for route in solution.routes:
        customers = [customer_id for customer_id in route.customer_ids if customer_id not in removed]
        if customers:
            routes.append(Route(route.depot_id, customers))
    return Solution(solution.instance_name, routes)


def _customer_locations(solution: Solution) -> tuple[dict[int, int], dict[int, int]]:
    route_by_customer: dict[int, int] = {}
    depot_by_customer: dict[int, int] = {}
    for route_index, route in enumerate(solution.routes):
        for customer_id in route.customer_ids:
            route_by_customer[customer_id] = route_index
            depot_by_customer[customer_id] = route.depot_id
    return route_by_customer, depot_by_customer


def _assert_partial_valid(instance: Instance, solution: Solution) -> None:
    counts = Counter(_all_customer_ids(solution))
    duplicates = [customer_id for customer_id, count in counts.items() if count > 1]
    if duplicates:
        raise ValueError(f"Partial solution contains duplicate customers: {duplicates}")
    depot_loads: dict[int, float] = {}
    depot_route_counts: Counter[int] = Counter()
    for route in solution.routes:
        load = route_load(instance, route)
        if load > instance.vehicle_capacity + EPS:
            raise ValueError(f"Partial route exceeds vehicle capacity: depot {route.depot_id}")
        depot = instance.depots_by_id[route.depot_id]
        depot_loads[route.depot_id] = depot_loads.get(route.depot_id, 0.0) + load
        depot_route_counts[route.depot_id] += 1
        if depot_loads[route.depot_id] > depot.capacity + EPS:
            raise ValueError(f"Partial depot load exceeds capacity: depot {route.depot_id}")
        if depot_route_counts[route.depot_id] > depot.vehicle_limit:
            raise ValueError(f"Partial depot route count exceeds limit: depot {route.depot_id}")
