from __future__ import annotations

import random
from dataclasses import dataclass

from smio_clrp.algorithms.alns.operators import RepairResult
from smio_clrp.algorithms.common import EPS, depot_loads, depot_route_counts, insertion_delta, route_load
from smio_clrp.core.instance import Instance
from smio_clrp.core.solution import Route, Solution
from smio_clrp.evaluation.cost import route_distance
from smio_clrp.evaluation.validator import validate_solution


@dataclass(frozen=True)
class InsertionMove:
    customer_id: int
    incremental_cost: float
    depot_id: int
    route_index: int | None
    position: int


def greedy_repair(
    instance: Instance,
    partial_solution: Solution,
    removed_customer_ids: list[int],
    rng: random.Random,
) -> RepairResult:
    return _repair(instance, partial_solution, removed_customer_ids, rng, regret_k=1, noise=0.0)


def regret2_repair(
    instance: Instance,
    partial_solution: Solution,
    removed_customer_ids: list[int],
    rng: random.Random,
) -> RepairResult:
    return _repair(instance, partial_solution, removed_customer_ids, rng, regret_k=2, noise=0.0)


def regret3_repair(
    instance: Instance,
    partial_solution: Solution,
    removed_customer_ids: list[int],
    rng: random.Random,
) -> RepairResult:
    return _repair(instance, partial_solution, removed_customer_ids, rng, regret_k=3, noise=0.0)


def noise_repair(
    instance: Instance,
    partial_solution: Solution,
    removed_customer_ids: list[int],
    rng: random.Random,
) -> RepairResult:
    return _repair(instance, partial_solution, removed_customer_ids, rng, regret_k=2, noise=0.05)


REPAIR_OPERATORS = {
    "greedy": greedy_repair,
    "regret2": regret2_repair,
    "regret3": regret3_repair,
    "noise": noise_repair,
}


def _repair(
    instance: Instance,
    partial_solution: Solution,
    removed_customer_ids: list[int],
    rng: random.Random,
    regret_k: int,
    noise: float,
) -> RepairResult:
    routes = [Route(route.depot_id, list(route.customer_ids)) for route in partial_solution.routes]
    remaining = sorted(set(removed_customer_ids))

    # Insertion order is computed once, from regret scores against the initial partial
    # solution, instead of recomputing every remaining customer's options after each single
    # insertion (that made repair O(removed^2) calls to _insertion_options -- the dominant
    # cost once each call itself stopped being O(n) per candidate; measured ~7-10s to repair
    # ~76 customers on a 350-customer instance before this change). Each customer's actual
    # insertion spot is still freshly computed against the current routes when it's placed,
    # so feasibility is exact -- only the *priority order* is based on a one-time snapshot.
    try:
        order = _insertion_order(instance, routes, remaining, rng, regret_k, noise)
        for customer_id in order:
            options = _insertion_options(instance, routes, customer_id, rng, noise)
            if not options:
                raise ValueError(f"No feasible insertion for customer {customer_id}")
            options.sort(key=lambda move: (move.incremental_cost, move.depot_id, move.route_index if move.route_index is not None else -1, move.position))
            routes = _apply_move(routes, options[0])
    except ValueError as exc:
        return RepairResult(None, False, {"error": str(exc), "remaining": remaining})

    solution = Solution(partial_solution.instance_name, routes)
    validation = validate_solution(instance, solution)
    if not validation.is_feasible:
        return RepairResult(None, False, {"error": "; ".join(validation.errors)})
    return RepairResult(solution, True, {"cost": validation.cost})


def _insertion_order(
    instance: Instance,
    routes: list[Route],
    remaining: list[int],
    rng: random.Random,
    regret_k: int,
    noise: float,
) -> list[int]:
    scored: list[tuple[float, float, int]] = []
    for customer_id in remaining:
        options = _insertion_options(instance, routes, customer_id, rng, noise)
        if not options:
            raise ValueError(f"No feasible insertion for customer {customer_id}")
        options.sort(key=lambda move: (move.incremental_cost, move.depot_id, move.route_index if move.route_index is not None else -1, move.position))
        if regret_k <= 1:
            regret = -options[0].incremental_cost
        else:
            reference = options[min(regret_k, len(options)) - 1].incremental_cost
            regret = reference - options[0].incremental_cost
        scored.append((-regret, options[0].incremental_cost, customer_id))
    scored.sort()
    return [customer_id for _, _, customer_id in scored]


def _insertion_options(
    instance: Instance,
    routes: list[Route],
    customer_id: int,
    rng: random.Random,
    noise: float,
) -> list[InsertionMove]:
    """All feasible insertion spots for one customer, scored with route-local distance
    deltas instead of a full-solution objective_cost per candidate (that was O(n) per
    candidate -- called for every remaining customer on every repair step, making a single
    repair O(removed^2 * n) or worse and intractable above ~100-150 customers)."""
    customer = instance.customers_by_id[customer_id]
    if customer.demand > instance.vehicle_capacity + EPS:
        return []
    opened_depot_ids = {route.depot_id for route in routes}
    loads = depot_loads(instance, routes)
    counts = depot_route_counts(routes)
    options: list[InsertionMove] = []

    for route_index, route in enumerate(routes):
        if route_load(instance, route) + customer.demand > instance.vehicle_capacity + EPS:
            continue
        depot = instance.depots_by_id[route.depot_id]
        if loads.get(route.depot_id, 0.0) + customer.demand > depot.capacity + EPS:
            continue
        for position in range(len(route.customer_ids) + 1):
            incremental = insertion_delta(instance, route, customer_id, position)
            options.append(
                InsertionMove(customer_id, _with_noise(incremental, rng, noise), route.depot_id, route_index, position)
            )

    for depot in sorted(instance.depots, key=lambda item: item.id):
        if loads.get(depot.id, 0.0) + customer.demand > depot.capacity + EPS:
            continue
        if counts.get(depot.id, 0) + 1 > depot.vehicle_limit:
            continue
        opening_cost = 0.0 if depot.id in opened_depot_ids else depot.opening_cost
        incremental = route_distance(instance, Route(depot.id, [customer_id])) + instance.route_fixed_cost + opening_cost
        options.append(InsertionMove(customer_id, _with_noise(incremental, rng, noise), depot.id, None, 0))
    return options


def _apply_move(routes: list[Route], move: InsertionMove) -> list[Route]:
    updated = [Route(route.depot_id, list(route.customer_ids)) for route in routes]
    if move.route_index is None:
        updated.append(Route(move.depot_id, [move.customer_id]))
        return updated
    customers = list(updated[move.route_index].customer_ids)
    customers.insert(move.position, move.customer_id)
    updated[move.route_index] = Route(move.depot_id, customers)
    return updated


def _with_noise(cost: float, rng: random.Random, noise: float) -> float:
    if noise <= 0:
        return cost
    return cost * (1.0 + rng.uniform(-noise, noise))
