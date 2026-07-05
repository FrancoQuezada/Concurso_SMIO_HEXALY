from __future__ import annotations

import random
from dataclasses import dataclass

from smio_clrp.algorithms.alns.operators import RepairResult
from smio_clrp.algorithms.common import EPS, route_load
from smio_clrp.core.instance import Instance
from smio_clrp.core.solution import Route, Solution
from smio_clrp.evaluation.cost import objective_cost
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
    try:
        while remaining:
            customer_id, move = _select_insertion(instance, routes, remaining, rng, regret_k, noise)
            routes = _apply_move(routes, move)
            remaining.remove(customer_id)
    except ValueError as exc:
        return RepairResult(None, False, {"error": str(exc), "remaining": remaining})

    solution = Solution(partial_solution.instance_name, routes)
    validation = validate_solution(instance, solution)
    if not validation.is_feasible:
        return RepairResult(None, False, {"error": "; ".join(validation.errors)})
    return RepairResult(solution, True, {"cost": validation.cost})


def _select_insertion(
    instance: Instance,
    routes: list[Route],
    remaining: list[int],
    rng: random.Random,
    regret_k: int,
    noise: float,
) -> tuple[int, InsertionMove]:
    best_choice: tuple[float, float, int, InsertionMove] | None = None
    for customer_id in remaining:
        options = _insertion_options(instance, routes, customer_id, rng, noise)
        if not options:
            raise ValueError(f"No feasible insertion for customer {customer_id}")
        options.sort(key=lambda move: (move.incremental_cost, move.depot_id, move.route_index or -1, move.position))
        if regret_k <= 1:
            regret = -options[0].incremental_cost
        else:
            reference = options[min(regret_k, len(options)) - 1].incremental_cost
            regret = reference - options[0].incremental_cost
        choice = (regret, -options[0].incremental_cost, -customer_id, options[0])
        if best_choice is None or choice > best_choice:
            best_choice = choice
    assert best_choice is not None
    return -best_choice[2], best_choice[3]


def _insertion_options(
    instance: Instance,
    routes: list[Route],
    customer_id: int,
    rng: random.Random,
    noise: float,
) -> list[InsertionMove]:
    customer = instance.customers_by_id[customer_id]
    if customer.demand > instance.vehicle_capacity + EPS:
        return []
    base = Solution(instance.name, routes)
    base_cost = objective_cost(instance, base)
    options: list[InsertionMove] = []

    for route_index, route in enumerate(routes):
        if route_load(instance, route) + customer.demand > instance.vehicle_capacity + EPS:
            continue
        for position in range(len(route.customer_ids) + 1):
            candidate_routes = [Route(item.depot_id, list(item.customer_ids)) for item in routes]
            customers = list(candidate_routes[route_index].customer_ids)
            customers.insert(position, customer_id)
            candidate_routes[route_index] = Route(route.depot_id, customers)
            if not _partial_constraints_ok(instance, candidate_routes):
                continue
            incremental = objective_cost(instance, Solution(instance.name, candidate_routes)) - base_cost
            options.append(InsertionMove(customer_id, _with_noise(incremental, rng, noise), route.depot_id, route_index, position))

    for depot in sorted(instance.depots, key=lambda item: item.id):
        candidate_routes = [Route(item.depot_id, list(item.customer_ids)) for item in routes]
        candidate_routes.append(Route(depot.id, [customer_id]))
        if not _partial_constraints_ok(instance, candidate_routes):
            continue
        incremental = objective_cost(instance, Solution(instance.name, candidate_routes)) - base_cost
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


def _partial_constraints_ok(instance: Instance, routes: list[Route]) -> bool:
    depot_loads: dict[int, float] = {}
    depot_route_counts: dict[int, int] = {}
    for route in routes:
        load = route_load(instance, route)
        if load > instance.vehicle_capacity + EPS:
            return False
        depot = instance.depots_by_id[route.depot_id]
        depot_loads[route.depot_id] = depot_loads.get(route.depot_id, 0.0) + load
        depot_route_counts[route.depot_id] = depot_route_counts.get(route.depot_id, 0) + 1
        if depot_loads[route.depot_id] > depot.capacity + EPS:
            return False
        if depot_route_counts[route.depot_id] > depot.vehicle_limit:
            return False
    return True
