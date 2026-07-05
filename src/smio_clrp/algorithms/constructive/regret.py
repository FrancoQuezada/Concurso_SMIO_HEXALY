from __future__ import annotations

import time
from dataclasses import dataclass

from smio_clrp.algorithms.base import Solver, SolverConfig, SolverResult
from smio_clrp.algorithms.common import EPS, route_load
from smio_clrp.core.solution import Route, Solution
from smio_clrp.core.instance import Instance
from smio_clrp.evaluation.cost import objective_cost
from smio_clrp.evaluation.validator import validate_solution


@dataclass(frozen=True)
class InsertionOption:
    incremental_cost: float
    depot_id: int
    route_index: int | None
    position: int


class RegretInsertionSolver(Solver):
    algorithm_name = "regret"

    def solve(self, instance: Instance) -> SolverResult:
        start = time.perf_counter()
        regret_k = int(self.config.metadata.get("regret_k", 2))
        try:
            solution = regret_insertion(instance, regret_k=regret_k)
            validation = validate_solution(instance, solution)
            return SolverResult(
                solution=solution if validation.is_feasible else None,
                cost=validation.cost if validation.is_feasible else None,
                feasible=validation.is_feasible,
                runtime_seconds=time.perf_counter() - start,
                seed=self.config.seed,
                algorithm_name=self.algorithm_name,
                metadata={"regret_k": regret_k, "validation_errors": validation.errors},
            )
        except ValueError as exc:
            return SolverResult(
                solution=None,
                cost=None,
                feasible=False,
                runtime_seconds=time.perf_counter() - start,
                seed=self.config.seed,
                algorithm_name=self.algorithm_name,
                metadata={"error": str(exc), "regret_k": regret_k},
            )


def regret_insertion(instance: Instance, regret_k: int = 2) -> Solution:
    if regret_k not in {2, 3}:
        raise ValueError("regret_k must be 2 or 3")
    routes: list[Route] = []
    remaining = {customer.id for customer in instance.customers}

    while remaining:
        selected_customer, selected_option = _select_customer(instance, routes, remaining, regret_k)
        routes = _apply_insertion(routes, selected_customer, selected_option)
        remaining.remove(selected_customer)

    return Solution(instance_name=instance.name, routes=routes)


def _select_customer(
    instance: Instance,
    routes: list[Route],
    remaining: set[int],
    regret_k: int,
) -> tuple[int, InsertionOption]:
    best_choice: tuple[float, float, int, InsertionOption] | None = None
    for customer_id in sorted(remaining):
        options = sorted(
            _insertion_options(instance, routes, customer_id),
            key=lambda option: (
                option.incremental_cost,
                option.depot_id,
                -1 if option.route_index is None else option.route_index,
                option.position,
            ),
        )
        if not options:
            raise ValueError(f"No feasible insertion for customer {customer_id}")
        reference = options[min(regret_k, len(options)) - 1].incremental_cost
        regret = reference - options[0].incremental_cost
        choice = (regret, -options[0].incremental_cost, -customer_id, options[0])
        if best_choice is None or choice > best_choice:
            best_choice = choice
    assert best_choice is not None
    return -best_choice[2], best_choice[3]


def _insertion_options(instance: Instance, routes: list[Route], customer_id: int) -> list[InsertionOption]:
    customer = instance.customers_by_id[customer_id]
    if customer.demand > instance.vehicle_capacity + EPS:
        return []
    options: list[InsertionOption] = []
    depot_loads = _current_depot_loads(instance, routes)
    depot_route_counts = _current_depot_route_counts(routes)
    opened = {route.depot_id for route in routes}

    for route_index, route in enumerate(routes):
        depot = instance.depots_by_id[route.depot_id]
        if depot_loads[route.depot_id] + customer.demand > depot.capacity + EPS:
            continue
        if route_load(instance, route) + customer.demand > instance.vehicle_capacity + EPS:
            continue
        for position in range(len(route.customer_ids) + 1):
            options.append(
                InsertionOption(
                    incremental_cost=_route_insertion_delta(instance, route, customer_id, position),
                    depot_id=route.depot_id,
                    route_index=route_index,
                    position=position,
                )
            )

    for depot in sorted(instance.depots, key=lambda item: item.id):
        if depot_loads.get(depot.id, 0.0) + customer.demand > depot.capacity + EPS:
            continue
        if depot_route_counts.get(depot.id, 0) >= depot.vehicle_limit:
            continue
        route = Route(depot.id, [customer_id])
        incremental = objective_cost(instance, Solution(instance.name, [route]))
        if depot.id in opened:
            incremental -= depot.opening_cost
        options.append(InsertionOption(incremental, depot.id, None, 0))
    return options


def _apply_insertion(routes: list[Route], customer_id: int, option: InsertionOption) -> list[Route]:
    updated = [Route(route.depot_id, list(route.customer_ids)) for route in routes]
    if option.route_index is None:
        updated.append(Route(option.depot_id, [customer_id]))
        return updated
    route = updated[option.route_index]
    customers = list(route.customer_ids)
    customers.insert(option.position, customer_id)
    updated[option.route_index] = Route(route.depot_id, customers)
    return updated


def _route_insertion_delta(instance: Instance, route: Route, customer_id: int, position: int) -> float:
    before = objective_cost(instance, Solution(instance.name, [route]))
    customers = list(route.customer_ids)
    customers.insert(position, customer_id)
    after = objective_cost(instance, Solution(instance.name, [Route(route.depot_id, customers)]))
    return after - before


def _current_depot_loads(instance: Instance, routes: list[Route]) -> dict[int, float]:
    loads: dict[int, float] = {}
    for route in routes:
        loads[route.depot_id] = loads.get(route.depot_id, 0.0) + route_load(instance, route)
    return loads


def _current_depot_route_counts(routes: list[Route]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for route in routes:
        counts[route.depot_id] = counts.get(route.depot_id, 0) + 1
    return counts
