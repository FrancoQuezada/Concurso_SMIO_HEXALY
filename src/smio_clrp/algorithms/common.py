from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from typing import Iterable

from smio_clrp.core.instance import Instance
from smio_clrp.core.solution import Route, Solution
from smio_clrp.evaluation.cost import objective_cost, route_distance
from smio_clrp.evaluation.validator import validate_solution


EPS = 1e-9


def customer_demand(instance: Instance, customer_id: int) -> float:
    return instance.customers_by_id[customer_id].demand


def route_load(instance: Instance, route: Route) -> float:
    return sum(customer_demand(instance, customer_id) for customer_id in route.customer_ids)


def depot_loads(instance: Instance, routes: Iterable[Route]) -> dict[int, float]:
    loads: dict[int, float] = defaultdict(float)
    for route in routes:
        loads[route.depot_id] += route_load(instance, route)
    return dict(loads)


def depot_route_counts(routes: Iterable[Route]) -> Counter[int]:
    return Counter(route.depot_id for route in routes)


def solution_cost_or_inf(instance: Instance, solution: Solution) -> float:
    validation = validate_solution(instance, solution)
    if not validation.is_feasible:
        return float("inf")
    return objective_cost(instance, solution)


def clone_solution(solution: Solution) -> Solution:
    return Solution(
        instance_name=solution.instance_name,
        routes=[Route(route.depot_id, list(route.customer_ids)) for route in solution.routes],
        reported_cost=solution.reported_cost,
    )


def replace_route(routes: list[Route], index: int, route: Route) -> list[Route]:
    updated = list(routes)
    updated[index] = route
    return [route for route in updated if route.customer_ids]


def route_with_customers(route: Route, customer_ids: list[int]) -> Route:
    return replace(route, customer_ids=list(customer_ids))


def insertion_delta(instance: Instance, route: Route, customer_id: int, position: int) -> float:
    before = route_distance(instance, route)
    customers = list(route.customer_ids)
    customers.insert(position, customer_id)
    after = route_distance(instance, Route(route.depot_id, customers))
    return after - before


def assert_feasible(instance: Instance, solution: Solution) -> Solution:
    validation = validate_solution(instance, solution)
    if not validation.is_feasible:
        raise ValueError("; ".join(validation.errors))
    return solution
