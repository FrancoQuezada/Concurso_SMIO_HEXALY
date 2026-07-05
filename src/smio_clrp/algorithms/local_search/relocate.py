from __future__ import annotations

from smio_clrp.core.instance import Instance
from smio_clrp.core.solution import Route, Solution
from smio_clrp.evaluation.cost import objective_cost
from smio_clrp.evaluation.validator import validate_solution


def relocate_customers(instance: Instance, solution: Solution) -> Solution:
    return relocate_customer(instance, solution)


def relocate_customer(instance: Instance, solution: Solution) -> Solution:
    best = solution
    best_cost = objective_cost(instance, solution)
    routes = solution.routes
    for source_index, source in enumerate(routes):
        for source_position, customer_id in enumerate(source.customer_ids):
            without_customer = _remove_customer(routes, source_index, source_position)
            for target_index, target in enumerate(without_customer):
                for target_position in range(len(target.customer_ids) + 1):
                    candidate_routes = _insert_customer(
                        without_customer,
                        target_index,
                        target_position,
                        customer_id,
                    )
                    best, best_cost = _accept_if_better(instance, solution, candidate_routes, best, best_cost)
            for depot in instance.depots:
                candidate_routes = without_customer + [Route(depot.id, [customer_id])]
                best, best_cost = _accept_if_better(instance, solution, candidate_routes, best, best_cost)
    return best


def _remove_customer(routes: list[Route], route_index: int, position: int) -> list[Route]:
    updated: list[Route] = []
    for index, route in enumerate(routes):
        customers = list(route.customer_ids)
        if index == route_index:
            customers.pop(position)
        if customers:
            updated.append(Route(route.depot_id, customers))
    return updated


def _insert_customer(
    routes: list[Route],
    route_index: int,
    position: int,
    customer_id: int,
) -> list[Route]:
    updated: list[Route] = []
    for index, route in enumerate(routes):
        customers = list(route.customer_ids)
        if index == route_index:
            customers.insert(position, customer_id)
        updated.append(Route(route.depot_id, customers))
    return updated


def _accept_if_better(
    instance: Instance,
    base: Solution,
    candidate_routes: list[Route],
    best: Solution,
    best_cost: float,
) -> tuple[Solution, float]:
    candidate = Solution(base.instance_name, candidate_routes)
    validation = validate_solution(instance, candidate)
    if not validation.is_feasible:
        return best, best_cost
    cost = objective_cost(instance, candidate)
    if cost + 1e-9 < best_cost:
        return candidate, cost
    return best, best_cost
