from __future__ import annotations

import time
from collections.abc import Callable

from smio_clrp.core.instance import Instance
from smio_clrp.core.solution import Solution
from smio_clrp.evaluation.cost import objective_cost
from smio_clrp.evaluation.validator import validate_solution
from smio_clrp.algorithms.local_search.relocate import relocate_customer
from smio_clrp.algorithms.local_search.route_reinsertion import route_reinsertion
from smio_clrp.algorithms.local_search.swap import swap_customers
from smio_clrp.algorithms.local_search.two_opt import improve_solution_two_opt
from smio_clrp.algorithms.local_search.two_opt_star import inter_route_two_opt_star
from smio_clrp.algorithms.local_search.depot_reassignment import reassign_route_depot


Operator = Callable[[Instance, Solution], Solution]

OPERATORS: dict[str, Operator] = {
    "two_opt": improve_solution_two_opt,
    "relocate": relocate_customer,
    "swap": swap_customers,
    "route_reinsertion": route_reinsertion,
    "two_opt_star": inter_route_two_opt_star,
    "route_depot_reassignment": reassign_route_depot,
}


def improve_solution(
    instance: Instance,
    solution: Solution,
    operators: list[str] | None = None,
    max_iterations: int = 50,
    time_limit_seconds: float | None = None,
) -> Solution:
    validation = validate_solution(instance, solution)
    if not validation.is_feasible:
        raise ValueError("Local search requires a feasible input solution")

    selected = operators or list(OPERATORS)
    current = solution
    current_cost = objective_cost(instance, current)
    start = time.perf_counter()

    for _ in range(max_iterations):
        if time_limit_seconds is not None and time.perf_counter() - start >= time_limit_seconds:
            break
        improved = False
        for operator_name in selected:
            operator = OPERATORS[operator_name]
            candidate = operator(instance, current)
            candidate_validation = validate_solution(instance, candidate)
            if not candidate_validation.is_feasible:
                continue
            candidate_cost = objective_cost(instance, candidate)
            if candidate_cost + 1e-9 < current_cost:
                current = candidate
                current_cost = candidate_cost
                improved = True
                break
        if not improved:
            break
    return current
