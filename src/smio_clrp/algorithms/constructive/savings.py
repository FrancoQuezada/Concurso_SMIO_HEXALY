from __future__ import annotations

import time
from dataclasses import dataclass, field

from smio_clrp.algorithms.base import Solver, SolverConfig, SolverResult
from smio_clrp.algorithms.common import EPS, route_load
from smio_clrp.core.distance import distance
from smio_clrp.core.instance import Instance
from smio_clrp.core.solution import Route, Solution
from smio_clrp.evaluation.cost import objective_cost, route_distance
from smio_clrp.evaluation.validator import validate_solution


@dataclass
class _DepotAssignment:
    customers: dict[int, list[int]] = field(default_factory=dict)
    loads: dict[int, float] = field(default_factory=dict)


class SavingsConstructiveSolver(Solver):
    algorithm_name = "savings"

    def solve(self, instance: Instance) -> SolverResult:
        start = time.perf_counter()
        try:
            solution = savings_routes_per_depot(instance)
            validation = validate_solution(instance, solution)
            return SolverResult(
                solution=solution if validation.is_feasible else None,
                cost=validation.cost if validation.is_feasible else None,
                feasible=validation.is_feasible,
                runtime_seconds=time.perf_counter() - start,
                seed=self.config.seed,
                algorithm_name=self.algorithm_name,
                metadata={"validation_errors": validation.errors},
            )
        except ValueError as exc:
            return SolverResult(
                solution=None,
                cost=None,
                feasible=False,
                runtime_seconds=time.perf_counter() - start,
                seed=self.config.seed,
                algorithm_name=self.algorithm_name,
                metadata={"error": str(exc)},
            )


def savings_routes_per_depot(instance: Instance) -> Solution:
    assignment = _assign_customers_to_depots(instance)
    routes: list[Route] = []
    for depot in sorted(instance.depots, key=lambda item: item.id):
        customers = assignment.customers.get(depot.id, [])
        if not customers:
            continue
        depot_routes = _build_savings_routes(instance, depot.id, customers)
        if len(depot_routes) > depot.vehicle_limit:
            raise ValueError(
                f"Depot {depot.id} requires {len(depot_routes)} routes, limit is {depot.vehicle_limit}"
            )
        routes.extend(depot_routes)
    return Solution(instance_name=instance.name, routes=routes)


def _assign_customers_to_depots(instance: Instance) -> _DepotAssignment:
    assigned = _DepotAssignment(
        customers={depot.id: [] for depot in instance.depots},
        loads={depot.id: 0.0 for depot in instance.depots},
    )
    customers = sorted(instance.customers, key=lambda customer: (-customer.demand, customer.id))
    for customer in customers:
        if customer.demand > instance.vehicle_capacity + EPS:
            raise ValueError(f"Customer {customer.id} demand exceeds vehicle capacity")
        candidates: list[tuple[float, int]] = []
        for depot in sorted(instance.depots, key=lambda item: item.id):
            new_load = assigned.loads[depot.id] + customer.demand
            if new_load > depot.capacity + EPS:
                continue
            minimum_routes = _minimum_routes(instance, assigned.customers[depot.id] + [customer.id])
            if minimum_routes > depot.vehicle_limit:
                continue
            score = (
                distance(instance, ("depot", depot.id), ("customer", customer.id))
                + distance(instance, ("customer", customer.id), ("depot", depot.id))
                + (0.0 if assigned.customers[depot.id] else depot.opening_cost)
            )
            candidates.append((score, depot.id))
        if not candidates:
            raise ValueError(f"No feasible depot assignment for customer {customer.id}")
        _, depot_id = min(candidates, key=lambda item: (item[0], item[1]))
        assigned.customers[depot_id].append(customer.id)
        assigned.loads[depot_id] += customer.demand
    return assigned


def _minimum_routes(instance: Instance, customer_ids: list[int]) -> int:
    total_demand = sum(instance.customers_by_id[customer_id].demand for customer_id in customer_ids)
    return int(-(-total_demand // instance.vehicle_capacity))


def _build_savings_routes(instance: Instance, depot_id: int, customer_ids: list[int]) -> list[Route]:
    routes = [Route(depot_id, [customer_id]) for customer_id in sorted(customer_ids)]
    while True:
        merge = _best_merge(instance, routes)
        if merge is None:
            break
        first, second, merged_route, saving = merge
        depot = instance.depots_by_id[depot_id]
        must_reduce_routes = len(routes) > depot.vehicle_limit
        if saving <= EPS and not must_reduce_routes:
            break
        routes = [
            route
            for index, route in enumerate(routes)
            if index not in {first, second}
        ]
        routes.append(merged_route)
        routes.sort(key=lambda route: (route.customer_ids[0], len(route.customer_ids)))
    return routes


def _best_merge(instance: Instance, routes: list[Route]) -> tuple[int, int, Route, float] | None:
    best: tuple[int, int, Route, float] | None = None
    for first_index, first in enumerate(routes):
        for second_index, second in enumerate(routes):
            if first_index >= second_index:
                continue
            if route_load(instance, first) + route_load(instance, second) > instance.vehicle_capacity + EPS:
                continue
            for customers in (
                first.customer_ids + second.customer_ids,
                second.customer_ids + first.customer_ids,
                list(reversed(first.customer_ids)) + second.customer_ids,
                first.customer_ids + list(reversed(second.customer_ids)),
            ):
                merged = Route(first.depot_id, customers)
                before = route_distance(instance, first) + route_distance(instance, second)
                after = route_distance(instance, merged)
                saving = before - after
                candidate = (first_index, second_index, merged, saving)
                if best is None or (saving, -after) > (best[3], -route_distance(instance, best[2])):
                    best = candidate
    return best
