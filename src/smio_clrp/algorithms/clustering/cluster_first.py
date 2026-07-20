from __future__ import annotations

import time
from dataclasses import dataclass, field

from smio_clrp.algorithms.base import Solver, SolverResult
from smio_clrp.core.distance import distance
from smio_clrp.core.instance import Instance
from smio_clrp.core.solution import Route, Solution
from smio_clrp.evaluation.cost import objective_cost
from smio_clrp.evaluation.validator import validate_solution


@dataclass
class _ClusterRoute:
    customer_ids: list[int] = field(default_factory=list)
    load: float = 0.0


class ClusterFirstSolver(Solver):
    algorithm_name = "cluster_first"

    def solve(self, instance: Instance) -> SolverResult:
        start = time.perf_counter()
        try:
            routes = _cluster_first_routes(instance)
            solution = Solution(instance.name, routes)
            validation = validate_solution(instance, solution)
            return SolverResult(
                solution if validation.is_feasible else None,
                objective_cost(instance, solution) if validation.is_feasible else None,
                validation.is_feasible,
                time.perf_counter() - start,
                self.config.seed,
                self.algorithm_name,
                {"validation_errors": validation.errors},
            )
        except ValueError as exc:
            return SolverResult(None, None, False, time.perf_counter() - start, self.config.seed, self.algorithm_name, {"error": str(exc)})


def _cluster_first_routes(instance: Instance) -> list[Route]:
    clusters: dict[int, list[_ClusterRoute]] = {depot.id: [] for depot in instance.depots}
    depot_loads = {depot.id: 0.0 for depot in instance.depots}
    customers = sorted(instance.customers, key=lambda item: (-item.demand, item.id))

    for customer in customers:
        candidates: list[tuple[float, float, int, int | None]] = []
        for depot in instance.depots:
            effective_capacity = min(depot.capacity, depot.vehicle_limit * instance.vehicle_capacity)
            if depot_loads[depot.id] + customer.demand > effective_capacity + 1e-9:
                continue
            depot_distance = distance(instance, ("depot", depot.id), ("customer", customer.id))
            for route_index, route in enumerate(clusters[depot.id]):
                if route.load + customer.demand <= instance.vehicle_capacity + 1e-9:
                    candidates.append((depot_distance, route.load, depot.id, route_index))
            if len(clusters[depot.id]) < depot.vehicle_limit:
                candidates.append((depot_distance, 0.0, depot.id, None))
        if not candidates:
            raise ValueError(f"No feasible cluster assignment for customer {customer.id}")

        _, _, depot_id, route_index = min(
            candidates,
            key=lambda item: (item[0], -item[1], item[2], -1 if item[3] is None else item[3]),
        )
        depot_loads[depot_id] += customer.demand
        if route_index is None:
            clusters[depot_id].append(_ClusterRoute([customer.id], customer.demand))
        else:
            route = clusters[depot_id][route_index]
            route.customer_ids.append(customer.id)
            route.load += customer.demand

    result: list[Route] = []
    for depot_id, depot_routes in sorted(clusters.items()):
        for route in depot_routes:
            result.append(Route(depot_id, _nearest_neighbor_order(instance, depot_id, route.customer_ids)))
    return result


def _nearest_neighbor_order(instance: Instance, depot_id: int, customer_ids: list[int]) -> list[int]:
    remaining = set(customer_ids)
    ordered: list[int] = []
    current = ("depot", depot_id)
    while remaining:
        customer_id = min(
            remaining,
            key=lambda item: (distance(instance, current, ("customer", item)), item),
        )
        ordered.append(customer_id)
        remaining.remove(customer_id)
        current = ("customer", customer_id)
    return ordered
