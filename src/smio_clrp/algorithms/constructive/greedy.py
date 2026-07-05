from __future__ import annotations

import time
from dataclasses import dataclass, field

from smio_clrp.algorithms.base import Solver, SolverConfig, SolverResult
from smio_clrp.core.distance import distance
from smio_clrp.core.instance import Instance
from smio_clrp.core.solution import Route, Solution
from smio_clrp.evaluation.cost import objective_cost
from smio_clrp.evaluation.validator import validate_solution


@dataclass
class _WorkingRoute:
    depot_id: int
    customer_ids: list[int] = field(default_factory=list)
    load: float = 0.0


class GreedyNearestDepotSolver(Solver):
    algorithm_name = "greedy_nearest_depot"

    def solve(self, instance: Instance) -> SolverResult:
        start = time.perf_counter()
        try:
            routes = _build_routes(instance)
            solution = Solution(instance_name=instance.name, routes=routes)
            validation = validate_solution(instance, solution)
            cost = objective_cost(instance, solution) if validation.is_feasible else None
            return SolverResult(
                solution=solution,
                cost=cost,
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


def solve_greedy_nearest_depot(instance: Instance, seed: int = 1) -> SolverResult:
    return GreedyNearestDepotSolver(SolverConfig(seed=seed)).solve(instance)


def _build_routes(instance: Instance) -> list[Route]:
    remaining_depot_capacity = {depot.id: depot.capacity for depot in instance.depots}
    working_routes: dict[int, list[_WorkingRoute]] = {depot.id: [] for depot in instance.depots}
    customer_order = sorted(instance.customers, key=lambda customer: (-customer.demand, customer.id))

    for customer in customer_order:
        candidates: list[tuple[float, int, int | None]] = []
        for depot in sorted(instance.depots, key=lambda item: item.id):
            if remaining_depot_capacity[depot.id] + 1e-9 < customer.demand:
                continue
            for route_index, route in enumerate(working_routes[depot.id]):
                if route.load + customer.demand <= instance.vehicle_capacity + 1e-9:
                    incremental = _append_increment(instance, route, customer.id)
                    candidates.append((incremental, depot.id, route_index))
            if len(working_routes[depot.id]) < depot.vehicle_limit:
                depot_round_trip = (
                    distance(instance, ("depot", depot.id), ("customer", customer.id))
                    + distance(instance, ("customer", customer.id), ("depot", depot.id))
                    + instance.route_fixed_cost
                )
                candidates.append((depot_round_trip, depot.id, None))

        if not candidates:
            raise ValueError(f"No feasible depot/route assignment for customer {customer.id}")

        _, depot_id, route_index = min(candidates, key=lambda item: (item[0], item[1], item[2] or -1))
        remaining_depot_capacity[depot_id] -= customer.demand
        if route_index is None:
            working_routes[depot_id].append(
                _WorkingRoute(depot_id=depot_id, customer_ids=[customer.id], load=customer.demand)
            )
        else:
            route = working_routes[depot_id][route_index]
            route.customer_ids.append(customer.id)
            route.load += customer.demand

    final_routes: list[Route] = []
    for depot_id in sorted(working_routes):
        for route in working_routes[depot_id]:
            ordered = _nearest_neighbor_order(instance, depot_id, route.customer_ids)
            final_routes.append(Route(depot_id=depot_id, customer_ids=ordered))
    return final_routes


def _append_increment(instance: Instance, route: _WorkingRoute, customer_id: int) -> float:
    if not route.customer_ids:
        return distance(instance, ("depot", route.depot_id), ("customer", customer_id))
    last = route.customer_ids[-1]
    old_return = distance(instance, ("customer", last), ("depot", route.depot_id))
    new_leg = distance(instance, ("customer", last), ("customer", customer_id))
    new_return = distance(instance, ("customer", customer_id), ("depot", route.depot_id))
    return new_leg + new_return - old_return


def _nearest_neighbor_order(instance: Instance, depot_id: int, customer_ids: list[int]) -> list[int]:
    remaining = set(customer_ids)
    ordered: list[int] = []
    current = ("depot", depot_id)
    while remaining:
        next_customer = min(
            remaining,
            key=lambda customer_id: (
                distance(instance, current, ("customer", customer_id)),
                customer_id,
            ),
        )
        ordered.append(next_customer)
        remaining.remove(next_customer)
        current = ("customer", next_customer)
    return ordered
