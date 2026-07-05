from __future__ import annotations

import random

from smio_clrp.algorithms.common import EPS, route_load
from smio_clrp.algorithms.fixopt.backend import FixOptResult
from smio_clrp.algorithms.fixopt.config import FixOptConfig
from smio_clrp.algorithms.fixopt.neighborhoods import FixOptNeighborhood
from smio_clrp.algorithms.local_search.driver import improve_solution
from smio_clrp.core.instance import Instance
from smio_clrp.core.solution import Route, Solution
from smio_clrp.evaluation.cost import objective_cost
from smio_clrp.evaluation.validator import validate_solution


class HeuristicFixOptBackend:
    backend_name = "heuristic"

    def __init__(self, seed: int = 1) -> None:
        self.rng = random.Random(seed)

    def reoptimize(
        self,
        instance: Instance,
        solution: Solution,
        neighborhood: FixOptNeighborhood,
        config: FixOptConfig,
    ) -> FixOptResult:
        routes = [
            Route(route.depot_id, [customer_id for customer_id in route.customer_ids if customer_id not in neighborhood.released_customer_ids])
            for route in solution.routes
        ]
        routes = [route for route in routes if route.customer_ids]
        remaining = sorted(neighborhood.released_customer_ids, key=lambda item: (-instance.customers_by_id[item].demand, item))
        try:
            for customer_id in remaining:
                routes = self._insert_customer(instance, routes, customer_id, neighborhood.candidate_depot_ids)
        except ValueError as exc:
            return FixOptResult(None, False, self.backend_name, {"error": str(exc)})

        candidate = Solution(solution.instance_name, routes)
        validation = validate_solution(instance, candidate)
        if not validation.is_feasible:
            return FixOptResult(None, False, self.backend_name, {"error": "; ".join(validation.errors)})
        if config.local_search_after_subproblem:
            candidate = improve_solution(instance, candidate, max_iterations=5)
            validation = validate_solution(instance, candidate)
            if not validation.is_feasible:
                return FixOptResult(None, False, self.backend_name, {"error": "; ".join(validation.errors)})
        return FixOptResult(candidate, True, self.backend_name, {"cost": objective_cost(instance, candidate)})

    def _insert_customer(
        self,
        instance: Instance,
        routes: list[Route],
        customer_id: int,
        candidate_depot_ids: list[int],
    ) -> list[Route]:
        best_routes: list[Route] | None = None
        best_cost = float("inf")
        candidate_depots = set(candidate_depot_ids)
        base_cost = objective_cost(instance, Solution(instance.name, routes))

        for route_index, route in enumerate(routes):
            if route.depot_id not in candidate_depots:
                continue
            if route_load(instance, route) + instance.customers_by_id[customer_id].demand > instance.vehicle_capacity + EPS:
                continue
            for position in range(len(route.customer_ids) + 1):
                candidate = [Route(item.depot_id, list(item.customer_ids)) for item in routes]
                customers = list(candidate[route_index].customer_ids)
                customers.insert(position, customer_id)
                candidate[route_index] = Route(route.depot_id, customers)
                if not _partial_constraints_ok(instance, candidate):
                    continue
                cost = objective_cost(instance, Solution(instance.name, candidate)) - base_cost
                if cost < best_cost:
                    best_routes = candidate
                    best_cost = cost

        for depot_id in sorted(candidate_depots):
            candidate = [Route(item.depot_id, list(item.customer_ids)) for item in routes]
            candidate.append(Route(depot_id, [customer_id]))
            if not _partial_constraints_ok(instance, candidate):
                continue
            cost = objective_cost(instance, Solution(instance.name, candidate)) - base_cost
            if cost < best_cost:
                best_routes = candidate
                best_cost = cost

        if best_routes is None:
            raise ValueError(f"No restricted feasible insertion for customer {customer_id}")
        return best_routes


def _partial_constraints_ok(instance: Instance, routes: list[Route]) -> bool:
    depot_loads: dict[int, float] = {}
    depot_routes: dict[int, int] = {}
    for route in routes:
        load = route_load(instance, route)
        if load > instance.vehicle_capacity + EPS:
            return False
        depot = instance.depots_by_id[route.depot_id]
        depot_loads[route.depot_id] = depot_loads.get(route.depot_id, 0.0) + load
        depot_routes[route.depot_id] = depot_routes.get(route.depot_id, 0) + 1
        if depot_loads[route.depot_id] > depot.capacity + EPS:
            return False
        if depot_routes[route.depot_id] > depot.vehicle_limit:
            return False
    return True
