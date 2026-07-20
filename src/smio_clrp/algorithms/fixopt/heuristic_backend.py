from __future__ import annotations

import random

import numpy as np

from smio_clrp.algorithms.common import EPS, depot_loads, depot_route_counts, insertion_delta, route_load
from smio_clrp.algorithms.fixopt.backend import FixOptResult
from smio_clrp.algorithms.fixopt.config import FixOptConfig
from smio_clrp.algorithms.fixopt.neighborhoods import FixOptNeighborhood
from smio_clrp.algorithms.local_search.driver import improve_solution
from smio_clrp.core.instance import Instance
from smio_clrp.core.solution import Route, Solution
from smio_clrp.evaluation.cost import objective_cost, route_distance
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
            # Polish only the routes this subproblem actually touched, not the whole
            # solution: validate_solution/improve_solution require a *complete* solution
            # (every instance customer present exactly once), so we build a small
            # sub-instance scoped to the touched customers/depots rather than passing a
            # partial Solution against the full instance. The routes untouched here are
            # exactly neighborhood.fixed_routes per the F&O contract ("the rest of the
            # solution is held fixed"); the final validate_solution below still re-checks
            # the merged solution against the real instance as a safety net, since the
            # sub-instance's depot capacity/vehicle-limit figures don't know about demand
            # already committed elsewhere by untouched fixed routes at the same depot, a
            # polish move can turn out infeasible once merged back in. That's an optional
            # improvement failing, not the subproblem itself failing, so we keep the
            # already-feasible pre-polish `candidate` rather than discarding the subproblem.
            fixed_keys = {(route.depot_id, tuple(route.customer_ids)) for route in neighborhood.fixed_routes}
            touched = [route for route in routes if (route.depot_id, tuple(route.customer_ids)) not in fixed_keys]
            untouched = [route for route in routes if (route.depot_id, tuple(route.customer_ids)) in fixed_keys]
            if touched:
                sub_instance = _touched_sub_instance(instance, touched, neighborhood.candidate_depot_ids)
                try:
                    improved = improve_solution(sub_instance, Solution(solution.instance_name, touched), max_iterations=5)
                    polished = Solution(solution.instance_name, untouched + improved.routes)
                    if validate_solution(instance, polished).is_feasible:
                        candidate = polished
                except ValueError:
                    pass
        return FixOptResult(candidate, True, self.backend_name, {"cost": objective_cost(instance, candidate)})

    def _insert_customer(
        self,
        instance: Instance,
        routes: list[Route],
        customer_id: int,
        candidate_depot_ids: list[int],
    ) -> list[Route]:
        candidate_depots = set(candidate_depot_ids)
        demand = instance.customers_by_id[customer_id].demand
        opened_depot_ids = {route.depot_id for route in routes}
        loads = depot_loads(instance, routes)
        counts = depot_route_counts(routes)

        best_cost = float("inf")
        best_move: tuple[int, int] | None = None  # (route_index, position); route_index=-1 means new route

        for route_index, route in enumerate(routes):
            if route.depot_id not in candidate_depots:
                continue
            depot = instance.depots_by_id[route.depot_id]
            if route_load(instance, route) + demand > instance.vehicle_capacity + EPS:
                continue
            if loads[route.depot_id] + demand > depot.capacity + EPS:
                continue
            for position in range(len(route.customer_ids) + 1):
                cost = insertion_delta(instance, route, customer_id, position)
                if cost < best_cost:
                    best_cost = cost
                    best_move = (route_index, position)

        if demand <= instance.vehicle_capacity + EPS:
            for depot_id in sorted(candidate_depots):
                depot = instance.depots_by_id[depot_id]
                if loads.get(depot_id, 0.0) + demand > depot.capacity + EPS:
                    continue
                if counts.get(depot_id, 0) + 1 > depot.vehicle_limit:
                    continue
                opening_cost = 0.0 if depot_id in opened_depot_ids else depot.opening_cost
                cost = route_distance(instance, Route(depot_id, [customer_id])) + instance.route_fixed_cost + opening_cost
                if cost < best_cost:
                    best_cost = cost
                    best_move = (-1, depot_id)

        if best_move is None:
            raise ValueError(f"No restricted feasible insertion for customer {customer_id}")

        route_index, position_or_depot = best_move
        if route_index == -1:
            return routes + [Route(position_or_depot, [customer_id])]
        updated = [Route(item.depot_id, list(item.customer_ids)) for item in routes]
        customers = list(updated[route_index].customer_ids)
        customers.insert(position_or_depot, customer_id)
        updated[route_index] = Route(routes[route_index].depot_id, customers)
        return updated


def _touched_sub_instance(instance: Instance, touched_routes: list[Route], candidate_depot_ids: list[int]) -> Instance:
    """Build a sub-instance covering only the touched customers plus candidate depots,
    so local_search's full-coverage validation applies to a small solution instead of
    the whole instance. Depot ids include candidate_depot_ids (not just the depots the
    touched routes currently use) so relocation can still open a route at any of them."""
    customer_ids = [customer_id for route in touched_routes for customer_id in route.customer_ids]
    depot_ids = sorted({route.depot_id for route in touched_routes} | set(candidate_depot_ids))
    depots = [instance.depots_by_id[depot_id] for depot_id in depot_ids]
    customers = [instance.customers_by_id[customer_id] for customer_id in customer_ids]
    distance_matrix = None
    if instance.distance_format == "FULL_MATRIX":
        indices = [instance.node_index[("depot", depot_id)] for depot_id in depot_ids]
        indices += [instance.node_index[("customer", customer_id)] for customer_id in customer_ids]
        distance_matrix = instance.distance_matrix[np.ix_(indices, indices)]
    return Instance(
        name=instance.name,
        depots=depots,
        customers=customers,
        vehicle_capacity=instance.vehicle_capacity,
        route_fixed_cost=instance.route_fixed_cost,
        distance_format=instance.distance_format,
        distance_matrix=distance_matrix,
    )
