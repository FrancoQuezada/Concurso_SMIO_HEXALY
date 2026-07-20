from __future__ import annotations

import time

from smio_clrp.algorithms.alns.alns_solver import ALNSSolver
from smio_clrp.algorithms.base import Solver, SolverConfig, SolverResult
from smio_clrp.algorithms.clustering.clusterers import ClusterResult, cluster_customers
from smio_clrp.algorithms.common import clone_solution
from smio_clrp.algorithms.constructive.regret import regret_insertion
from smio_clrp.algorithms.fixopt.fixopt_solver import FixOptimizeSolver
from smio_clrp.algorithms.local_search.driver import improve_solution
from smio_clrp.core.distance import distance
from smio_clrp.core.instance import Instance
from smio_clrp.core.solution import Route, Solution
from smio_clrp.evaluation.cost import objective_cost
from smio_clrp.evaluation.validator import validate_solution


EPS = 1e-9


class ClusteredConstructiveSolver(Solver):
    """Build a feasible CLRP seed from capacity-aware customer clusters."""

    algorithm_name = "clustered"

    def solve(self, instance: Instance) -> SolverResult:
        start = time.perf_counter()
        try:
            result = cluster_customers(
                instance,
                method=str(self.config.metadata.get("cluster_method", "auto")),
                num_clusters=_optional_int(self.config.metadata.get("cluster_count")),
                max_iterations=int(self.config.metadata.get("cluster_iterations", 20)),
            )
            solution, fallback = _solution_from_clusters(instance, result)
            if bool(self.config.metadata.get("cluster_local_search", True)):
                solution = improve_solution(
                    instance,
                    solution,
                    max_iterations=int(self.config.metadata.get("cluster_local_search_iterations", 25)),
                    time_limit_seconds=self.config.time_limit_seconds,
                )
            validation = validate_solution(instance, solution)
            return SolverResult(
                solution=solution if validation.is_feasible else None,
                cost=objective_cost(instance, solution) if validation.is_feasible else None,
                feasible=validation.is_feasible,
                runtime_seconds=time.perf_counter() - start,
                seed=self.config.seed,
                algorithm_name=self.algorithm_name,
                metadata={
                    "cluster_method_requested": result.requested_method,
                    "cluster_method_used": result.effective_method,
                    "cluster_count": len(result.clusters),
                    "cluster_loads": result.loads,
                    "cluster_iterations": result.iterations,
                    "cluster_metadata": result.metadata,
                    "construction_fallback": fallback,
                    "validation_errors": validation.errors,
                },
            )
        except ValueError as exc:
            return SolverResult(None, None, False, time.perf_counter() - start, self.config.seed, self.algorithm_name, {"error": str(exc)})


class ClusteredHybridSolver(Solver):
    """Run capacity-aware clustering, then ALNS, then Fix-and-Optimize."""

    algorithm_name = "clustered_hybrid"

    def solve(self, instance: Instance) -> SolverResult:
        start = time.perf_counter()
        clustered = ClusteredConstructiveSolver(self.config).solve(instance)
        if clustered.solution is None:
            clustered.algorithm_name = self.algorithm_name
            return clustered
        alns = ALNSSolver(config=self.config, initial_solution=clustered.solution).solve(instance)
        incumbent = alns.solution or clustered.solution
        incumbent_cost = alns.cost if alns.solution is not None else clustered.cost
        fix_config = SolverConfig(
            seed=self.config.seed,
            time_limit_seconds=self.config.metadata.get("fixopt_time_limit", self.config.time_limit_seconds),
            metadata=self.config.metadata,
        )
        fix = FixOptimizeSolver(initial_solution=incumbent, config=fix_config).solve(instance)
        if fix.solution is not None and (incumbent_cost is None or fix.cost is not None and fix.cost <= incumbent_cost + EPS):
            final_solution, final_cost = fix.solution, fix.cost
        else:
            final_solution, final_cost = incumbent, incumbent_cost
        return SolverResult(
            solution=clone_solution(final_solution),
            cost=final_cost,
            feasible=final_solution is not None,
            runtime_seconds=time.perf_counter() - start,
            seed=self.config.seed,
            algorithm_name=self.algorithm_name,
            metadata={
                "clustered_cost": clustered.cost,
                "cluster_metadata": clustered.metadata,
                "alns_initial_cost": alns.metadata.get("initial_cost"),
                "alns_best_cost": alns.cost,
                "fixopt_best_cost": fix.cost,
                "final_cost": final_cost,
            },
        )


def _solution_from_clusters(instance: Instance, result: ClusterResult) -> tuple[Solution, str | None]:
    remaining_capacity = {depot.id: depot.capacity for depot in instance.depots}
    remaining_routes = {depot.id: depot.vehicle_limit for depot in instance.depots}
    opened: set[int] = set()
    routes: list[Route] = []
    ordered_clusters = sorted(
        result.clusters,
        key=lambda cluster: (-sum(instance.customers_by_id[customer_id].demand for customer_id in cluster), min(cluster)),
    )
    for cluster in ordered_clusters:
        load = sum(instance.customers_by_id[customer_id].demand for customer_id in cluster)
        feasible_depots = [
            depot
            for depot in instance.depots
            if remaining_capacity[depot.id] + EPS >= load and remaining_routes[depot.id] > 0
        ]
        if not feasible_depots:
            fallback = regret_insertion(instance, regret_k=3)
            return fallback, "regret3: clustered routes could not be assigned to depot capacities"
        depot = min(
            feasible_depots,
            key=lambda item: (_cluster_depot_cost(instance, item.id, cluster, item.id not in opened), item.id),
        )
        route = Route(depot.id, _nearest_neighbor_order(instance, depot.id, cluster))
        routes.append(route)
        opened.add(depot.id)
        remaining_capacity[depot.id] -= load
        remaining_routes[depot.id] -= 1
    return Solution(instance_name=instance.name, routes=routes), None


def _cluster_depot_cost(instance: Instance, depot_id: int, cluster: list[int], unopened: bool) -> float:
    first = min(
        cluster,
        key=lambda customer_id: (distance(instance, ("depot", depot_id), ("customer", customer_id)), customer_id),
    )
    last = min(
        cluster,
        key=lambda customer_id: (distance(instance, ("customer", customer_id), ("depot", depot_id)), customer_id),
    )
    return (
        distance(instance, ("depot", depot_id), ("customer", first))
        + distance(instance, ("customer", last), ("depot", depot_id))
        + instance.route_fixed_cost
        + (instance.depots_by_id[depot_id].opening_cost if unopened else 0.0)
    )


def _nearest_neighbor_order(instance: Instance, depot_id: int, customer_ids: list[int]) -> list[int]:
    remaining = set(customer_ids)
    ordered: list[int] = []
    current = ("depot", depot_id)
    while remaining:
        next_customer = min(
            remaining,
            key=lambda customer_id: (distance(instance, current, ("customer", customer_id)), customer_id),
        )
        ordered.append(next_customer)
        remaining.remove(next_customer)
        current = ("customer", next_customer)
    return ordered


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)
