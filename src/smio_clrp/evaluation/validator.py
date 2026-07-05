from __future__ import annotations

import math
from collections import Counter, defaultdict

from smio_clrp.core.instance import Instance
from smio_clrp.core.solution import Solution, ValidationResult
from smio_clrp.evaluation.cost import objective_cost


def validate_solution(instance: Instance, solution: Solution, tolerance: float = 1e-4) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    customer_counts: Counter[int] = Counter()
    depot_demand: dict[int, float] = defaultdict(float)
    depot_routes: Counter[int] = Counter()

    if solution.instance_name and solution.instance_name != instance.name:
        warnings.append(
            f"Solution instance name '{solution.instance_name}' differs from '{instance.name}'"
        )

    for route_index, route in enumerate(solution.routes, start=1):
        if route.depot_id not in instance.depots_by_id:
            errors.append(f"Route {route_index} references unknown depot {route.depot_id}")
        if not route.customer_ids:
            errors.append(f"Route {route_index} assigned to depot {route.depot_id} is empty")

        route_demand = 0.0
        for customer_id in route.customer_ids:
            customer_counts[customer_id] += 1
            customer = instance.customers_by_id.get(customer_id)
            if customer is None:
                errors.append(f"Route {route_index} references unknown customer {customer_id}")
            else:
                route_demand += customer.demand

        if route_demand > instance.vehicle_capacity + tolerance:
            errors.append(
                f"Route {route_index} demand {route_demand:g} exceeds vehicle capacity "
                f"{instance.vehicle_capacity:g}"
            )
        depot_demand[route.depot_id] += route_demand
        depot_routes[route.depot_id] += 1

    expected_customers = set(instance.customers_by_id)
    seen_customers = set(customer_counts)
    for customer_id in sorted(expected_customers - seen_customers):
        errors.append(f"Missing customer {customer_id}")
    for customer_id in sorted(seen_customers - expected_customers):
        if customer_counts[customer_id] == 1:
            continue
    for customer_id, count in sorted(customer_counts.items()):
        if customer_id in expected_customers and count > 1:
            errors.append(f"Customer {customer_id} appears {count} times")

    for depot_id, assigned_demand in sorted(depot_demand.items()):
        depot = instance.depots_by_id.get(depot_id)
        if depot is None:
            continue
        if assigned_demand > depot.capacity + tolerance:
            errors.append(
                f"Depot {depot_id} assigned demand {assigned_demand:g} exceeds capacity "
                f"{depot.capacity:g}"
            )
        routes_count = depot_routes[depot_id]
        if routes_count > depot.vehicle_limit:
            errors.append(
                f"Depot {depot_id} uses {routes_count} routes, exceeding limit {depot.vehicle_limit}"
            )

    cost = math.nan
    if not any("unknown" in error for error in errors):
        try:
            cost = objective_cost(instance, solution)
        except Exception as exc:  # pragma: no cover - defensive aggregation
            errors.append(f"Could not recompute cost: {exc}")

    if solution.reported_cost is not None and not math.isnan(cost):
        if abs(solution.reported_cost - cost) > tolerance:
            errors.append(
                f"Reported cost {solution.reported_cost:.10g} differs from recomputed "
                f"cost {cost:.10g}"
            )

    stats = {
        "customers": len(instance.customers),
        "routes": len(solution.routes),
        "depots_opened": len(solution.opened_depot_ids),
        "depot_demand": dict(depot_demand),
        "depot_routes": dict(depot_routes),
    }
    return ValidationResult(
        is_feasible=not errors,
        cost=cost,
        errors=errors,
        warnings=warnings,
        stats=stats,
    )
