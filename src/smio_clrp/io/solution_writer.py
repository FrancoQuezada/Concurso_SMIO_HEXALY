from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from smio_clrp.core.instance import Instance
from smio_clrp.core.solution import Route, Solution
from smio_clrp.evaluation.cost import objective_cost


def write_solution(
    solution: Solution,
    path: str | Path,
    instance: Instance | None = None,
    recompute_cost: bool = True,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_solution(solution, instance, recompute_cost), encoding="utf-8")


def format_solution(
    solution: Solution,
    instance: Instance | None = None,
    recompute_cost: bool = True,
) -> str:
    if recompute_cost:
        if instance is None:
            raise ValueError("instance is required when recompute_cost=True")
        cost = objective_cost(instance, solution)
    elif solution.reported_cost is not None:
        cost = solution.reported_cost
    else:
        raise ValueError("solution.reported_cost is required when recompute_cost=False")

    routes_by_depot: dict[int, list[Route]] = defaultdict(list)
    for route in solution.routes:
        if route.customer_ids:
            routes_by_depot[route.depot_id].append(route)

    lines = [
        f"# instance = {solution.instance_name or (instance.name if instance else '')}",
        f"COST: {cost:.10f}",
        f"DEPOTS_OPENED: {len(routes_by_depot)}",
        f"ROUTES: {sum(len(routes) for routes in routes_by_depot.values())}",
    ]
    for depot_id in sorted(routes_by_depot):
        lines.append(f"DEPOT {depot_id}")
        for route in routes_by_depot[depot_id]:
            customers = " ".join(str(customer_id) for customer_id in route.customer_ids)
            lines.append(f"ROUTE: {customers}")
    lines.append("EOF")
    return "\n".join(lines) + "\n"
