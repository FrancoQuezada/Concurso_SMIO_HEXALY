from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

# Hard cap on the multi-customer route candidates generated per subproblem (on top of the
# always-included per-customer singletons). Subset enumeration is combinatorial in the
# number of released customers; without a cap, a 12-customer neighborhood with low-demand
# customers can produce tens of thousands of candidates (each needing a nearest-neighbor +
# 2-opt sequencing pass), turning candidate generation itself into the bottleneck before
# Gurobi is even called. Singletons are generated separately and are never subject to this
# cap, so the set-partitioning model always has a feasible covering fallback.
MAX_EXTRA_CANDIDATE_ROUTES = 2000

from smio_clrp.algorithms.common import EPS, route_load
from smio_clrp.algorithms.fixopt.backend import BackendUnavailable, FixOptResult
from smio_clrp.algorithms.fixopt.config import FixOptConfig
from smio_clrp.algorithms.fixopt.neighborhoods import FixOptNeighborhood
from smio_clrp.algorithms.local_search.two_opt import intra_route_two_opt
from smio_clrp.core.distance import distance
from smio_clrp.core.instance import Instance
from smio_clrp.core.solution import Route, Solution
from smio_clrp.evaluation.cost import objective_cost, route_distance
from smio_clrp.evaluation.validator import validate_solution


class MIPFixOptBackend:
    backend_name = "mip"

    def __init__(self, seed: int = 1) -> None:
        try:
            import gurobipy  # noqa: F401
        except ImportError as exc:
            raise BackendUnavailable("gurobipy is not installed") from exc
        self.seed = seed

    def reoptimize(
        self,
        instance: Instance,
        solution: Solution,
        neighborhood: FixOptNeighborhood,
        config: FixOptConfig,
    ) -> FixOptResult:
        if len(neighborhood.released_customer_ids) > config.max_customers_per_subproblem:
            return FixOptResult(
                None,
                False,
                self.backend_name,
                {"error": "restricted MIP skipped because subproblem is too large"},
            )
        if not neighborhood.candidate_depot_ids:
            return FixOptResult(None, False, self.backend_name, {"error": "no candidate depots for MIP subproblem"})

        candidates = _generate_candidate_routes(instance, neighborhood)
        if not candidates:
            return FixOptResult(None, False, self.backend_name, {"error": "no feasible candidate routes generated"})

        import gurobipy as gp

        try:
            chosen_routes, subproblem_objective, status = _solve_set_partitioning(
                instance, neighborhood, candidates, config, self.seed
            )
        except gp.GurobiError as exc:
            return FixOptResult(None, False, self.backend_name, {"error": f"gurobi error: {exc}"})

        if chosen_routes is None:
            reason = (
                "subproblem is infeasible given fixed-route capacity/vehicle_limit usage"
                if status == gp.GRB.INFEASIBLE
                else "gurobi found no feasible solution within the time limit"
            )
            return FixOptResult(None, False, self.backend_name, {"error": reason, "gurobi_status": status})

        candidate_solution = Solution(solution.instance_name, neighborhood.fixed_routes + chosen_routes)
        validation = validate_solution(instance, candidate_solution)
        if not validation.is_feasible:
            return FixOptResult(None, False, self.backend_name, {"error": "; ".join(validation.errors)})

        return FixOptResult(
            candidate_solution,
            True,
            self.backend_name,
            {
                "cost": objective_cost(instance, candidate_solution),
                "subproblem_objective": subproblem_objective,
                "gurobi_status": status,
                "num_candidate_routes": len(candidates),
            },
        )


@dataclass(frozen=True)
class _RouteCandidate:
    route: Route
    cost: float
    demand: float
    customer_ids: frozenset[int]


def _generate_candidate_routes(instance: Instance, neighborhood: FixOptNeighborhood) -> list[_RouteCandidate]:
    """Enumerate candidate routes: one per (feasible customer subset, candidate depot) pair."""
    released = neighborhood.released_customer_ids
    demand_by_customer = {customer_id: instance.customers_by_id[customer_id].demand for customer_id in released}
    depot_ids = neighborhood.candidate_depot_ids

    def make_candidate(depot_id: int, subset: tuple[int, ...]) -> _RouteCandidate:
        route = _sequence_route(instance, depot_id, subset)
        cost = route_distance(instance, route) + instance.route_fixed_cost
        demand = sum(demand_by_customer[customer_id] for customer_id in subset)
        return _RouteCandidate(route, cost, demand, frozenset(subset))

    # Always include every (customer, depot) singleton first and uncapped, so the
    # set-partitioning model always has a feasible covering fallback even if the
    # multi-customer enumeration below gets truncated by the candidate cap.
    candidates: list[_RouteCandidate] = [
        make_candidate(depot_id, (customer_id,))
        for depot_id in depot_ids
        for customer_id in released
        if demand_by_customer[customer_id] <= instance.vehicle_capacity + EPS
    ]

    # Interleaved subset-outer, depot-inner: exhausting all subsets for one depot before
    # moving to the next (depot-outer) let the cap starve every depot but the first of any
    # multi-customer option once len(subsets) alone exceeded the cap (e.g. 1774 subsets for
    # a 12-customer neighborhood vs. a 2000 cap shared across up to 7+ candidate depots) --
    # confirmed as the cause of spurious set-partitioning infeasibilities: depots with real
    # spare capacity had only singleton candidates left and couldn't consolidate customers
    # into few enough routes to fit their vehicle_limit. Interleaving gives every candidate
    # depot a fair, roughly equal share of subset variety before the cap is hit.
    #
    # `extra` (not len(candidates)) is what's compared to the cap: candidates already holds
    # the singletons above, and _feasible_customer_subsets excludes size-1 subsets (they'd
    # otherwise duplicate the singleton loop), so there is no overlap left to deduplicate --
    # the cap now applies to exactly the "extra" multi-customer candidates it's documented to.
    max_subset_size = _max_candidate_subset_size(instance, demand_by_customer)
    # Bounding by MAX_EXTRA_CANDIDATE_ROUTES here (not just when converting subsets into
    # candidates below) matters a lot once max_customers_per_subproblem is pushed well past
    # the ~12-customer default: with enough released customers and low enough per-customer
    # demand relative to capacity, the number of *feasible* subsets up to max_subset_size is
    # combinatorial and was previously fully materialized into a list before any truncation
    # -- confirmed to balloon a single process to ~18GB RSS in under a minute at
    # max_customers_per_subproblem=60 on a large-scale instance, well before Gurobi ever saw
    # a candidate.
    subsets = _feasible_customer_subsets(
        released, demand_by_customer, instance.vehicle_capacity, max_subset_size, MAX_EXTRA_CANDIDATE_ROUTES
    )
    extra = 0
    for subset in subsets:
        for depot_id in depot_ids:
            if extra >= MAX_EXTRA_CANDIDATE_ROUTES:
                return candidates
            candidates.append(make_candidate(depot_id, subset))
            extra += 1
    return candidates


def _max_candidate_subset_size(instance: Instance, demand_by_customer: dict[int, float]) -> int:
    """Cap subset size near a "typical" route's customer count (capacity / average demand),
    with a little slack, instead of enumerating up to the full capacity-limited subset size."""
    if not demand_by_customer:
        return 0
    avg_demand = sum(demand_by_customer.values()) / len(demand_by_customer)
    typical_route_size = max(1, math.ceil(instance.vehicle_capacity / max(avg_demand, EPS)))
    return min(len(demand_by_customer), typical_route_size + 2)


def _feasible_customer_subsets(
    customers: list[int],
    demand: dict[int, float],
    capacity: float,
    max_subset_size: int,
    limit: int,
) -> list[tuple[int, ...]]:
    """Subsets of size >= 2 of released customers whose demand fits one vehicle, up to
    max_subset_size. Size-1 subsets are excluded on purpose: _generate_candidate_routes
    always adds every (customer, depot) singleton separately, so including them here too
    would just duplicate that work.

    Stops as soon as `limit` subsets are found, instead of enumerating every feasible
    subset and truncating afterwards: the feasible-subset count is combinatorial in the
    number of released customers, and materializing it all before truncation is what
    caused unbounded memory growth on subproblems larger than the ~12-customer default."""
    subsets: list[tuple[int, ...]] = []
    current: list[int] = []

    def backtrack(start: int, current_demand: float) -> None:
        if len(subsets) >= limit:
            return
        if len(current) >= 2:
            subsets.append(tuple(current))
            if len(subsets) >= limit:
                return
        if len(current) >= max_subset_size:
            return
        for index in range(start, len(customers)):
            customer_id = customers[index]
            added_demand = current_demand + demand[customer_id]
            if added_demand > capacity + EPS:
                continue
            current.append(customer_id)
            backtrack(index + 1, added_demand)
            current.pop()
            if len(subsets) >= limit:
                return

    backtrack(0, 0.0)
    return subsets


def _sequence_route(instance: Instance, depot_id: int, customer_ids: tuple[int, ...]) -> Route:
    """Order a fixed customer subset with nearest-neighbor + 2-opt (route packing is the MIP's job, not TSP)."""
    remaining = list(customer_ids)
    ordered: list[int] = []
    current = ("depot", depot_id)
    while remaining:
        nearest = min(remaining, key=lambda customer_id: distance(instance, current, ("customer", customer_id)))
        ordered.append(nearest)
        remaining.remove(nearest)
        current = ("customer", nearest)
    return intra_route_two_opt(instance, Route(depot_id, ordered))


def _solve_set_partitioning(
    instance: Instance,
    neighborhood: FixOptNeighborhood,
    candidates: list[_RouteCandidate],
    config: FixOptConfig,
    seed: int,
) -> tuple[list[Route] | None, float | None, int]:
    import gurobipy as gp
    from gurobipy import GRB

    fixed_routes = neighborhood.fixed_routes
    already_open_depots = {route.depot_id for route in fixed_routes}
    fixed_route_counts = Counter(route.depot_id for route in fixed_routes)
    fixed_demand: dict[int, float] = defaultdict(float)
    for route in fixed_routes:
        fixed_demand[route.depot_id] += route_load(instance, route)

    routes_by_depot: dict[int, list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        routes_by_depot[candidate.route.depot_id].append(index)

    model = gp.Model("fixopt_set_partitioning")
    try:
        model.Params.OutputFlag = 0
        model.Params.Seed = seed
        if config.mip_time_limit_seconds is not None:
            model.Params.TimeLimit = config.mip_time_limit_seconds

        x = model.addVars(len(candidates), vtype=GRB.BINARY, name="x")
        # Only depots not already opened by a fixed route need an "open" decision (and its cost).
        open_vars = {
            depot_id: model.addVar(vtype=GRB.BINARY, name=f"open_{depot_id}")
            for depot_id in neighborhood.candidate_depot_ids
            if depot_id not in already_open_depots
        }

        for customer_id in neighborhood.released_customer_ids:
            covering = [index for index, candidate in enumerate(candidates) if customer_id in candidate.customer_ids]
            model.addConstr(gp.quicksum(x[index] for index in covering) == 1, name=f"cover_{customer_id}")

        for depot_id in neighborhood.candidate_depot_ids:
            depot = instance.depots_by_id[depot_id]
            depot_route_indices = routes_by_depot.get(depot_id, [])
            model.addConstr(
                gp.quicksum(x[index] for index in depot_route_indices) + fixed_route_counts.get(depot_id, 0)
                <= depot.vehicle_limit,
                name=f"vehicle_limit_{depot_id}",
            )
            model.addConstr(
                gp.quicksum(x[index] * candidates[index].demand for index in depot_route_indices)
                + fixed_demand.get(depot_id, 0.0)
                <= depot.capacity,
                name=f"depot_capacity_{depot_id}",
            )
            if depot_id in open_vars:
                for index in depot_route_indices:
                    model.addConstr(x[index] <= open_vars[depot_id], name=f"link_open_{depot_id}_{index}")

        objective = gp.quicksum(candidates[index].cost * x[index] for index in range(len(candidates)))
        if open_vars:
            objective += gp.quicksum(
                instance.depots_by_id[depot_id].opening_cost * var for depot_id, var in open_vars.items()
            )
        model.setObjective(objective, GRB.MINIMIZE)
        model.optimize()

        if model.SolCount == 0:
            return None, None, int(model.Status)

        chosen_routes = [candidates[index].route for index in range(len(candidates)) if x[index].X > 0.5]
        return chosen_routes, float(model.ObjVal), int(model.Status)
    finally:
        # Gurobi models wrap C-level allocations that Python's refcounting/GC does not
        # reclaim promptly on their own -- confirmed as the cause of a process ballooning
        # to ~20GB RSS after a few hundred reopt.py iterations (each building a fresh
        # model here) on a large-scale instance, without ever raising in Python itself.
        model.dispose()


def gurobi_available() -> bool:
    try:
        import gurobipy  # noqa: F401
    except ImportError:
        return False
    return True
