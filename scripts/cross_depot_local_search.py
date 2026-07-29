from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from smio_clrp.algorithms.common import route_load
from smio_clrp.core.solution import Route, Solution
from smio_clrp.evaluation.cost import objective_cost
from smio_clrp.evaluation.validator import validate_solution
from smio_clrp.io.instance_reader import read_instance
from smio_clrp.io.solution_reader import read_solution
from smio_clrp.io.solution_writer import write_solution


EPS = 1e-9


def _distance_matrix(instance) -> np.ndarray:
    if instance.distance_matrix is not None:
        return np.asarray(instance.distance_matrix, dtype=float)
    nodes = [*instance.depots, *instance.customers]
    x = np.asarray([float(node.x) for node in nodes])
    y = np.asarray([float(node.y) for node in nodes])
    return np.round(np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :]), 1)


def _replacement_delta(
    matrix: np.ndarray,
    depot_node: int,
    customer_nodes: dict[int, int],
    route: list[int],
    position: int,
    replacement: int,
) -> float:
    old_node = customer_nodes[route[position]]
    new_node = customer_nodes[replacement]
    previous = depot_node if position == 0 else customer_nodes[route[position - 1]]
    following = depot_node if position + 1 == len(route) else customer_nodes[route[position + 1]]
    return float(
        matrix[previous, new_node]
        + matrix[new_node, following]
        - matrix[previous, old_node]
        - matrix[old_node, following]
    )


def _removal_delta(
    matrix: np.ndarray,
    depot_node: int,
    customer_nodes: dict[int, int],
    route: list[int],
    position: int,
) -> float:
    node = customer_nodes[route[position]]
    previous = depot_node if position == 0 else customer_nodes[route[position - 1]]
    following = depot_node if position + 1 == len(route) else customer_nodes[route[position + 1]]
    return float(matrix[previous, following] - matrix[previous, node] - matrix[node, following])


def _insertion_delta(
    matrix: np.ndarray,
    depot_node: int,
    customer_nodes: dict[int, int],
    route: list[int],
    position: int,
    customer_id: int,
) -> float:
    node = customer_nodes[customer_id]
    previous = depot_node if position == 0 else customer_nodes[route[position - 1]]
    following = depot_node if position == len(route) else customer_nodes[route[position]]
    return float(matrix[previous, node] + matrix[node, following] - matrix[previous, following])


def _best_cross_depot_swap(
    instance,
    routes: list[Route],
    route_loads: list[float],
    depot_loads: dict[int, float],
    matrix: np.ndarray,
    customer_nodes: dict[int, int],
    demands: dict[int, float],
    deadline: float,
):
    best_delta = -EPS
    best_move = None
    depot_nodes = {
        depot.id: instance.node_index[('depot', depot.id)] for depot in instance.depots
    }
    for first_index, first in enumerate(routes):
        for second_index in range(first_index + 1, len(routes)):
            if time.perf_counter() >= deadline:
                return best_move
            second = routes[second_index]
            if first.depot_id == second.depot_id:
                continue
            first_depot = instance.depots_by_id[first.depot_id]
            second_depot = instance.depots_by_id[second.depot_id]
            first_customers = first.customer_ids
            second_customers = second.customer_ids
            for first_position, first_customer in enumerate(first_customers):
                first_demand = demands[first_customer]
                for second_position, second_customer in enumerate(second_customers):
                    second_demand = demands[second_customer]
                    if (
                        route_loads[first_index] - first_demand + second_demand
                        > instance.vehicle_capacity + EPS
                    ):
                        continue
                    if (
                        route_loads[second_index] - second_demand + first_demand
                        > instance.vehicle_capacity + EPS
                    ):
                        continue
                    if (
                        depot_loads[first.depot_id] - first_demand + second_demand
                        > first_depot.capacity + EPS
                    ):
                        continue
                    if (
                        depot_loads[second.depot_id] - second_demand + first_demand
                        > second_depot.capacity + EPS
                    ):
                        continue
                    delta = _replacement_delta(
                        matrix,
                        depot_nodes[first.depot_id],
                        customer_nodes,
                        first_customers,
                        first_position,
                        second_customer,
                    )
                    delta += _replacement_delta(
                        matrix,
                        depot_nodes[second.depot_id],
                        customer_nodes,
                        second_customers,
                        second_position,
                        first_customer,
                    )
                    if delta < best_delta:
                        best_delta = delta
                        best_move = (
                            best_delta,
                            first_index,
                            first_position,
                            second_index,
                            second_position,
                        )
    return best_move


def _best_cross_depot_relocate(
    instance,
    routes: list[Route],
    route_loads: list[float],
    depot_loads: dict[int, float],
    matrix: np.ndarray,
    customer_nodes: dict[int, int],
    demands: dict[int, float],
    deadline: float,
):
    best_delta = -EPS
    best_move = None
    depot_nodes = {
        depot.id: instance.node_index[('depot', depot.id)] for depot in instance.depots
    }
    for source_index, source in enumerate(routes):
        if len(source.customer_ids) <= 1:
            continue
        for target_index, target in enumerate(routes):
            if time.perf_counter() >= deadline:
                return best_move
            if source.depot_id == target.depot_id:
                continue
            target_depot = instance.depots_by_id[target.depot_id]
            for source_position, customer_id in enumerate(source.customer_ids):
                demand = demands[customer_id]
                if route_loads[target_index] + demand > instance.vehicle_capacity + EPS:
                    continue
                if depot_loads[target.depot_id] + demand > target_depot.capacity + EPS:
                    continue
                removal = _removal_delta(
                    matrix,
                    depot_nodes[source.depot_id],
                    customer_nodes,
                    source.customer_ids,
                    source_position,
                )
                for target_position in range(len(target.customer_ids) + 1):
                    delta = removal + _insertion_delta(
                        matrix,
                        depot_nodes[target.depot_id],
                        customer_nodes,
                        target.customer_ids,
                        target_position,
                        customer_id,
                    )
                    if delta < best_delta:
                        best_delta = delta
                        best_move = (
                            best_delta,
                            source_index,
                            source_position,
                            target_index,
                            target_position,
                        )
    return best_move


def _apply_swap(
    routes: list[Route],
    route_loads: list[float],
    depot_loads: dict[int, float],
    demands: dict[int, float],
    move,
) -> None:
    _, first_index, first_position, second_index, second_position = move
    first = routes[first_index]
    second = routes[second_index]
    first_customer = first.customer_ids[first_position]
    second_customer = second.customer_ids[second_position]
    first_demand = demands[first_customer]
    second_demand = demands[second_customer]
    first.customer_ids[first_position], second.customer_ids[second_position] = (
        second_customer,
        first_customer,
    )
    route_loads[first_index] += second_demand - first_demand
    route_loads[second_index] += first_demand - second_demand
    depot_loads[first.depot_id] += second_demand - first_demand
    depot_loads[second.depot_id] += first_demand - second_demand


def _apply_relocate(
    routes: list[Route],
    route_loads: list[float],
    depot_loads: dict[int, float],
    demands: dict[int, float],
    move,
) -> None:
    _, source_index, source_position, target_index, target_position = move
    source = routes[source_index]
    target = routes[target_index]
    customer_id = source.customer_ids.pop(source_position)
    target.customer_ids.insert(target_position, customer_id)
    demand = demands[customer_id]
    route_loads[source_index] -= demand
    route_loads[target_index] += demand
    depot_loads[source.depot_id] -= demand
    depot_loads[target.depot_id] += demand


def main() -> int:
    parser = argparse.ArgumentParser(description='Fast cross-depot CLRP local search')
    parser.add_argument('instance')
    parser.add_argument('seed_solution')
    parser.add_argument('--output', required=True)
    parser.add_argument('--time-limit', type=float, default=300.0)
    parser.add_argument('--max-iterations', type=int, default=100)
    parser.add_argument('--operators', default='swap,relocate')
    args = parser.parse_args()

    instance = read_instance(args.instance)
    seed = read_solution(args.seed_solution)
    validation = validate_solution(instance, seed)
    if not validation.is_feasible:
        raise ValueError('Seed solution is infeasible: ' + '; '.join(validation.errors))

    routes = [Route(route.depot_id, list(route.customer_ids)) for route in seed.routes]
    route_loads = [route_load(instance, route) for route in routes]
    depot_loads = {depot.id: 0.0 for depot in instance.depots}
    for route, load in zip(routes, route_loads):
        depot_loads[route.depot_id] += load
    matrix = _distance_matrix(instance)
    customer_nodes = {
        customer.id: instance.node_index[('customer', customer.id)]
        for customer in instance.customers
    }
    demands = {customer.id: customer.demand for customer in instance.customers}
    operators = [item.strip() for item in args.operators.split(',') if item.strip()]
    unknown = set(operators) - {'swap', 'relocate'}
    if unknown:
        raise ValueError(f'Unknown operators: {sorted(unknown)}')

    initial_cost = objective_cost(instance, seed)
    current_cost = initial_cost
    deadline = time.perf_counter() + max(0.01, args.time_limit)
    accepted = 0
    for iteration in range(max(0, args.max_iterations)):
        if time.perf_counter() >= deadline:
            break
        improved = False
        ordered = operators[iteration % len(operators) :] + operators[: iteration % len(operators)]
        for operator in ordered:
            if operator == 'swap':
                move = _best_cross_depot_swap(
                    instance,
                    routes,
                    route_loads,
                    depot_loads,
                    matrix,
                    customer_nodes,
                    demands,
                    deadline,
                )
                if move is not None:
                    _apply_swap(routes, route_loads, depot_loads, demands, move)
            else:
                move = _best_cross_depot_relocate(
                    instance,
                    routes,
                    route_loads,
                    depot_loads,
                    matrix,
                    customer_nodes,
                    demands,
                    deadline,
                )
                if move is not None:
                    _apply_relocate(routes, route_loads, depot_loads, demands, move)
            if move is not None:
                current_cost += move[0]
                accepted += 1
                improved = True
                print(
                    f'iteration={iteration + 1} operator={operator} '
                    f'delta={move[0]:.2f} estimated_cost={current_cost:.2f}',
                    flush=True,
                )
                break
        if not improved:
            break

    solution = Solution(instance.name, routes)
    final_validation = validate_solution(instance, solution)
    if not final_validation.is_feasible:
        raise RuntimeError('Local-search output is infeasible: ' + '; '.join(final_validation.errors))
    write_solution(solution, Path(args.output), instance=instance)
    final_cost = objective_cost(instance, solution)
    print(f'solution: {args.output}')
    print('feasible: true')
    print(f'initial_cost: {initial_cost:.10f}')
    print(f'cost: {final_cost:.10f}')
    print(f'improvement: {initial_cost - final_cost:.10f}')
    print(f'accepted_moves: {accepted}')
    print(f'routes: {len(solution.routes)}')
    print(f'depots_opened: {len(solution.opened_depot_ids)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
