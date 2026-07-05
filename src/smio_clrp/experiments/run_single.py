from __future__ import annotations

from pathlib import Path

from smio_clrp.algorithms.constructive.greedy import solve_greedy_nearest_depot
from smio_clrp.io.instance_reader import read_instance
from smio_clrp.io.solution_writer import write_solution


def run_single(instance_path: str | Path, output_path: str | Path, seed: int = 1) -> dict[str, object]:
    instance = read_instance(instance_path)
    result = solve_greedy_nearest_depot(instance, seed=seed)
    if result.solution is None:
        return {
            "instance": instance.name,
            "cost": None,
            "feasible": False,
            "runtime": result.runtime_seconds,
            "algorithm": result.algorithm_name,
            "seed": seed,
            "error": result.metadata.get("error"),
        }
    write_solution(result.solution, output_path, instance=instance)
    return {
        "instance": instance.name,
        "cost": result.cost,
        "feasible": result.feasible,
        "runtime": result.runtime_seconds,
        "algorithm": result.algorithm_name,
        "seed": seed,
    }
