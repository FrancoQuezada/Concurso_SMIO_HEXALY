from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from smio_clrp.evaluation.validator import validate_solution
from smio_clrp.experiments.metadata import runtime_environment, utc_now_iso, write_json
from smio_clrp.experiments.solvers import make_solver
from smio_clrp.io.instance_reader import read_instance
from smio_clrp.io.solution_reader import read_solution
from smio_clrp.io.solution_writer import write_solution


@dataclass
class RunResult:
    run_id: str
    timestamp: str
    instance: str
    instance_path: str
    algorithm: str
    seed: int
    feasible: bool
    cost: float | None
    runtime_seconds: float
    time_limit_seconds: float | None
    solution_path: str
    metadata_path: str
    error_message: str

    def summary_row(self) -> dict[str, object]:
        return asdict(self)


SUMMARY_FIELDS = [
    "run_id",
    "timestamp",
    "instance",
    "instance_path",
    "algorithm",
    "seed",
    "feasible",
    "cost",
    "runtime_seconds",
    "time_limit_seconds",
    "solution_path",
    "metadata_path",
    "error_message",
]


def run_single(
    instance_path: str | Path,
    algorithm: str = "greedy_nearest_depot",
    seed: int = 1,
    time_limit: float | None = None,
    solver_parameters: dict[str, Any] | None = None,
    output_solution_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    run_id: str = "single",
    save_metadata: bool = True,
) -> RunResult:
    instance_path = Path(instance_path)
    start_timestamp = utc_now_iso()
    instance = read_instance(instance_path)
    output_solution_path = Path(output_solution_path or f"{instance.name}_{algorithm}_seed{seed}.sol")
    metadata_path = Path(metadata_path or output_solution_path.with_suffix(".json"))
    parameters = dict(solver_parameters or {})

    feasible = False
    cost: float | None = None
    error_message = ""
    validation_errors: list[str] = []
    validation_warnings: list[str] = []
    solver_metadata: dict[str, Any] = {}
    runtime_seconds = 0.0

    try:
        solver = make_solver(algorithm, seed=seed, time_limit=time_limit, parameters=parameters)
        result = solver.solve(instance)
        runtime_seconds = result.runtime_seconds
        solver_metadata = result.metadata
        if result.solution is None:
            error_message = str(result.metadata.get("error", "solver returned no solution"))
        else:
            write_solution(result.solution, output_solution_path, instance=instance)
            parsed = read_solution(output_solution_path)
            validation = validate_solution(instance, parsed)
            feasible = validation.is_feasible
            cost = validation.cost if feasible else None
            validation_errors = validation.errors
            validation_warnings = validation.warnings
            if not feasible:
                error_message = "; ".join(validation.errors)
    except Exception as exc:
        error_message = str(exc)

    end_timestamp = utc_now_iso()
    metadata = {
        "instance": instance.name,
        "instance_path": str(instance_path),
        "algorithm": algorithm,
        "seed": seed,
        "solver_config": {"time_limit": time_limit, "parameters": parameters},
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
        "runtime_seconds": runtime_seconds,
        "feasible": feasible,
        "cost": cost,
        "validation_errors": validation_errors,
        "validation_warnings": validation_warnings,
        "solver_metadata": solver_metadata,
        **runtime_environment(),
    }
    if save_metadata:
        write_json(metadata_path, metadata)

    return RunResult(
        run_id=run_id,
        timestamp=start_timestamp,
        instance=instance.name,
        instance_path=str(instance_path),
        algorithm=algorithm,
        seed=seed,
        feasible=feasible,
        cost=cost,
        runtime_seconds=runtime_seconds,
        time_limit_seconds=time_limit,
        solution_path=str(output_solution_path) if output_solution_path.exists() else "",
        metadata_path=str(metadata_path) if save_metadata else "",
        error_message=error_message,
    )
