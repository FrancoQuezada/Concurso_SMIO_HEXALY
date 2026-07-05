from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any

from smio_clrp.evaluation.validator import validate_solution
from smio_clrp.io.instance_reader import read_instance
from smio_clrp.io.solution_reader import read_solution


def update_best_registry(
    instance_dir: str | Path,
    run_dir: str | Path,
    best_dir: str | Path = "results/best",
    replace_ties: bool = False,
) -> list[dict[str, Any]]:
    instance_dir = Path(instance_dir)
    run_dir = Path(run_dir)
    best_dir = Path(best_dir)
    best_dir.mkdir(parents=True, exist_ok=True)
    instances = _instances_by_name(instance_dir)
    summary_rows = _candidate_rows(run_dir)
    accepted: list[dict[str, Any]] = []
    for row in summary_rows:
        solution_path = Path(row.get("solution_path", ""))
        if not solution_path.exists():
            continue
        instance_name = row.get("instance") or _instance_name_from_solution(solution_path)
        if instance_name not in instances:
            continue
        instance = read_instance(instances[instance_name])
        solution = read_solution(solution_path)
        validation = validate_solution(instance, solution)
        if not validation.is_feasible:
            continue
        target = best_dir / f"{instance_name}.sol"
        incumbent_cost = _incumbent_cost(target, instance)
        better = incumbent_cost is None or validation.cost < incumbent_cost - 1e-4
        tie = incumbent_cost is not None and abs(validation.cost - incumbent_cost) <= 1e-4
        if better or (replace_ties and tie):
            shutil.copyfile(solution_path, target)
            accepted.append(
                {
                    "instance": instance_name,
                    "cost": validation.cost,
                    "solution_path": str(target),
                    "source_solution_path": str(solution_path),
                    "metadata_path": row.get("metadata_path", ""),
                }
            )
    _write_best_summary(best_dir, instances)
    _write_best_metadata(best_dir, accepted)
    return accepted


def list_best(best_dir: str | Path = "results/best") -> list[dict[str, str]]:
    best_dir = Path(best_dir)
    summary_path = best_dir / "best_summary.csv"
    if not summary_path.exists():
        return []
    with summary_path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _candidate_rows(run_dir: Path) -> list[dict[str, str]]:
    summary_path = run_dir / "summary.csv"
    if summary_path.exists():
        with summary_path.open(newline="", encoding="utf-8") as file:
            return list(csv.DictReader(file))
    rows = []
    for solution_path in sorted((run_dir / "solutions").rglob("*.sol")):
        rows.append(
            {
                "instance": _instance_name_from_solution(solution_path),
                "solution_path": str(solution_path),
                "metadata_path": "",
            }
        )
    return rows


def _instances_by_name(instance_dir: Path) -> dict[str, Path]:
    instances = {}
    for path in sorted(instance_dir.glob("*.txt")):
        instance = read_instance(path)
        instances[instance.name] = path
    return instances


def _instance_name_from_solution(path: Path) -> str:
    stem = path.stem
    return re.sub(r"_seed\d+$", "", stem)


def _incumbent_cost(path: Path, instance) -> float | None:
    if not path.exists():
        return None
    validation = validate_solution(instance, read_solution(path))
    return validation.cost if validation.is_feasible else None


def _write_best_summary(best_dir: Path, instances: dict[str, Path]) -> None:
    rows = []
    for instance_name, instance_path in sorted(instances.items()):
        solution_path = best_dir / f"{instance_name}.sol"
        if not solution_path.exists():
            continue
        instance = read_instance(instance_path)
        validation = validate_solution(instance, read_solution(solution_path))
        if validation.is_feasible:
            rows.append(
                {
                    "instance": instance_name,
                    "cost": f"{validation.cost:.10f}",
                    "solution_path": str(solution_path),
                }
            )
    with (best_dir / "best_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["instance", "cost", "solution_path"])
        writer.writeheader()
        writer.writerows(rows)


def _write_best_metadata(best_dir: Path, accepted: list[dict[str, Any]]) -> None:
    (best_dir / "best_metadata.json").write_text(
        json.dumps({"accepted": accepted}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
