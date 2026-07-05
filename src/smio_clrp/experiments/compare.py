from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def compare_algorithms(summary_csv: str | Path, output_csv: str | Path | None = None) -> list[dict[str, Any]]:
    with Path(summary_csv).open(newline="", encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if row.get("feasible") in {"True", "true", "1"}]
    by_instance: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_instance.setdefault(row["instance"], []).append(row)
    output: list[dict[str, Any]] = []
    for instance, instance_rows in sorted(by_instance.items()):
        costs = {
            row["algorithm"]: float(row["cost"])
            for row in instance_rows
            if row.get("cost") not in {"", None}
        }
        runtimes = {
            row["algorithm"]: float(row["runtime_seconds"])
            for row in instance_rows
            if row.get("runtime_seconds") not in {"", None}
        }
        if not costs:
            continue
        best_algorithm = min(costs, key=costs.get)
        best_cost = costs[best_algorithm]
        gaps = {algorithm: 100.0 * (cost - best_cost) / best_cost for algorithm, cost in costs.items()}
        output.append(
            {
                "instance": instance,
                "best_algorithm": best_algorithm,
                "best_cost": best_cost,
                "algorithm_costs": json.dumps(costs, sort_keys=True),
                "gaps_to_best": json.dumps(gaps, sort_keys=True),
                "runtimes": json.dumps(runtimes, sort_keys=True),
            }
        )
    if output_csv:
        _write_rows(output_csv, output, ["instance", "best_algorithm", "best_cost", "algorithm_costs", "gaps_to_best", "runtimes"])
    return output


def compare_to_reference(
    best_summary_csv: str | Path,
    reference_csv: str | Path,
    output_csv: str | Path | None = None,
) -> list[dict[str, Any]]:
    our = _read_costs(best_summary_csv)
    reference = _read_costs(reference_csv)
    rows = []
    for instance, reference_cost in sorted(reference.items()):
        our_cost = our.get(instance)
        gap = None if our_cost is None else 100.0 * (our_cost - reference_cost) / reference_cost
        rows.append(
            {
                "instance": instance,
                "our_cost": our_cost,
                "reference_cost": reference_cost,
                "gap_percent": gap,
            }
        )
    if output_csv:
        _write_rows(output_csv, rows, ["instance", "our_cost", "reference_cost", "gap_percent"])
    return rows


def _read_costs(path: str | Path) -> dict[str, float]:
    costs: dict[str, float] = {}
    with Path(path).open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if not row.get("cost"):
                continue
            value = float(row["cost"])
            instance = row["instance"]
            costs[instance] = min(costs.get(instance, value), value)
    return costs


def _write_rows(path: str | Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
