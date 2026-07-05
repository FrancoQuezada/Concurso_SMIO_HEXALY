from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from smio_clrp.experiments.run_single import RunResult, SUMMARY_FIELDS, run_single


@dataclass
class BatchConfig:
    instance_dir: str | Path
    output_dir: str | Path = "results/runs"
    algorithms: list[str] = field(default_factory=lambda: ["constructive_ls"])
    seeds: list[int] = field(default_factory=lambda: [1])
    time_limits: dict[str, float | None] = field(default_factory=dict)
    algorithm_parameters: dict[str, dict[str, Any]] = field(default_factory=dict)
    validate_outputs: bool = True
    save_metadata: bool = True
    overwrite: bool = False
    run_id: str | None = None


def make_run_id(prefix: str = "run") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}"


def run_batch(config: BatchConfig) -> list[RunResult]:
    instance_dir = Path(config.instance_dir)
    run_id = config.run_id or make_run_id("benchmark")
    output_root = Path(config.output_dir)
    run_dir = output_root if output_root.name == run_id else output_root / run_id
    solutions_root = run_dir / "solutions"
    metadata_root = run_dir / "metadata"
    run_dir.mkdir(parents=True, exist_ok=True)

    rows: list[RunResult] = []
    instance_paths = sorted(path for path in instance_dir.glob("*.txt") if path.is_file())
    for instance_path in instance_paths:
        instance_stem = instance_path.stem
        for algorithm in config.algorithms:
            for seed in config.seeds:
                solution_path = solutions_root / algorithm / f"{instance_stem}_seed{seed}.sol"
                metadata_path = metadata_root / algorithm / f"{instance_stem}_seed{seed}.json"
                if solution_path.exists() and not config.overwrite:
                    continue
                result = run_single(
                    instance_path=instance_path,
                    algorithm=algorithm,
                    seed=seed,
                    time_limit=config.time_limits.get(algorithm),
                    solver_parameters=config.algorithm_parameters.get(algorithm, {}),
                    output_solution_path=solution_path,
                    metadata_path=metadata_path,
                    run_id=run_id,
                    save_metadata=config.save_metadata,
                )
                rows.append(result)

    _write_summary(run_dir / "summary.csv", rows)
    return rows


def _write_summary(path: Path, rows: list[RunResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.summary_row())
