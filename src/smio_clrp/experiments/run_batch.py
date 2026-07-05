from __future__ import annotations

import csv
from pathlib import Path

from smio_clrp.experiments.run_single import run_single


def run_batch(instance_dir: str | Path, output_dir: str | Path, seed: int = 1) -> list[dict[str, object]]:
    instance_dir = Path(instance_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for instance_path in sorted(instance_dir.glob("*.txt")):
        output_path = output_dir / f"{instance_path.stem}.sol"
        rows.append(run_single(instance_path, output_path, seed=seed))

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["instance", "cost", "feasible", "runtime", "algorithm", "seed", "error"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return rows
