from __future__ import annotations

from pathlib import Path

from smio_clrp.experiments.metadata import read_json
from smio_clrp.experiments.run_batch import BatchConfig, run_batch


def load_batch_config(path: str | Path) -> BatchConfig:
    data = read_json(path)
    return BatchConfig(
        instance_dir=data["instance_dir"],
        output_dir=data.get("output_dir", "results/runs"),
        algorithms=list(data.get("algorithms", ["constructive_ls"])),
        seeds=[int(seed) for seed in data.get("seeds", [1])],
        time_limits=data.get("time_limits", {}),
        algorithm_parameters=data.get("algorithm_parameters", {}),
        validate_outputs=bool(data.get("validate_outputs", True)),
        save_metadata=bool(data.get("save_metadata", True)),
        overwrite=bool(data.get("overwrite", False)),
        run_id=data.get("run_id"),
    )


def run_benchmark_from_config(path: str | Path):
    return run_batch(load_batch_config(path))
