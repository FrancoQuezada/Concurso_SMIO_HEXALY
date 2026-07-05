from __future__ import annotations

import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from smio_clrp.evaluation.validator import validate_solution
from smio_clrp.io.instance_reader import read_instance
from smio_clrp.io.solution_reader import read_solution


def build_submission_bundle(
    instance_dir: str | Path,
    solution_dir: str | Path,
    output_zip: str | Path | None = None,
) -> Path:
    instance_dir = Path(instance_dir)
    solution_dir = Path(solution_dir)
    output_zip = Path(output_zip or _default_zip_path())
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    instances = {read_instance(path).name: path for path in sorted(instance_dir.glob("*.txt"))}
    included: list[str] = []
    skipped: list[str] = []
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for solution_path in sorted(solution_dir.rglob("*.sol")):
            instance_name = _instance_name(solution_path)
            instance_path = instances.get(instance_name)
            if instance_path is None:
                skipped.append(str(solution_path))
                continue
            validation = validate_solution(read_instance(instance_path), read_solution(solution_path))
            if not validation.is_feasible:
                skipped.append(str(solution_path))
                continue
            archive.write(solution_path, arcname=solution_path.name)
            included.append(str(solution_path))
        metadata = {"included": included, "skipped": skipped}
        archive.writestr("bundle_metadata.json", json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return output_zip


def _default_zip_path() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"submissions/submission_{stamp}.zip"


def _instance_name(path: Path) -> str:
    return re.sub(r"_seed\d+$", "", path.stem)
