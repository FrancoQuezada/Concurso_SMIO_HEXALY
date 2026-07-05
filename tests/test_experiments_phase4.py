import csv
import json
import subprocess
import sys
import zipfile
from pathlib import Path

from smio_clrp.experiments.benchmark import run_benchmark_from_config
from smio_clrp.experiments.compare import compare_algorithms, compare_to_reference
from smio_clrp.experiments.registry import update_best_registry
from smio_clrp.experiments.run_batch import BatchConfig, run_batch
from smio_clrp.experiments.run_single import run_single
from smio_clrp.experiments.submission import build_submission_bundle
from smio_clrp.io.instance_reader import read_instance
from smio_clrp.io.solution_reader import read_solution
from smio_clrp.evaluation.validator import validate_solution


def test_single_run_solves_sample_and_writes_solution_and_metadata(tmp_path):
    solution_path = tmp_path / "tiny_coords.sol"
    metadata_path = tmp_path / "tiny_coords.json"

    result = run_single(
        "data/samples/tiny_coords.txt",
        algorithm="constructive_ls",
        seed=1,
        time_limit=5,
        solver_parameters={"num_starts": 3, "max_iterations": 10},
        output_solution_path=solution_path,
        metadata_path=metadata_path,
        run_id="test_single",
    )

    assert result.feasible
    assert solution_path.exists()
    assert metadata_path.exists()
    instance = read_instance("data/samples/tiny_coords.txt")
    assert validate_solution(instance, read_solution(solution_path)).is_feasible
    assert json.loads(metadata_path.read_text())["algorithm"] == "constructive_ls"


def test_batch_runner_runs_two_algorithms_and_writes_outputs(tmp_path):
    rows = run_batch(
        BatchConfig(
            instance_dir="data/samples",
            output_dir=tmp_path,
            run_id="batch",
            algorithms=["constructive_ls", "alns"],
            seeds=[1],
            time_limits={"constructive_ls": 5, "alns": 5},
            algorithm_parameters={
                "constructive_ls": {"num_starts": 3, "max_iterations": 10},
                "alns": {"num_starts": 3, "max_iterations": 20},
            },
            overwrite=True,
        )
    )

    run_dir = tmp_path / "batch"
    assert len(rows) == 4
    assert (run_dir / "summary.csv").exists()
    assert list((run_dir / "solutions").rglob("*.sol"))
    assert list((run_dir / "metadata").rglob("*.json"))


def test_batch_runner_records_failure_without_crashing(tmp_path):
    rows = run_batch(
        BatchConfig(
            instance_dir="data/samples",
            output_dir=tmp_path,
            run_id="failure",
            algorithms=["not_an_algorithm"],
            seeds=[1],
            overwrite=True,
        )
    )

    assert rows
    assert not rows[0].feasible
    assert rows[0].error_message


def test_benchmark_config_runs(tmp_path):
    config = {
        "instance_dir": "data/samples",
        "output_dir": str(tmp_path),
        "run_id": "benchmark",
        "algorithms": ["constructive_ls", "alns"],
        "seeds": [1],
        "time_limits": {"constructive_ls": 5, "alns": 5},
        "algorithm_parameters": {
            "constructive_ls": {"num_starts": 3, "max_iterations": 10},
            "alns": {"num_starts": 3, "max_iterations": 20},
        },
        "overwrite": True,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    rows = run_benchmark_from_config(config_path)

    assert len(rows) == 4
    assert (tmp_path / "benchmark" / "summary.csv").exists()


def test_registry_accepts_feasible_solution_and_writes_best(tmp_path):
    rows = run_batch(
        BatchConfig(
            instance_dir="data/samples",
            output_dir=tmp_path,
            run_id="registry",
            algorithms=["constructive_ls"],
            seeds=[1],
            algorithm_parameters={"constructive_ls": {"num_starts": 3}},
            overwrite=True,
        )
    )
    assert rows

    accepted = update_best_registry("data/samples", tmp_path / "registry", tmp_path / "best")

    assert accepted
    assert (tmp_path / "best" / "tiny_coords.sol").exists()
    assert (tmp_path / "best" / "best_summary.csv").exists()
    assert (tmp_path / "best" / "best_metadata.json").exists()


def test_registry_rejects_infeasible_solution(tmp_path):
    run_dir = tmp_path / "bad_run"
    solution_dir = run_dir / "solutions" / "bad"
    solution_dir.mkdir(parents=True)
    bad_solution = solution_dir / "tiny_coords_seed1.sol"
    bad_solution.write_text(
        """# instance = tiny_coords
COST : 0
DEPOTS_OPENED : 1
ROUTES : 1
DEPOT 1
ROUTE : 101 101
EOF
""",
        encoding="utf-8",
    )
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["instance", "solution_path", "metadata_path"])
        writer.writeheader()
        writer.writerow({"instance": "tiny_coords", "solution_path": str(bad_solution), "metadata_path": ""})

    accepted = update_best_registry("data/samples", run_dir, tmp_path / "best")

    assert accepted == []
    assert not (tmp_path / "best" / "tiny_coords.sol").exists()


def test_registry_keeps_better_cost_and_keeps_tie_by_default(tmp_path):
    run_batch(
        BatchConfig(
            instance_dir="data/samples",
            output_dir=tmp_path,
            run_id="first",
            algorithms=["constructive_ls"],
            seeds=[1],
            algorithm_parameters={"constructive_ls": {"num_starts": 3}},
            overwrite=True,
        )
    )
    first = update_best_registry("data/samples", tmp_path / "first", tmp_path / "best")
    before = (tmp_path / "best" / "tiny_coords.sol").read_text()
    second = update_best_registry("data/samples", tmp_path / "first", tmp_path / "best")
    after = (tmp_path / "best" / "tiny_coords.sol").read_text()

    assert first
    assert second == []
    assert before == after


def test_compare_algorithms_and_reference_gap(tmp_path):
    summary = tmp_path / "summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["instance", "algorithm", "feasible", "cost", "runtime_seconds"])
        writer.writeheader()
        writer.writerow({"instance": "i1", "algorithm": "a", "feasible": "True", "cost": "10", "runtime_seconds": "1"})
        writer.writerow({"instance": "i1", "algorithm": "b", "feasible": "True", "cost": "12", "runtime_seconds": "2"})
    reference = tmp_path / "reference.csv"
    reference.write_text("instance,cost\ni1,8\n", encoding="utf-8")

    comparison = compare_algorithms(summary)
    gaps = compare_to_reference(summary, reference)

    assert comparison[0]["best_algorithm"] == "a"
    assert gaps[0]["gap_percent"] == 25.0


def test_submission_bundle_includes_only_feasible_solutions(tmp_path):
    run_batch(
        BatchConfig(
            instance_dir="data/samples",
            output_dir=tmp_path,
            run_id="bundle_run",
            algorithms=["constructive_ls"],
            seeds=[1],
            algorithm_parameters={"constructive_ls": {"num_starts": 3}},
            overwrite=True,
        )
    )
    update_best_registry("data/samples", tmp_path / "bundle_run", tmp_path / "best")
    bad = tmp_path / "best" / "bad.sol"
    bad.write_text("COST : 0\nDEPOTS_OPENED : 0\nROUTES : 0\nEOF\n", encoding="utf-8")

    bundle = build_submission_bundle("data/samples", tmp_path / "best", tmp_path / "bundle.zip")

    with zipfile.ZipFile(bundle) as archive:
        names = set(archive.namelist())
    assert "tiny_coords.sol" in names
    assert "bad.sol" not in names


def test_main_scripts_smoke(tmp_path):
    config = {
        "instance_dir": "data/samples",
        "output_dir": str(tmp_path / "runs"),
        "run_id": "script_smoke",
        "algorithms": ["constructive_ls"],
        "seeds": [1],
        "algorithm_parameters": {"constructive_ls": {"num_starts": 3}},
        "overwrite": True,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    subprocess.run([sys.executable, "scripts/run_benchmark.py", "--config", str(config_path)], check=True)
    subprocess.run(
        [
            sys.executable,
            "scripts/validate_solution_dir.py",
            "--instance-dir",
            "data/samples",
            "--solution-dir",
            str(tmp_path / "runs" / "script_smoke" / "solutions"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/update_best_registry.py",
            "--instance-dir",
            "data/samples",
            "--run-dir",
            str(tmp_path / "runs" / "script_smoke"),
            "--best-dir",
            str(tmp_path / "best"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/build_submission_bundle.py",
            "--instance-dir",
            "data/samples",
            "--solution-dir",
            str(tmp_path / "best"),
            "--output",
            str(tmp_path / "bundle.zip"),
        ],
        check=True,
    )
    assert (tmp_path / "bundle.zip").exists()
