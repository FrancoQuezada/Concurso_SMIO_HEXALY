# Experiments

Official instances are not available yet. Put future official `.txt` instances under `data/official/`. The existing files in `data/samples/` are tiny artificial development samples for tests and smoke checks only.

## Directory Structure

- `configs/`: JSON experiment configs.
- `results/runs/<run_id>/solutions/<algorithm>/`: official-format solution files.
- `results/runs/<run_id>/metadata/<algorithm>/`: JSON metadata per run.
- `results/runs/<run_id>/summary.csv`: batch summary.
- `results/best/`: validated local best solutions and registry summaries.
- `submissions/`: zip bundles built from validated solutions.

Generated results and bundles are ignored by Git.

## Run a Benchmark

```bash
python scripts/run_benchmark.py --config configs/smoke_samples.json
```

The config format supports `instance_dir`, `output_dir`, `algorithms`, `seeds`, `time_limits`, `algorithm_parameters`, `validate_outputs`, `save_metadata`, and `overwrite`.

## Batch Solve

```bash
clrp batch-solve data/samples --output-dir results/runs/smoke_alns --algorithm alns --seed 1 --time-limit 5 --max-iterations 100
```

The runner continues after individual failures and records errors in `summary.csv`.

## Metadata

Each metadata JSON includes the instance, algorithm, seed, solver config, timestamps, runtime, feasibility, cost, validation messages, solver metadata, git commit, Python version, and platform.

## Best Registry

```bash
python scripts/update_best_registry.py --instance-dir data/samples --run-dir results/runs/smoke_alns --best-dir results/best
```

The registry validates candidate solutions and compares recomputed costs. It keeps incumbents on ties unless `--replace-ties` is used.

## Compare Algorithms

```bash
python scripts/compare_algorithms.py --summary-csv results/runs/smoke_samples/summary.csv --output-csv results/runs/smoke_samples/compare.csv
```

Reference comparisons can be run later with a CSV containing `instance,cost`.

## Build a Submission Bundle

```bash
python scripts/build_submission_bundle.py --instance-dir data/samples --solution-dir results/best --output submissions/smoke_bundle.zip
```

The bundle helper validates every solution against its matching instance and includes only feasible official-format `.sol` files.
