#!/usr/bin/env python3
"""Iterated Local Search: repeatedly polish the current best solution to a local
optimum (improve_solution with all operators, including or_opt), then apply one
random destroy+repair perturbation (reusing the ALNS building blocks directly) to
escape that local optimum, and polish again. Keeps the best solution seen across all
rounds. Checkpoints (writes --output) on every improvement, not just at the end, so
an interrupted run never loses progress.

Usage:
    python scripts/ils_polish.py data/official/clrp-small-04.txt results/competition/best.sol \
        --output results/competition/best_ils.sol --rounds 200 --time-limit 900 --seed 1
"""
from __future__ import annotations

import argparse
import gc
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smio_clrp.algorithms.alns.config import ALNSConfig
from smio_clrp.algorithms.alns.destroy import DESTROY_OPERATORS
from smio_clrp.algorithms.alns.repair import REPAIR_OPERATORS
from smio_clrp.algorithms.common import clone_solution
from smio_clrp.algorithms.local_search.driver import improve_solution
from smio_clrp.evaluation.cost import objective_cost
from smio_clrp.evaluation.validator import validate_solution
from smio_clrp.io.instance_reader import read_instance
from smio_clrp.io.solution_reader import read_solution
from smio_clrp.io.solution_writer import write_solution

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("instance_path")
parser.add_argument("solution_path")
parser.add_argument("--output", required=True)
parser.add_argument("--rounds", type=int, default=30)
parser.add_argument("--time-limit", type=float, default=600.0)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--destroy-fraction-min", type=float, default=0.1)
parser.add_argument("--destroy-fraction-max", type=float, default=0.3)
args = parser.parse_args()

instance = read_instance(args.instance_path)
solution = read_solution(args.solution_path)
validation = validate_solution(instance, solution)
if not validation.is_feasible:
    print(f"ERROR: initial solution infeasible: {validation.errors}")
    raise SystemExit(1)

rng = random.Random(args.seed)
config = ALNSConfig(
    seed=args.seed,
    destroy_fraction_min=args.destroy_fraction_min,
    destroy_fraction_max=args.destroy_fraction_max,
)
destroy_names = list(DESTROY_OPERATORS)
repair_names = list(REPAIR_OPERATORS)

best = clone_solution(solution)
best_cost = objective_cost(instance, best)
current = clone_solution(best)
current_cost = best_cost
print(f"initial: {best_cost:.2f}")

t0 = time.perf_counter()
round_num = 0
while round_num < args.rounds and time.perf_counter() - t0 < args.time_limit:
    round_num += 1
    if round_num > 1:
        destroy_name = rng.choice(destroy_names)
        repair_name = rng.choice(repair_names)
        try:
            destroy_result = DESTROY_OPERATORS[destroy_name](instance, current, rng, config)
            repair_result = REPAIR_OPERATORS[repair_name](
                instance, destroy_result.partial_solution, destroy_result.removed_customer_ids, rng
            )
            if not repair_result.success or repair_result.solution is None:
                continue
            perturbed = repair_result.solution
            perturbed_validation = validate_solution(instance, perturbed)
        except Exception as exc:
            # A transient failure here (including a MemoryError under memory pressure
            # from many hours of accumulated allocations on large instances) must not
            # abort a run that may still have hours of budget left -- treat it like any
            # other failed perturbation and keep going with the current incumbent.
            print(f"round {round_num}: perturbation failed ({exc}), skipping")
            continue
        if not perturbed_validation.is_feasible:
            continue
        current = perturbed

    remaining_time = args.time_limit - (time.perf_counter() - t0)
    if remaining_time <= 0:
        break
    try:
        polished = improve_solution(instance, current, max_iterations=200, time_limit_seconds=min(60.0, remaining_time))
        polished_cost = objective_cost(instance, polished)
        polished_validation = validate_solution(instance, polished)
    except Exception as exc:
        print(f"round {round_num}: polishing failed ({exc}), skipping")
        continue
    if not polished_validation.is_feasible:
        continue

    if round_num % 20 == 0:
        gc.collect()

    if polished_cost + 1e-6 < best_cost:
        best = clone_solution(polished)
        best_cost = polished_cost
        current = clone_solution(polished)
        current_cost = polished_cost
        write_solution(best, args.output, instance=instance)
        print(f"round {round_num}: NEW BEST {best_cost:.2f} (t={time.perf_counter()-t0:.1f}s) [checkpointed]")
    elif polished_cost + 1e-6 < current_cost + (current_cost * 0.02):
        current = clone_solution(polished)
        current_cost = polished_cost
    else:
        current = clone_solution(best)
        current_cost = best_cost

print(f"FINAL: {best_cost:.2f} after {round_num} rounds, {time.perf_counter()-t0:.1f}s")
write_solution(best, args.output, instance=instance)
print(f"written: {args.output}")
