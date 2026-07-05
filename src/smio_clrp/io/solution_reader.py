from __future__ import annotations

from pathlib import Path

from smio_clrp.core.solution import Route, Solution


class SolutionFormatError(ValueError):
    """Raised when a solution file is malformed."""


def read_solution(path: str | Path) -> Solution:
    path = Path(path)
    return parse_solution_text(path.read_text(encoding="utf-8"))


def parse_solution_text(text: str) -> Solution:
    instance_name: str | None = None
    reported_cost: float | None = None
    routes: list[Route] = []
    current_depot: int | None = None
    declared_routes: int | None = None
    declared_depots: int | None = None
    depot_blocks: set[int] = set()

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            maybe_name = line[1:].strip()
            if maybe_name.lower().startswith("instance"):
                instance_name = _parse_instance_comment(maybe_name, line_number)
            continue
        if line.upper() == "EOF":
            break

        key, value = _split_key_value(line, line_number)
        key = key.upper()
        if key == "COST":
            reported_cost = _parse_float(value, "COST", line_number)
        elif key == "DEPOTS_OPENED":
            declared_depots = _parse_int(value, "DEPOTS_OPENED", line_number)
        elif key == "ROUTES":
            declared_routes = _parse_int(value, "ROUTES", line_number)
        elif key == "DEPOT":
            current_depot = _parse_int(value, "DEPOT", line_number)
            depot_blocks.add(current_depot)
        elif key == "ROUTE":
            if current_depot is None:
                raise SolutionFormatError(f"ROUTE before DEPOT at line {line_number}")
            customer_ids = [_parse_int(part, "ROUTE customer", line_number) for part in value.split()]
            routes.append(Route(depot_id=current_depot, customer_ids=customer_ids))
        else:
            raise SolutionFormatError(f"Unknown solution key at line {line_number}: {key}")

    if reported_cost is None:
        raise SolutionFormatError("Missing COST line")
    if declared_routes is not None and declared_routes != len(routes):
        raise SolutionFormatError(f"Declared ROUTES={declared_routes}, parsed {len(routes)}")
    if declared_depots is not None and declared_depots != len(depot_blocks):
        raise SolutionFormatError(
            f"Declared DEPOTS_OPENED={declared_depots}, parsed {len(depot_blocks)}"
        )
    return Solution(instance_name=instance_name, routes=routes, reported_cost=reported_cost)


def _split_key_value(line: str, line_number: int) -> tuple[str, str]:
    if ":" in line:
        key, value = line.split(":", 1)
    else:
        parts = line.split(maxsplit=1)
        if len(parts) == 1 and parts[0].upper() == "ROUTE":
            return "ROUTE", ""
        if len(parts) != 2:
            raise SolutionFormatError(f"Malformed line {line_number}: {line}")
        key, value = parts
    return key.strip(), value.strip()


def _parse_instance_comment(line: str, line_number: int) -> str:
    if "=" in line:
        key, value = line.split("=", 1)
    else:
        key, value = _split_key_value(line, line_number)
    if key.strip().lower() != "instance":
        raise SolutionFormatError(f"Malformed instance comment at line {line_number}: {line}")
    return value.strip()


def _parse_float(value: str, label: str, line_number: int) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise SolutionFormatError(f"Invalid {label} at line {line_number}: {value}") from exc
    if number < 0:
        raise SolutionFormatError(f"{label} at line {line_number} must be non-negative")
    return number


def _parse_int(value: str, label: str, line_number: int) -> int:
    number = _parse_float(value, label, line_number)
    if not number.is_integer():
        raise SolutionFormatError(f"{label} at line {line_number} must be an integer")
    return int(number)
