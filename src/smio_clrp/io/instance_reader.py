from __future__ import annotations

from pathlib import Path

import numpy as np

from smio_clrp.core.instance import Customer, Depot, Instance

SECTION_NAMES = {"DEPOT_SECTION", "CUSTOMER_SECTION", "DISTANCE_SECTION"}


class InstanceFormatError(ValueError):
    """Raised when an instance file does not match the supported plain-text format."""


def read_instance(path: str | Path) -> Instance:
    path = Path(path)
    return parse_instance_text(path.read_text(encoding="utf-8"))


def parse_instance_text(text: str) -> Instance:
    headers: dict[str, str] = {}
    depots: list[Depot] = []
    customers: list[Customer] = []
    matrix_rows: list[list[float]] = []
    section: str | None = None

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = _clean_line(raw_line)
        if not line:
            continue
        upper = line.upper()
        if upper == "EOF":
            break
        if upper in SECTION_NAMES:
            section = upper
            continue

        if section is None:
            key, value = _parse_header(line, line_number)
            headers[key] = value
        elif section == "DEPOT_SECTION":
            depots.append(_parse_depot(line, line_number, headers.get("DISTANCE_FORMAT", "")))
        elif section == "CUSTOMER_SECTION":
            customers.append(_parse_customer(line, line_number, headers.get("DISTANCE_FORMAT", "")))
        elif section == "DISTANCE_SECTION":
            matrix_rows.append(_parse_float_row(line, line_number))
        else:  # pragma: no cover - section is constrained above
            raise InstanceFormatError(f"Unsupported section {section} at line {line_number}")

    required = ["NAME", "CUSTOMERS", "DEPOTS", "VEHICLE_CAPACITY", "ROUTE_FIXED_COST", "DISTANCE_FORMAT"]
    missing = [key for key in required if key not in headers]
    if missing:
        raise InstanceFormatError(f"Missing required headers: {', '.join(missing)}")

    expected_depots = _parse_int(headers["DEPOTS"], "DEPOTS")
    expected_customers = _parse_int(headers["CUSTOMERS"], "CUSTOMERS")
    if len(depots) != expected_depots:
        raise InstanceFormatError(f"Expected {expected_depots} depots, found {len(depots)}")
    if len(customers) != expected_customers:
        raise InstanceFormatError(f"Expected {expected_customers} customers, found {len(customers)}")

    distance_format = headers["DISTANCE_FORMAT"].upper()
    distance_matrix = None
    if distance_format == "FULL_MATRIX":
        expected_size = expected_depots + expected_customers
        if len(matrix_rows) != expected_size:
            raise InstanceFormatError(
                f"DISTANCE_SECTION must contain {expected_size} rows, found {len(matrix_rows)}"
            )
        if any(len(row) != expected_size for row in matrix_rows):
            raise InstanceFormatError(f"DISTANCE_SECTION rows must each contain {expected_size} values")
        distance_matrix = np.asarray(matrix_rows, dtype=float)
    elif distance_format != "COORDS":
        raise InstanceFormatError(f"Unsupported DISTANCE_FORMAT: {distance_format}")

    try:
        return Instance(
            name=headers["NAME"],
            depots=depots,
            customers=customers,
            vehicle_capacity=_parse_float(headers["VEHICLE_CAPACITY"], "VEHICLE_CAPACITY"),
            route_fixed_cost=_parse_float(headers["ROUTE_FIXED_COST"], "ROUTE_FIXED_COST"),
            distance_format=distance_format,  # type: ignore[arg-type]
            distance_matrix=distance_matrix,
        )
    except ValueError as exc:
        raise InstanceFormatError(str(exc)) from exc


def _clean_line(raw_line: str) -> str:
    stripped = raw_line.strip()
    if stripped.startswith("#"):
        return ""
    return stripped.split("#", 1)[0].strip()


def _parse_header(line: str, line_number: int) -> tuple[str, str]:
    if ":" in line:
        key, value = line.split(":", 1)
    else:
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise InstanceFormatError(f"Malformed header at line {line_number}: {line}")
        key, value = parts
    key = key.strip().upper()
    value = value.strip()
    if not key or not value:
        raise InstanceFormatError(f"Malformed header at line {line_number}: {line}")
    return key, value


def _parse_depot(line: str, line_number: int, distance_format: str) -> Depot:
    parts = line.split()
    if len(parts) < 4:
        raise InstanceFormatError(
            f"Depot line {line_number} must contain id opening_cost capacity vehicle_limit"
        )
    if distance_format.upper() == "COORDS" and len(parts) < 6:
        raise InstanceFormatError(f"Depot line {line_number} must include x y coordinates")
    x_y = (_parse_float(parts[4], f"depot x at line {line_number}"), _parse_float(parts[5], f"depot y at line {line_number}")) if len(parts) >= 6 else (None, None)
    return Depot(
        id=_parse_int(parts[0], f"depot id at line {line_number}"),
        opening_cost=_parse_float(parts[1], f"depot opening cost at line {line_number}"),
        capacity=_parse_float(parts[2], f"depot capacity at line {line_number}"),
        vehicle_limit=_parse_int(parts[3], f"depot vehicle limit at line {line_number}"),
        x=x_y[0],
        y=x_y[1],
    )


def _parse_customer(line: str, line_number: int, distance_format: str) -> Customer:
    parts = line.split()
    if len(parts) < 2:
        raise InstanceFormatError(f"Customer line {line_number} must contain id demand")
    if distance_format.upper() == "COORDS" and len(parts) < 4:
        raise InstanceFormatError(f"Customer line {line_number} must include x y coordinates")
    x_y = (_parse_float(parts[2], f"customer x at line {line_number}"), _parse_float(parts[3], f"customer y at line {line_number}")) if len(parts) >= 4 else (None, None)
    return Customer(
        id=_parse_int(parts[0], f"customer id at line {line_number}"),
        demand=_parse_float(parts[1], f"customer demand at line {line_number}"),
        x=x_y[0],
        y=x_y[1],
    )


def _parse_float_row(line: str, line_number: int) -> list[float]:
    return [_parse_float(part, f"distance value at line {line_number}") for part in line.split()]


def _parse_float(value: str, label: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise InstanceFormatError(f"Invalid numeric value for {label}: {value}") from exc
    if number < 0:
        raise InstanceFormatError(f"{label} must be non-negative")
    return number


def _parse_int(value: str, label: str) -> int:
    number = _parse_float(value, label)
    if not number.is_integer():
        raise InstanceFormatError(f"{label} must be an integer")
    return int(number)
