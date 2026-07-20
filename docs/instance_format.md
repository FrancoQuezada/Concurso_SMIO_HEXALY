# Instance and Solution Formats

The parser supports the official-style plain-text format used by the SMIO-Hexaly CLRP specification.

## Instance Format

Supported headers:

- `NAME`
- `CUSTOMERS`
- `DEPOTS`
- `VEHICLE_CAPACITY`
- `ROUTE_FIXED_COST`
- `DISTANCE_FORMAT`

Supported sections:

- `DEPOT_SECTION`
- `CUSTOMER_SECTION`
- `DISTANCE_SECTION` for `FULL_MATRIX`
- `EOF`

Lines beginning with `#` are ignored in instance files. Arbitrary whitespace is accepted.

Depot rows follow the official spec (section 4.1):

```text
depot_id x y opening_cost capacity max_vehicles
```

Customer rows are:

```text
customer_id x y demand
```

`x y` are always present, regardless of `DISTANCE_FORMAT` — for `FULL_MATRIX` instances they carry the underlying geographic coordinates even though route costs come from the explicit matrix, not from these coordinates. For `DISTANCE_FORMAT : FULL_MATRIX`, the distance matrix must have size `(depots + customers) x (depots + customers)` and use node order depots first, then customers.

## Distance Convention

For `COORDS`, each pairwise Euclidean distance is rounded to one decimal place before route costs are summed:

```text
round(sqrt((x_a - x_b)^2 + (y_a - y_b)^2), 1)
```

Route totals and global objective totals are not rounded.

For `FULL_MATRIX`, distances are read directly from the matrix and may be asymmetric.

## Solution Format

Supported fields:

```text
# instance = <instance_name>
COST : <total_cost>
DEPOTS_OPENED : <number_of_open_depots>
ROUTES : <total_number_of_routes>
DEPOT <depot_id>
ROUTE : <customer_id_1> <customer_id_2> ...
EOF
```

The solution reader preserves the reported cost. The writer recomputes the objective by default when an instance is provided.

## Validation Rules

A solution is feasible only if:

- every customer appears exactly once;
- no unknown customer or depot is referenced;
- every route is non-empty;
- every route demand is at most `VEHICLE_CAPACITY`;
- depot assigned demand is at most depot capacity;
- number of routes assigned to each depot is at most its vehicle limit;
- reported cost matches recomputed cost within `1e-4` when a reported cost exists.
