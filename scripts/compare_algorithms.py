#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smio_clrp.experiments.compare import compare_algorithms, compare_to_reference


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--reference-csv", default=None)
    args = parser.parse_args()
    if args.reference_csv:
        rows = compare_to_reference(args.summary_csv, args.reference_csv, args.output_csv)
    else:
        rows = compare_algorithms(args.summary_csv, args.output_csv)
    print(f"rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
