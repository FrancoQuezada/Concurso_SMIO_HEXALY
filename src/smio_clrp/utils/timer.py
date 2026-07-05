from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


@dataclass
class Timer:
    elapsed_seconds: float = 0.0


@contextmanager
def timed() -> Iterator[Timer]:
    timer = Timer()
    start = time.perf_counter()
    try:
        yield timer
    finally:
        timer.elapsed_seconds = time.perf_counter() - start
