from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from smio_clrp.algorithms.fixopt.config import FixOptConfig
from smio_clrp.algorithms.fixopt.neighborhoods import FixOptNeighborhood
from smio_clrp.core.instance import Instance
from smio_clrp.core.solution import Solution


class BackendUnavailable(RuntimeError):
    pass


@dataclass
class FixOptResult:
    solution: Solution | None
    success: bool
    backend_used: str
    metadata: dict[str, Any] = field(default_factory=dict)


class FixOptBackend(Protocol):
    backend_name: str

    def reoptimize(
        self,
        instance: Instance,
        solution: Solution,
        neighborhood: FixOptNeighborhood,
        config: FixOptConfig,
    ) -> FixOptResult:
        ...
