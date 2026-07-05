from __future__ import annotations

import logging


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(format="%(levelname)s:%(name)s:%(message)s", level=level)
