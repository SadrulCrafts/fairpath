"""One logging setup for the CLI, the pipeline and the API."""

from __future__ import annotations

import logging
import os
import warnings


def setup_logging(level: str | None = None) -> None:
    level = level or os.getenv("FAIRPATH_LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # third-party chatter that says nothing useful at INFO level
    for noisy in ("optuna", "shap", "matplotlib", "lightgbm", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
    warnings.filterwarnings("ignore", message=".*X does not have valid feature names.*")
