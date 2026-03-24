"""Saves the trained sklearn pipeline as a pickle bundle with metadata."""

import pickle

from config.settings import MODEL_PATH
from app.logger import get_logger

logger = get_logger(__name__)


def model_save(pipeline, metrics: dict, version: str, run_id: str) -> None:
    """Save pipeline and metadata as a pickle bundle to MODEL_PATH."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "pipeline": pipeline,
        "version":  version,
        "metrics":  metrics,
        "run_id":   run_id,
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)

    logger.info(f"Model bundle saved → {MODEL_PATH}")
