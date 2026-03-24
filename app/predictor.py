"""Loads the trained model bundle and exposes a predict method."""

import pickle
from pathlib import Path

from config.settings import MODEL_PATH
from app.logger import get_logger

logger = get_logger(__name__)


class Predictor:
    """Loads the model bundle once at startup and serves predictions."""

    def __init__(self):
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run the training pipeline first.")

        with open(MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)

        self.pipeline = bundle["pipeline"]
        self.version  = bundle["version"]
        self.run_id   = bundle["run_id"]
        logger.info(f"Model loaded | version: {self.version} | run_id: {self.run_id}")

    def predict(self, text: str) -> dict:
        """Run prediction on a single text. Returns label and confidence."""
        label_idx  = self.pipeline.predict([text])[0]
        proba      = self.pipeline.predict_proba([text])[0]
        confidence = round(float(proba[label_idx]), 4)
        label      = "spam" if label_idx == 1 else "ham"
        return {"label": label, "confidence": confidence}
