"""Loads the trained model bundle and exposes a predict method."""

import pickle
from pathlib import Path

from app.artifacts import verify_model_sha256
from app.logger import get_logger
from config.settings import MODEL_PATH, MODEL_SHA256, REQUIRE_MODEL_SHA256

logger = get_logger(__name__)


class Predictor:
    """Loads the model bundle once at startup and serves predictions."""

    def __init__(self):
        model_path = Path(MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Run the training pipeline first to create model/model.pkl."
            )

        expected_sha256 = MODEL_SHA256
        if not expected_sha256:
            sha_path = model_path.with_suffix(model_path.suffix + ".sha256")
            if sha_path.exists():
                expected_sha256 = sha_path.read_text(encoding="utf-8").strip().split()[0]

        if REQUIRE_MODEL_SHA256 and not expected_sha256:
            raise ValueError(
                "Model checksum is required. Set MODEL_SHA256 or include "
                f"{model_path.name}.sha256 next to the model."
            )

        verify_model_sha256(model_path, expected_sha256=expected_sha256)

        with open(model_path, "rb") as f:
            bundle = pickle.load(f)

        self.pipeline = bundle["pipeline"]
        self.version = bundle["version"]
        self.run_id = bundle["run_id"]
        logger.info(f"Model loaded | version: {self.version} | run_id: {self.run_id}")

    def predict(self, text: str) -> dict:
        """Run prediction on a single text. Returns label and confidence."""
        label_idx = self.pipeline.predict([text])[0]
        proba = self.pipeline.predict_proba([text])[0]
        confidence = round(float(proba[label_idx]), 4)
        label = "spam" if label_idx == 1 else "ham"
        return {"label": label, "confidence": confidence}
