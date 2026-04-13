"""Project-wide configuration and constants."""

import os
from pathlib import Path

from app.logger import get_logger

logger = get_logger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data")))
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(PROJECT_ROOT / "model")))

INPUT_PATH = Path(os.getenv("INPUT_PATH", str(DATA_DIR / "sms+spam+collection.zip")))
OUTPUT_PATH = Path(os.getenv("OUTPUT_PATH", str(DATA_DIR / "sms_preprocessed.csv")))
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(MODEL_DIR / "model.pkl")))

# Services
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

MODEL_SHA256 = os.getenv("MODEL_SHA256")

# When running behind a reverse proxy/load balancer, configure which proxy IPs are allowed
# to supply X-Forwarded-For / X-Real-IP. If unset, the app will not trust those headers.
TRUST_PROXY_HEADERS = os.getenv("TRUST_PROXY_HEADERS", "0").strip() in {"1", "true", "yes", "on"}
TRUSTED_PROXY_IPS = {
    ip.strip()
    for ip in os.getenv("TRUSTED_PROXY_IPS", "").split(",")
    if ip.strip()
}

# For production safety, you can require a model checksum to be present (via env or sidecar .sha256 file).
REQUIRE_MODEL_SHA256 = os.getenv("REQUIRE_MODEL_SHA256", "0").strip() in {"1", "true", "yes", "on"}

# Words that look like stopwords but are strong spam signals — never remove these
KEEP_WORDS = {
    "free",
    "win",
    "won",
    "not",
    "no",
    "call",
    "txt",
    "prize",
    "claim",
    "urgent",
    "cash",
    "offer",
    "deal",
}

# Minimal stopword list (no NLTK needed)
STOP_WORDS = {
    "the",
    "a",
    "an",
    "is",
    "it",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "and",
    "or",
    "this",
    "that",
    "with",
    "as",
    "was",
    "are",
    "be",
    "have",
    "has",
    "had",
    "but",
    "they",
    "their",
    "you",
    "your",
    "we",
    "our",
    "he",
    "she",
    "his",
    "her",
    "its",
    "will",
    "would",
    "could",
    "should",
    "do",
    "did",
    "does",
    "been",
    "being",
    "from",
    "by",
    "about",
    "up",
    "out",
    "so",
    "if",
    "me",
    "my",
    "us",
    "than",
    "just",
    "i",
    "am",
    "ll",
    "re",
    "ve",
    "dont",
    "im",
} - KEEP_WORDS


TFIDF_PARAMS = {
    "max_features": 15_000,
    "ngram_range": (1, 2),
    "sublinear_tf": True,
    "min_df": 2,
    "analyzer": "word",
    "strip_accents": "unicode",
}

LR_PARAMS = {
    "C": 5.0,
    "max_iter": 1000,
    "class_weight": "balanced",
    "solver": "lbfgs",
    "random_state": 42,
}


def validate_paths() -> None:
    """Create required directories and verify input file exists."""
    DATA_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    if not INPUT_PATH.exists():
        logger.error(f"Input file not found: {INPUT_PATH}")
        logger.error("Place sms+spam+collection.zip inside the data/ folder.")
        raise FileNotFoundError(f"{INPUT_PATH} not found.")
