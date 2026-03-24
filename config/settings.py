"""Project-wide configuration and constants."""

from pathlib import Path

from app.logger import get_logger

logger = get_logger(__name__)

# Paths
BASE_DIR    = Path("data")
INPUT_PATH  = BASE_DIR / "sms+spam+collection.zip"
OUTPUT_PATH = BASE_DIR / "sms_preprocessed.csv"
MODEL_PATH  = Path("model/model.pkl")

# Services
MLFLOW_URI = "http://mlflow:5000"

# Words that look like stopwords but are strong spam signals — never remove these
KEEP_WORDS = {
    "free", "win", "won", "not", "no", "call", "txt",
    "prize", "claim", "urgent", "cash", "offer", "deal"
}

# Minimal stopword list (no NLTK needed)
STOP_WORDS = {
    "the", "a", "an", "is", "it", "in", "on", "at", "to", "for",
    "of", "and", "or", "this", "that", "with", "as", "was", "are",
    "be", "have", "has", "had", "but", "they", "their", "you", "your",
    "we", "our", "he", "she", "his", "her", "its", "will", "would",
    "could", "should", "do", "did", "does", "been", "being", "from",
    "by", "about", "up", "out", "so", "if", "me", "my", "us",
    "than", "just", "i", "am", "ll", "re", "ve", "dont", "im"
} - KEEP_WORDS


TFIDF_PARAMS = {
    "max_features":  15_000,
    "ngram_range":   (1, 2),
    "sublinear_tf":  True,
    "min_df":        2,
    "analyzer":      "word",
    "strip_accents": "unicode",
}

LR_PARAMS = {
    "C":            5.0,
    "max_iter":     1000,
    "class_weight": "balanced",
    "solver":       "lbfgs",
    "random_state": 42,
}


def validate_paths() -> None:
    """Create required directories and verify input file exists."""
    BASE_DIR.mkdir(exist_ok=True)
    Path("model").mkdir(exist_ok=True)

    if not INPUT_PATH.exists():
        logger.error(f"Input file not found: {INPUT_PATH}")
        logger.error("Place sms+spam+collection.zip inside the data/ folder.")
        raise FileNotFoundError(f"{INPUT_PATH} not found.")
