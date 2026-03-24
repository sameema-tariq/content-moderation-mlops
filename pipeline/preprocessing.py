"""Text cleaning pipeline for SMS spam classification using stdlib and sklearn only."""

import re
import string
import pandas as pd
from config.settings import STOP_WORDS
from app.logger import get_logger

logger = get_logger(__name__)


class SMSPreprocessor:
    """
    Clean raw SMS text into a form suitable for TF-IDF vectorisation.

    Usage:
        preprocessor = SMSPreprocessor()
        df["text_clean"] = preprocessor.fit_transform(df["text"])
    """

    def clean(self, text: str) -> str:
        """Clean a single SMS message."""

        # 1. Lowercase
        text = text.lower()

        # 2. Replace URLs — presence of URL is a spam signal, keep the token
        text = re.sub(r"http\S+|www\.\S+", " urltoken ", text)

        # 3. Replace long digit sequences (phone numbers)
        text = re.sub(r"\b\d{5,}\b", " phonetoken ", text)

        # 4. Replace currency amounts — strong spam signal
        text = re.sub(r"[£$]\d+|\d+p\b", " pricetoken ", text)

        # 5. Remove punctuation
        text = text.translate(
            str.maketrans(string.punctuation, " " * len(string.punctuation))
        )

        # 6. Remove remaining standalone digits
        text = re.sub(r"\b\d+\b", " ", text)

        # 7. Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # 8. Remove stopwords and single-character tokens
        tokens = [
            t for t in text.split()
            if len(t) > 1 and t not in STOP_WORDS
        ]

        return " ".join(tokens)

    def fit_transform(self, texts: pd.Series) -> pd.Series:
        """Clean an entire pandas Series of SMS messages."""
        logger.info(f"Preprocessing {len(texts):,} messages...")
        cleaned = texts.apply(self.clean)
        empty = cleaned.str.strip().eq("").sum()
        if empty:
            logger.warning(f"{empty} messages became empty after cleaning.")
        logger.info("Preprocessing complete.")
        return cleaned


def load_preprocessed_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function: load raw data and return it fully preprocessed.

    Returns DataFrame with columns:
        label      : str  — 'spam' or 'ham'
        text       : str  — original raw message
        text_clean : str  — cleaned message (feed this to TF-IDF)
        label_idx  : int  — 1=spam, 0=ham
    """
    preprocessor = SMSPreprocessor()
    df["text_clean"] = preprocessor.fit_transform(df["text"])
    return df

