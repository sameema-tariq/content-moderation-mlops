"""Utility functions for previewing and saving preprocessed SMS data."""

import pandas as pd

from app.logger import get_logger
from config.settings import OUTPUT_PATH

logger = get_logger(__name__)


def summarize_sms_info(df: pd.DataFrame) -> None:
    """Log before/after cleaning samples and word count stats for spam and ham."""
    logger.info("\n=== Before vs After (3 spam + 3 ham) ===\n")
    for label in ["spam", "ham"]:
        samples = df[df["label"] == label].head(3)
        for _, row in samples.iterrows():
            logger.info(f"[{row['label'].upper()}]")
            logger.info(f"  Original : {row['text'][:90]}")
            logger.info(f"  Cleaned  : {row['text_clean'][:90]}")

    df["word_count_before"] = df["text"].str.split().str.len()
    df["word_count_after"] = df["text_clean"].str.split().str.len()

    logger.info("=== Word Count Stats ===")
    for label in ["spam", "ham"]:
        subset = df[df["label"] == label]
        logger.info(
            f"{label.upper()} — before: {subset['word_count_before'].mean():.1f} words"
            f" | after: {subset['word_count_after'].mean():.1f} words"
        )


def save_preprocessed_csv(df: pd.DataFrame) -> None:
    """Save label_idx and text_clean columns to the configured output CSV path."""
    df[["label_idx", "text_clean"]].to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Saved preprocessed data to {OUTPUT_PATH}")
    logger.info(f"Shape: {df.shape} | Columns: {list(df[['label_idx', 'text_clean']].columns)}")
