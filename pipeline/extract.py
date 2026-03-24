"""Extracts SMS data from zip and saves it as a structured CSV."""

import zipfile
from pathlib import Path

import pandas as pd

from app.logger import get_logger

logger = get_logger(__name__)


def extract_to_csv(zip_path: str | Path) -> pd.DataFrame:
    """Read zip, parse SMS data, add label_idx, and save to CSV."""

    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open("SMSSpamCollection") as f:
            df = pd.read_csv(
                f,
                sep="\t",
                header=None,
                names=["label", "text"],
                encoding="utf-8",
            )

    df["label_idx"] = (df["label"] == "spam").astype(int)

    logger.info(f"Label distribution:\n{df['label'].value_counts().to_string()}")

    return df   