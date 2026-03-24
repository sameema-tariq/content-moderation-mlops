"""Train a TF-IDF + Logistic Regression spam classifier and log results to MLflow."""

import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from app.logger import get_logger
from pipeline.test_evaluate import model_predict, model_evaluate, log_to_mlflow
from pipeline.save_model import model_save

from config.settings import LR_PARAMS, TFIDF_PARAMS

logger = get_logger(__name__)

MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")


def build_pipeline() -> Pipeline:
    """Construct the TF-IDF + Logistic Regression sklearn pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
        ("clf",   LogisticRegression(**LR_PARAMS)),
    ])


def cross_validate(pipeline: Pipeline, X: pd.Series, y: pd.Series) -> dict:
    """Run 5-fold stratified cross-validation and return mean/std F1 scores."""
    logger.info("Running 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1")
    logger.info(f"CV F1: {scores.mean():.4f} (+/- {scores.std():.4f})")
    return {
        "cv_f1_mean": round(scores.mean(), 4),
        "cv_f1_std":  round(scores.std(),  4),
    }


def train(df: pd.DataFrame) -> None:
    """Split data, train pipeline, evaluate, cross-validate, log to MLflow, and save."""

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        df["text_clean"],
        df["label_idx"],
        test_size=0.2,
        stratify=df["label_idx"],
        random_state=42,
    )
    logger.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Build and train
    pipeline = build_pipeline()
    logger.info("Training pipeline: TF-IDF → LogisticRegression...")
    pipeline.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = model_predict(pipeline, X_test)
    metrics = model_evaluate(y_test, y_pred)

    # Cross-validation
    cv_metrics = cross_validate(pipeline, df["text_clean"], df["label_idx"])

    all_metrics = {
        **metrics,
        **cv_metrics,
        "train_size": len(X_train),
        "test_size":  len(X_test),
    }

    # Log to MLflow
    run_id = log_to_mlflow(
        pipeline=pipeline,
        metrics=all_metrics,
        tfidf_params=TFIDF_PARAMS,
        lr_params=LR_PARAMS,
        model_version=MODEL_VERSION,
    )

    # Save model bundle
    model_save(pipeline, metrics=all_metrics, version=MODEL_VERSION, run_id=run_id)
