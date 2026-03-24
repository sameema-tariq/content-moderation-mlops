"""Evaluates model predictions and logs metrics and artifacts to MLflow."""

import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from app.logger import get_logger
from config.settings import MLFLOW_URI

logger = get_logger(__name__)


def model_predict(pipeline: Pipeline, X_test) -> list:
    """Run predictions on the test set."""
    return pipeline.predict(X_test)


def model_evaluate(y_test, y_pred) -> dict:
    """Compute and log accuracy, F1, precision, recall, and confusion matrix."""
    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred),  4),
        "f1_spam":   round(f1_score(y_test, y_pred),        4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred),    4),
    }

    logger.info(f"Accuracy: {metrics['accuracy']} | F1: {metrics['f1_spam']} | "
                f"Precision: {metrics['precision']} | Recall: {metrics['recall']}")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['ham', 'spam'])}")

    cm = confusion_matrix(y_test, y_pred)
    logger.info(
        f"Confusion Matrix — "
        f"TN: {cm[0][0]}  FP: {cm[0][1]}  FN: {cm[1][0]}  TP: {cm[1][1]}"
    )

    return metrics


def log_to_mlflow(
    pipeline: Pipeline,
    metrics: dict,
    tfidf_params: dict,
    lr_params: dict,
    model_version: str,
) -> str:
    """Log params, metrics, and model to MLflow. Returns the run ID."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("spam-detection")

    with mlflow.start_run(run_name=f"tfidf-logreg-{model_version}") as run:
        mlflow.log_params({
            **{f"tfidf_{k}": str(v) for k, v in tfidf_params.items()},
            **{f"lr_{k}":    str(v) for k, v in lr_params.items()},
            "model_version": model_version,
        })
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        run_id = run.info.run_id
        logger.info(f"MLflow run logged | run_id: {run_id}")
        return run_id
