import os
from typing import Dict

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _get_mlflow():
    try:
        import mlflow  # type: ignore
    except Exception:
        return None

    tracking_arn = os.environ.get("MLFLOW_TRACKING_ARN")
    if not tracking_arn:
        return None

    mlflow.set_tracking_uri(tracking_arn)
    return mlflow


def test(
    featurizer_model,
    booster: xgb.Booster,
    X_test: np.ndarray,
    y_test: np.ndarray,
    experiment_name: str,
    run_id: str,
):
    dtest = xgb.DMatrix(X_test)
    preds = booster.predict(dtest)

    metrics: Dict[str, object] = {
        "rmse": float(mean_squared_error(y_test, preds, squared=False)),
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
    }

    mlflow = _get_mlflow()
    if mlflow:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_id=run_id):
            with mlflow.start_run(run_name="Test", nested=True):
                mlflow.autolog()
                for key, value in metrics.items():
                    if value is not None:
                        mlflow.log_metric(f"test_{key}", float(value))

    return {"metrics": metrics}
