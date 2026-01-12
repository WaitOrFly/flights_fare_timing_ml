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


def _safe_start_run(mlflow, run_id: str):
    if not run_id:
        return mlflow.start_run()
    try:
        return mlflow.start_run(run_id=run_id)
    except Exception:
        return mlflow.start_run(run_name=run_id)


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    eta: float,
    max_depth: int,
    min_child_weight: float,
    subsample: float,
    colsample_bytree: float,
    gamma: float,
    reg_lambda: float,
    reg_alpha: float,
    num_boost_round: int,
    early_stopping_rounds: int,
    experiment_name: str,
    run_id: str,
):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        "eta": eta,
        "max_depth": max_depth,
        "min_child_weight": min_child_weight,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "gamma": gamma,
        "lambda": reg_lambda,
        "alpha": reg_alpha,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
    }

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        evals=[(dval, "validation")],
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
    )

    preds = booster.predict(dval)
    metrics: Dict[str, object] = {
        "rmse": float(mean_squared_error(y_val, preds, squared=False)),
        "mae": float(mean_absolute_error(y_val, preds)),
        "r2": float(r2_score(y_val, preds)),
    }

    mlflow = _get_mlflow()
    if mlflow:
        mlflow.set_experiment(experiment_name)
        with _safe_start_run(mlflow, run_id):
            with mlflow.start_run(run_name="Train", nested=True):
                mlflow.autolog()
                mlflow.log_params(params)
                mlflow.log_param("early_stopping_rounds", early_stopping_rounds)
                for key, value in metrics.items():
                    if value is not None:
                        mlflow.log_metric(f"val_{key}", float(value))

    return booster
