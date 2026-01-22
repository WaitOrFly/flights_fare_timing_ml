import os
import tempfile
from typing import Dict

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import boto3


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
    base_score: float,
    model_artifacts_s3_uri: str,
    experiment_name: str,
    run_id: str,
):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    if not base_score or base_score <= 0:
        base_score = float(np.mean(y_train))

    params = {
        "eta": eta,
        "max_depth": max_depth,
        "min_child_weight": min_child_weight,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "gamma": gamma,
        "lambda": reg_lambda,
        "alpha": reg_alpha,
        "base_score": base_score,
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

    if model_artifacts_s3_uri:
        _upload_booster_to_s3(booster, model_artifacts_s3_uri)

    preds = booster.predict(dval)
    preds_original = np.expm1(preds)
    y_val_original = np.expm1(y_val)

    sample_count = min(5, len(preds))
    if sample_count > 0:
        print("[Train] sample preds (log):", preds[:sample_count])
        print("[Train] sample y_val (log):", y_val[:sample_count])
        print("[Train] sample preds (orig):", preds_original[:sample_count])
        print("[Train] sample y_val (orig):", y_val_original[:sample_count])
    metrics: Dict[str, object] = {
        "rmse": float(mean_squared_error(y_val_original, preds_original, squared=False)),
        "mae": float(mean_absolute_error(y_val_original, preds_original)),
        "r2": float(r2_score(y_val_original, preds_original)),
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


def _upload_booster_to_s3(booster: xgb.Booster, s3_uri: str) -> None:
    if not s3_uri.startswith("s3://"):
        raise ValueError("model_artifacts_s3_uri must start with 's3://'")

    s3_path = s3_uri.replace("s3://", "")
    bucket = s3_path.split("/")[0]
    prefix = "/".join(s3_path.split("/")[1:]).rstrip("/")
    key = f"{prefix}/artifacts/xgboost_model.bin" if prefix else "artifacts/xgboost_model.bin"

    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "xgboost_model.bin")
        booster.save_model(model_path)
        s3 = boto3.client("s3")
        with open(model_path, "rb") as handle:
            s3.put_object(Bucket=bucket, Key=key, Body=handle)

    print(f"[Train] S3 upload complete: s3://{bucket}/{key}")
