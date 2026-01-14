import os
import json
import mlflow
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

import boto3

def _to_dmatrix(X):
    import pandas as pd
    import numpy as np
    import xgboost as xgb

    if isinstance(X, pd.DataFrame):
        X = X.select_dtypes(include=["number"])
        X = X.fillna(0.0)
        X = X.values

    if hasattr(X, "toarray"):
        X = X.toarray()

    X = np.asarray(X, dtype=np.float32)
    return xgb.DMatrix(X)


def test(
    featurizer_s3_uri,
    model_s3_uri,
    X_test,
    y_test,             # log(price)
    y_test_original,    # INR 기준 정답
    output_metrics_s3_uri,
    experiment_name="main_experiment",
    run_id="run-01"
):
    # ===== MLflow 설정 =====
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_ARN"])
    mlflow.set_experiment(experiment_name)

    metrics = {}

    with mlflow.start_run(run_id=run_id):
        with mlflow.start_run(run_name="Test", nested=True):
            mlflow.autolog()

            print(f"Test features shape: {X_test.shape}")
            print(f"Test labels shape (log) : {y_test.shape}")
            print(f"Test labels shape (INR) : {y_test_original.shape}")

            # ===== Predict (log) =====
            # dtest = _to_dmatrix(X_test)
            # y_pred_log = booster.predict(dtest)

            # ===== Load model from S3 =====
            s3 = boto3.client("s3")

            local_model_path = "/tmp/xgboost_model.bin"

            s3_path = model_s3_uri.replace("s3://", "")
            bucket = s3_path.split("/")[0]
            key = "/".join(s3_path.split("/")[1:])

            s3.download_file(bucket, key, local_model_path)

            booster = xgb.Booster()
            booster.load_model(local_model_path)

            # ===== Predict (log) =====
            dtest = _to_dmatrix(X_test)
            y_pred_log = booster.predict(dtest)

            # ===== Restore to INR =====
            y_pred_price = np.expm1(y_pred_log)

            # ===== Metrics (INR 기준) =====
            y_test_original_np = np.asarray(y_test_original, dtype=np.float32)

            mae = mean_absolute_error(y_test_original_np, y_pred_price)
            rmse = mean_squared_error(
                y_test_original_np, y_pred_price, squared=False
            )

            mlflow.log_metric("test_mae_inr", mae)
            mlflow.log_metric("test_rmse_inr", rmse)

            print(f"[Test] MAE  (INR): ₹{mae:,.2f}")
            print(f"[Test] RMSE (INR): ₹{rmse:,.2f}")

            metrics = {
                "test_mae_inr": float(mae),
                "test_rmse_inr": float(rmse),
            }

            # 🔥 S3에 저장
            s3 = boto3.client("s3")
            bucket = output_metrics_s3_uri.replace("s3://", "").split("/")[0]
            key = "/".join(output_metrics_s3_uri.replace("s3://", "").split("/")[1:])

            s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=json.dumps(metrics),
                ContentType="application/json",
            )
    return output_metrics_s3_uri
