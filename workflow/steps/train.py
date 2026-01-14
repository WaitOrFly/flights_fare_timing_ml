# steps/train.py
import os
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import boto3

def _to_dmatrix(X, y=None):
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

    if y is not None:
        y = np.asarray(y, dtype=np.float32)
        return xgb.DMatrix(X, label=y)

    return xgb.DMatrix(X)


def train(
    X_train,
    y_train,           # log(price)
    X_val,
    y_val,             # log(price)
    y_val_original,    # INR 기준 정답
    output_model_s3_uri,
    eta=0.1,
    max_depth=6,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    num_boost_round=300,
    experiment_name="main_experiment",
    run_id="run-01",
):
    import mlflow

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_ARN"])
    mlflow.set_experiment(experiment_name)

    model = None
    metrics = {}

    with mlflow.start_run(run_id=run_id):
        with mlflow.start_run(run_name="Train", nested=True):
            mlflow.autolog()

             # ===== DMatrix (log target) =====
            dtrain = _to_dmatrix(X_train, y_train)
            dval   = _to_dmatrix(X_val, y_val)

            xgb_params = {
                "eta": eta,
                "max_depth": max_depth,
                "min_child_weight": min_child_weight,
                "subsample": subsample,
                "colsample_bytree": colsample_bytree,
                "objective": "reg:squarederror",
                "eval_metric": "mae",   # log-space metric (early stopping용)
                "tree_method": "hist",
            }

            trained_model = xgb.train(
                params=xgb_params,
                dtrain=dtrain,
                num_boost_round=num_boost_round,
                evals=[(dval, "validation")],
                verbose_eval=False,
            )

            # ===== Prediction (log → INR 복원) =====
            val_preds_log = trained_model.predict(dval)
            val_preds_price = np.expm1(val_preds_log)

             # ===== Metrics (INR 기준) =====
            mae = mean_absolute_error(y_val_original, val_preds_price)
            rmse = mean_squared_error(
                y_val_original,
                val_preds_price,
                squared=False
            )

            mlflow.log_metric("mae_inr", mae)
            mlflow.log_metric("rmse_inr", rmse)

            print(f"MAE  (INR): ₹{mae:,.2f}")
            print(f"RMSE (INR): ₹{rmse:,.2f}")

            mlflow.xgboost.log_model(
                trained_model,
                artifact_path="model"
            )

            # model = trained_model
            metrics = {
                "mae": mae,
                "rmse": rmse
            }

            # # SageMaker model artifact
            # model_file_path = "/opt/ml/model/xgboost_model.bin"
            # os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
            # model.save_model(model_file_path)
            # ===== Save model locally =====
            local_model_path = "/tmp/xgboost_model.bin"
            trained_model.save_model(local_model_path)

            # ===== Upload to S3 =====
            s3 = boto3.client("s3")
            model_s3_uri = output_model_s3_uri
            # s3://bucket/prefix/model/xgboost_model.bin
            s3_path = model_s3_uri.replace("s3://", "")
            bucket = s3_path.split("/")[0]
            key = "/".join(s3_path.split("/")[1:])

            s3.upload_file(local_model_path, bucket, key)

            model_s3_uri = f"s3://{bucket}/{key}"


    return model_s3_uri, metrics