import json
import os
import tarfile
from typing import Dict

import joblib
import xgboost as xgb
from sagemaker import image_uris
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.s3 import S3Uploader
from sagemaker.session import Session
from sagemaker.xgboost.model import XGBoostModel


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


def _save_model_artifacts(featurizer_model, booster: xgb.Booster, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    featurizer_path = os.path.join(output_dir, "featurizer.joblib")
    booster_path = os.path.join(output_dir, "xgboost-model.json")
    joblib.dump(featurizer_model, featurizer_path)
    booster.save_model(booster_path)

    tar_path = os.path.join(output_dir, "model.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(featurizer_path, arcname="featurizer.joblib")
        tar.add(booster_path, arcname="xgboost-model.json")
    return tar_path


def register(
    role: str,
    featurizer_model,
    booster: xgb.Booster,
    bucket_name: str,
    model_report_dict: Dict[str, object],
    model_package_group_name: str,
    model_approval_status: str,
    experiment_name: str,
    run_id: str,
):
    session = Session()
    region = session.boto_region_name
    framework_version = "1.7-1"
    image_uri = image_uris.retrieve(framework="xgboost", region=region, version=framework_version)

    work_dir = "/tmp/model_artifacts"
    tar_path = _save_model_artifacts(featurizer_model, booster, work_dir)

    s3_model_prefix = f"s3://{bucket_name}/{model_package_group_name}/model"
    model_data = S3Uploader.upload(tar_path, s3_model_prefix)

    evaluation_path = os.path.join(work_dir, "evaluation.json")
    with open(evaluation_path, "w") as f:
        json.dump(model_report_dict, f, indent=2)
    s3_eval_prefix = f"s3://{bucket_name}/{model_package_group_name}/evaluation"
    evaluation_s3_uri = S3Uploader.upload(evaluation_path, s3_eval_prefix)

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=evaluation_s3_uri,
            content_type="application/json",
        )
    )

    model = XGBoostModel(
        model_data=model_data,
        image_uri=image_uri,
        framework_version=framework_version,
        role=role,
        sagemaker_session=session,
    )

    model_package = model.register(
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
    )

    mlflow = _get_mlflow()
    if mlflow:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_id=run_id):
            with mlflow.start_run(run_name="Register", nested=True):
                mlflow.autolog()
                mlflow.log_params(
                    {
                        "model_package_group_name": model_package_group_name,
                        "model_approval_status": model_approval_status,
                        "model_data": model_data,
                        "evaluation_s3_uri": evaluation_s3_uri,
                    }
                )

    return model_package.model_package_arn
