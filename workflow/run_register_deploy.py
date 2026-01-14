import argparse
import json
import os
import tempfile

import boto3
import joblib

try:
    from sagemaker.session import Session
except Exception:
    Session = None
try:
    from sagemaker import get_execution_role
except Exception:
    get_execution_role = None

from steps.register import register
from steps.deploy import deploy


def _get_mlflow_server_arn():
    r = boto3.client("sagemaker").list_mlflow_tracking_servers()[
        "TrackingServerSummaries"
    ]
    if len(r) < 1:
        raise RuntimeError("No running MLflow tracking server found.")
    return r[0]["TrackingServerArn"]


def _ensure_xgboost():
    try:
        import xgboost  # type: ignore
        return xgboost
    except ModuleNotFoundError:
        import subprocess
        import sys

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "xgboost"]
        )
        import xgboost  # type: ignore
        return xgboost


def _parse_s3_uri(uri: str):
    s3_path = uri.replace("s3://", "")
    bucket = s3_path.split("/")[0]
    key = "/".join(s3_path.split("/")[1:])
    return bucket, key


def main():
    parser = argparse.ArgumentParser(
        description="Run register + deploy using latest S3 artifacts."
    )
    parser.add_argument(
        "--project-prefix",
        default="flight-fares-timing",
        help="S3 prefix used by pipeline outputs.",
    )
    parser.add_argument(
        "--bucket-name",
        default=None,
        help="S3 bucket (default: SageMaker default bucket).",
    )
    parser.add_argument(
        "--model-package-group-name",
        default="flight-fares-timing-model-package-group",
        help="Model package group name.",
    )
    parser.add_argument(
        "--model-approval-status",
        default="PendingManualApproval",
        help="Model approval status.",
    )
    parser.add_argument(
        "--experiment-name",
        default="flight-fares-timing-pipeline",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="MLflow run_id from last successful pipeline run.",
    )
    parser.add_argument(
        "--deploy-model",
        action="store_true",
        default=True,
        help="Deploy model after register.",
    )
    parser.add_argument(
        "--sklearn-image-uri",
        default=None,
        help="Custom SKLearn inference image URI for the featurizer container.",
    )
    args = parser.parse_args()

    if Session is None and not args.bucket_name:
        raise RuntimeError(
            "sagemaker.session.Session is unavailable; "
            "pass --bucket-name explicitly."
        )

    session = Session() if Session is not None else None
    bucket_name = args.bucket_name or session.default_bucket()

    if "MLFLOW_TRACKING_ARN" not in os.environ:
        os.environ["MLFLOW_TRACKING_ARN"] = _get_mlflow_server_arn()
    if args.sklearn_image_uri and "SKLEARN_IMAGE_URI" not in os.environ:
        os.environ["SKLEARN_IMAGE_URI"] = args.sklearn_image_uri
    if "REGISTER_OUTPUT_DIR" not in os.environ:
        os.environ["REGISTER_OUTPUT_DIR"] = os.path.join(
            os.getcwd(), "register_output"
        )

    # Build S3 paths based on pipeline outputs.
    model_s3_uri = (
        f"s3://{bucket_name}/{args.project_prefix}/model/xgboost_model.bin"
    )
    featurizer_s3_uri = (
        f"s3://{bucket_name}/{args.project_prefix}/processed/featurizer/featurizer.joblib"
    )
    metrics_s3_uri = (
        f"s3://{bucket_name}/{args.project_prefix}/metrics/test_metrics.json"
    )

    s3 = boto3.client("s3")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Download artifacts.
        bucket, key = _parse_s3_uri(model_s3_uri)
        local_model_path = os.path.join(tmpdir, "xgboost_model.bin")
        s3.download_file(bucket, key, local_model_path)

        bucket, key = _parse_s3_uri(featurizer_s3_uri)
        local_featurizer_path = os.path.join(tmpdir, "featurizer.joblib")
        s3.download_file(bucket, key, local_featurizer_path)

        # Load artifacts.
        featurizer_model = joblib.load(local_featurizer_path)

        xgboost = _ensure_xgboost()
        booster = xgboost.Booster()
        booster.load_model(local_model_path)

        # Load test metrics.
        bucket, key = _parse_s3_uri(metrics_s3_uri)
        metrics_obj = s3.get_object(Bucket=bucket, Key=key)
        model_report_dict = json.loads(
            metrics_obj["Body"].read().decode("utf-8")
        )

    if get_execution_role is not None:
        role = get_execution_role()
    else:
        role = os.environ.get("SAGEMAKER_ROLE_ARN") or os.environ.get("SM_ROLE_ARN")
        if not role and session is not None:
            role = session.get_execution_role()
        if not role:
            raise RuntimeError(
                "Execution role is unavailable. Set SAGEMAKER_ROLE_ARN or SM_ROLE_ARN."
            )

    # If run_id is missing, create a fresh MLflow run so register/deploy can attach.
    run_id = args.run_id
    if not run_id:
        import mlflow

        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_ARN"])
        mlflow.set_experiment(args.experiment_name)
        with mlflow.start_run(run_name="RegisterDeployBootstrap") as run:
            run_id = run.info.run_id

    model_package_arn = register(
        role=role,
        featurizer_model=featurizer_model,
        booster=booster,
        bucket_name=bucket_name,
        model_report_dict=model_report_dict,
        model_package_group_name=args.model_package_group_name,
        model_approval_status=args.model_approval_status,
        experiment_name=args.experiment_name,
        run_id=run_id,
    )

    sm_client = boto3.client("sagemaker")
    desc = sm_client.describe_model_package(
        ModelPackageName=model_package_arn
    )
    model_data_s3_uri = desc["InferenceSpecification"]["Containers"][0][
        "ModelDataUrl"
    ]

    deploy(
        role=role,
        project_prefix=args.project_prefix,
        model_data_s3_uri=model_data_s3_uri,
        model_package_arn=model_package_arn,
        deploy_model=args.deploy_model,
        experiment_name=args.experiment_name,
        run_id=run_id,
    )


if __name__ == "__main__":
    main()
