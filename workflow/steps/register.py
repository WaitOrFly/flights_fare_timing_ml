import argparse
import ast
import io
import json
import os
import importlib.metadata
import subprocess
import sys
import tempfile
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd


def _ensure_packages(requirements) -> None:
    command = [sys.executable, "-m", "pip", "install"]
    subprocess.check_call([*command, *requirements])


def _is_installed(package_name: str) -> bool:
    try:
        importlib.metadata.version(package_name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def _ensure_runtime_deps() -> None:
    if not _is_installed("sagemaker"):
        _ensure_packages(["sagemaker==2.219.0"])
    _ensure_packages(
        [
            "boto3==1.28.57",
            "botocore==1.31.85",
            "s3transfer==0.7.0",
            "scikit-learn==1.2.1",
        ]
    )
    if not _is_installed("xgboost"):
        _ensure_packages(["xgboost==1.7.6"])


_ensure_runtime_deps()

import boto3
import xgboost as xgb
import sklearn  # noqa: F401
from sagemaker import image_uris
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.pipeline import PipelineModel
from sagemaker.s3_utils import s3_path_join
from sagemaker.serve import CustomPayloadTranslator, InferenceSpec, ModelServer
from sagemaker.serve.builder.model_builder import ModelBuilder
from sagemaker.serve.builder.schema_builder import SchemaBuilder
from sagemaker.session import Session
from sagemaker.utils import unique_name_from_base


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


def _build_sklearn_model(
    role: str, featurizer_model, model_dir: str, requirements_path: str
):
    feature_names = getattr(featurizer_model, "feature_names_in_", None)
    if feature_names is None:
        feature_names = [
            "purchase_day_of_week",
            "purchase_time_bucket",
            "days_until_departure_bucket",
            "is_weekend_departure",
            "is_holiday_season",
            "price_trend_7d",
            "current_vs_historical_avg",
            "route_hash",
            "stops_count",
            "flight_duration_bucket",
        ]

    class SklearnRequestTranslator(CustomPayloadTranslator):
        def serialize_payload_to_bytes(self, payload: object) -> bytes:
            return payload.encode("utf-8")

        def deserialize_payload_from_stream(self, stream) -> pd.DataFrame:
            df = pd.read_csv(io.BytesIO(stream.read()), header=None)
            df.columns = feature_names
            return df

    class SklearnModelSpec(InferenceSpec):
        def invoke(self, input_object: object, model: object):
            return model.transform(input_object)

        def load(self, model_dir: str):
            model_path = os.path.join(model_dir, "sklearn_model.joblib")
            return joblib.load(model_path)

    schema_builder = SchemaBuilder(
        sample_input=",".join(feature_names),
        sample_output=np.zeros(8),
        input_translator=SklearnRequestTranslator(),
    )

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(featurizer_model, os.path.join(model_dir, "sklearn_model.joblib"))

    session = Session()
    image_uri = image_uris.retrieve(
        framework="sklearn", region=session.boto_region_name, version="1.2-1"
    )
    print(f"[Register] requirements_inference.txt: {requirements_path}")
    try:
        with open(requirements_path, "r", encoding="utf-8") as req_file:
            print("[Register] requirements_inference.txt contents:")
            print(req_file.read())
    except OSError as exc:
        print(f"[Register] Failed to read requirements_inference.txt: {exc}")
    model_builder = ModelBuilder(
        model_path=model_dir,
        name="sklearn_featurizer",
        dependencies={"requirements": requirements_path},
        image_uri=image_uri,
        schema_builder=schema_builder,
        model_server=ModelServer.TORCHSERVE,
        inference_spec=SklearnModelSpec(),
        role_arn=role,
    )
    return model_builder.build()


def _build_xgboost_model(
    role: str, booster: xgb.Booster, model_dir: str, requirements_path: str
):
    class RequestTranslator(CustomPayloadTranslator):
        def serialize_payload_to_bytes(self, payload: object) -> bytes:
            return self._convert_numpy_to_bytes(payload)

        def deserialize_payload_from_stream(self, stream) -> xgb.DMatrix:
            np_array = np.load(io.BytesIO(stream.read())).reshape((1, -1))
            return xgb.DMatrix(np_array)

        def _convert_numpy_to_bytes(self, np_array: np.ndarray) -> bytes:
            buffer = io.BytesIO()
            np.save(buffer, np_array)
            return buffer.getvalue()

    class XgbModelSpec(InferenceSpec):
        def invoke(self, input_object: object, model: object):
            return model.predict(input_object)

        def load(self, model_dir: str):
            model_path = os.path.join(model_dir, "xgboost_model.bin")
            xgb_model = xgb.Booster()
            xgb_model.load_model(model_path)
            return xgb_model

    schema_builder = SchemaBuilder(
        sample_input=np.zeros(8),
        sample_output=np.array([0.0]),
        input_translator=RequestTranslator(),
    )

    os.makedirs(model_dir, exist_ok=True)
    booster.save_model(os.path.join(model_dir, "xgboost_model.bin"))

    session = Session()
    image_uri = image_uris.retrieve(
        framework="xgboost", region=session.boto_region_name, version="1.7-1"
    )
    print(f"[Register] requirements_inference.txt: {requirements_path}")
    try:
        with open(requirements_path, "r", encoding="utf-8") as req_file:
            print("[Register] requirements_inference.txt contents:")
            print(req_file.read())
    except OSError as exc:
        print(f"[Register] Failed to read requirements_inference.txt: {exc}")
    model_builder = ModelBuilder(
        model_path=model_dir,
        dependencies={"requirements": requirements_path},
        schema_builder=schema_builder,
        role_arn=role,
        image_uri=image_uri,
        model_server=ModelServer.TORCHSERVE,
        inference_spec=XgbModelSpec(),
    )
    return model_builder.build()


def _parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"S3 uri must start with 's3://': {s3_uri}")
    s3_path = s3_uri[5:]
    bucket, _, key = s3_path.partition("/")
    return bucket, key


def _resolve_bucket_and_prefix(bucket_name: str) -> Tuple[str, str]:
    if bucket_name.startswith("s3://"):
        bucket, prefix = _parse_s3_uri(bucket_name)
        return bucket, prefix.rstrip("/")
    return bucket_name, ""


def _download_if_s3(path_or_uri: str, local_dir: str) -> str:
    if not path_or_uri.startswith("s3://"):
        return path_or_uri

    bucket, key = _parse_s3_uri(path_or_uri)
    if not key:
        raise ValueError(f"S3 uri must include an object key: {path_or_uri}")

    local_path = os.path.join(local_dir, os.path.basename(key))
    boto3.client("s3").download_file(bucket, key, local_path)
    return local_path


def _load_model_report(
    report_json: Optional[str], report_path: Optional[str]
) -> Dict[str, object]:
    if report_json:
        try:
            return json.loads(report_json)
        except json.JSONDecodeError:
            return ast.literal_eval(report_json)
    if report_path:
        if report_path.startswith("s3://"):
            with tempfile.TemporaryDirectory() as temp_dir:
                local_path = _download_if_s3(report_path, temp_dir)
                with open(local_path, "r", encoding="utf-8") as report_file:
                    return json.load(report_file)
        with open(report_path, "r", encoding="utf-8") as report_file:
            return json.load(report_file)
    raise ValueError("model_report_json or model_report_path must be provided.")


def _load_featurizer(path_or_uri: str):
    with tempfile.TemporaryDirectory() as temp_dir:
        local_path = _download_if_s3(path_or_uri, temp_dir)
        return joblib.load(local_path)


def _load_booster(path_or_uri: str) -> xgb.Booster:
    booster = xgb.Booster()
    with tempfile.TemporaryDirectory() as temp_dir:
        local_path = _download_if_s3(path_or_uri, temp_dir)
        booster.load_model(local_path)
    return booster


def _resolve_requirements_path(custom_path: Optional[str]) -> str:
    if custom_path:
        return custom_path
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "requirements_inference.txt"))


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
    requirements_path: Optional[str] = None,
):
    session = Session()
    sklearn_model_dir = tempfile.mkdtemp()
    xgboost_model_dir = tempfile.mkdtemp()
    resolved_requirements_path = _resolve_requirements_path(requirements_path)
    sklearn_model = _build_sklearn_model(
        role, featurizer_model, sklearn_model_dir, resolved_requirements_path
    )
    xgboost_model = _build_xgboost_model(
        role, booster, xgboost_model_dir, resolved_requirements_path
    )

    bucket, bucket_prefix = _resolve_bucket_and_prefix(bucket_name)
    eval_file_name = unique_name_from_base("evaluation")
    eval_prefix_parts = [bucket_prefix, model_package_group_name, "evaluation-report"]
    eval_prefix = "/".join(part for part in eval_prefix_parts if part)
    eval_report_s3_uri = s3_path_join(
        "s3://", bucket, eval_prefix, f"{eval_file_name}.json"
    )
    eval_bucket, eval_key = _parse_s3_uri(eval_report_s3_uri)
    session.boto_session.client("s3").put_object(
        Bucket=eval_bucket,
        Key=eval_key,
        Body=json.dumps(model_report_dict).encode("utf-8"),
    )

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=eval_report_s3_uri,
            content_type="application/json",
        )
    )

    pipeline_model_name = unique_name_from_base("flight-fare-pipeline-model")
    pipeline_model = PipelineModel(
        name=pipeline_model_name,
        sagemaker_session=xgboost_model.sagemaker_session,
        role=role,
        models=[sklearn_model, xgboost_model],
    )

    pipeline_model.register(
        content_types=["text/csv"],
        response_types=["application/x-npy"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )

    mlflow = _get_mlflow()
    if mlflow:
        mlflow.set_experiment(experiment_name)
        with _safe_start_run(mlflow, run_id):
            with mlflow.start_run(run_name="Register", nested=True):
                mlflow.autolog()
                mlflow.log_params(
                    {
                        "model_package_group_name": model_package_group_name,
                        "model_approval_status": model_approval_status,
                        "evaluation_s3_uri": eval_report_s3_uri,
                    }
                )

    sagemaker_client = session.boto_session.client("sagemaker")
    response = sagemaker_client.list_model_packages(
        MaxResults=100,
        ModelPackageGroupName=model_package_group_name,
        ModelPackageType="Versioned",
        SortBy="CreationTime",
        SortOrder="Descending",
    )
    model_package_arn = response["ModelPackageSummaryList"][0]["ModelPackageArn"]
    print(f"[Register] model_package_arn: {model_package_arn}")
    return model_package_arn


def main() -> None:
    parser = argparse.ArgumentParser(description="Register models using SageMaker.")
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--featurizer-model-path", required=True)
    parser.add_argument("--xgboost-model-path", required=True)
    parser.add_argument("--model-report-json")
    parser.add_argument("--model-report-path")
    parser.add_argument("--bucket-name", required=True)
    parser.add_argument("--model-package-group-name", required=True)
    parser.add_argument("--model-approval-status", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-model-package-path", required=True)
    parser.add_argument("--requirements-path")
    args = parser.parse_args()

    model_report_dict = _load_model_report(args.model_report_json, args.model_report_path)
    featurizer_model = _load_featurizer(args.featurizer_model_path)
    booster = _load_booster(args.xgboost_model_path)

    model_package_arn = register(
        role=args.role_arn,
        featurizer_model=featurizer_model,
        booster=booster,
        bucket_name=args.bucket_name,
        model_report_dict=model_report_dict,
        model_package_group_name=args.model_package_group_name,
        model_approval_status=args.model_approval_status,
        experiment_name=args.experiment_name,
        run_id=args.run_id,
        requirements_path=args.requirements_path,
    )

    os.makedirs(os.path.dirname(args.output_model_package_path), exist_ok=True)
    with open(args.output_model_package_path, "w", encoding="utf-8") as output_file:
        json.dump({"model_package_arn": model_package_arn}, output_file)


if __name__ == "__main__":
    main()
