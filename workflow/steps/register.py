import io
import json
import os
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import s3fs
import xgboost as xgb
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


def _build_sklearn_model(role: str, featurizer_model):
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

    model_path = "sklearn_model"
    os.makedirs(model_path, exist_ok=True)
    joblib.dump(featurizer_model, os.path.join(model_path, "sklearn_model.joblib"))

    session = Session()
    image_uri = image_uris.retrieve(
        framework="sklearn", region=session.boto_region_name, version="1.2-1"
    )
    requirements_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "requirements_inference.txt")
    )
    print(f"[Register] requirements_inference.txt: {requirements_path}")
    try:
        with open(requirements_path, "r", encoding="utf-8") as req_file:
            print("[Register] requirements_inference.txt contents:")
            print(req_file.read())
    except OSError as exc:
        print(f"[Register] Failed to read requirements_inference.txt: {exc}")
    model_builder = ModelBuilder(
        model_path=model_path,
        name="sklearn_featurizer",
        dependencies={"requirements": requirements_path},
        image_uri=image_uri,
        schema_builder=schema_builder,
        model_server=ModelServer.TORCHSERVE,
        inference_spec=SklearnModelSpec(),
        role_arn=role,
    )
    return model_builder.build()


def _build_xgboost_model(role: str, booster: xgb.Booster):
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

    schema_builder = SchemaBuilder(
        sample_input=np.zeros(8),
        sample_output=np.array([0.0]),
        input_translator=RequestTranslator(),
    )

    model_path = "xgboost_model"
    os.makedirs(model_path, exist_ok=True)
    booster.save_model(os.path.join(model_path, "xgboost_model.bin"))

    session = Session()
    image_uri = image_uris.retrieve(
        framework="xgboost", region=session.boto_region_name, version="1.7-1"
    )
    requirements_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "requirements_inference.txt")
    )
    print(f"[Register] requirements_inference.txt: {requirements_path}")
    try:
        with open(requirements_path, "r", encoding="utf-8") as req_file:
            print("[Register] requirements_inference.txt contents:")
            print(req_file.read())
    except OSError as exc:
        print(f"[Register] Failed to read requirements_inference.txt: {exc}")
    model_builder = ModelBuilder(
        model=booster,
        model_path=model_path,
        dependencies={"requirements": requirements_path},
        schema_builder=schema_builder,
        role_arn=role,
        image_uri=image_uri,
    )
    return model_builder.build()


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
    sklearn_model = _build_sklearn_model(role, featurizer_model)
    xgboost_model = _build_xgboost_model(role, booster)

    eval_file_name = unique_name_from_base("evaluation")
    eval_report_s3_uri = s3_path_join(
        "s3://",
        bucket_name,
        model_package_group_name,
        f"evaluation-report/{eval_file_name}.json",
    )
    s3_fs = s3fs.S3FileSystem()
    with s3_fs.open(eval_report_s3_uri, "wb") as f:
        f.write(json.dumps(model_report_dict).encode("utf-8"))

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
        with mlflow.start_run(run_id=run_id):
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
