import io
import json
import os
import tarfile
import tempfile
import shutil

import boto3
import joblib
import numpy as np
import pandas as pd

try:
    import mlflow
except ModuleNotFoundError:  # Optional for processing container
    class _DummyRun:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        @property
        def info(self):
            class _Info:
                run_id = "dummy-run"
            return _Info()

    class _DummyMlflow:
        def set_tracking_uri(self, *_args, **_kwargs):
            pass

        def set_experiment(self, *_args, **_kwargs):
            pass

        def start_run(self, *args, **_kwargs):
            return _DummyRun()

        def autolog(self, *_args, **_kwargs):
            pass

        def log_metric(self, *_args, **_kwargs):
            pass

        def log_param(self, *_args, **_kwargs):
            pass

        def set_tag(self, *_args, **_kwargs):
            pass

    mlflow = _DummyMlflow()

ModelMetrics = None
MetricsSource = None
PipelineModel = None
SKLearnModel = None
XGBoostModel = None
s3_path_join = None
unique_name_from_base = None
Session = None
_HAS_MODEL_METRICS = False


def _load_sagemaker():
    global ModelMetrics, MetricsSource, PipelineModel, SKLearnModel
    global XGBoostModel, s3_path_join, unique_name_from_base, Session
    global _HAS_MODEL_METRICS

    try:
        import sagemaker  # noqa: F401
    except ModuleNotFoundError:
        import subprocess
        import sys

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "sagemaker"]
        )

    try:
        from sagemaker import ModelMetrics, MetricsSource
        _HAS_MODEL_METRICS = True
    except Exception:
        try:
            from sagemaker.model_metrics import ModelMetrics, MetricsSource
            _HAS_MODEL_METRICS = True
        except Exception:
            ModelMetrics = None
            MetricsSource = None
            _HAS_MODEL_METRICS = False

    from sagemaker.pipeline import PipelineModel
    from sagemaker.s3_utils import s3_path_join
    from sagemaker.sklearn.model import SKLearnModel
    from sagemaker.utils import unique_name_from_base
    from sagemaker.xgboost.model import XGBoostModel
    try:
        from sagemaker.session import Session
    except Exception:
        Session = None

    globals().update(
        {
            "ModelMetrics": ModelMetrics,
            "MetricsSource": MetricsSource,
            "PipelineModel": PipelineModel,
            "s3_path_join": s3_path_join,
            "SKLearnModel": SKLearnModel,
            "unique_name_from_base": unique_name_from_base,
            "XGBoostModel": XGBoostModel,
            "Session": Session,
        }
    )


def _find_feature_engineer_path():
    candidates = [
        os.path.join(os.path.dirname(__file__), "feature_engineer.py"),
        os.path.join(os.getcwd(), "workflow", "steps", "feature_engineer.py"),
        os.path.join(os.getcwd(), "steps", "feature_engineer.py"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "feature_engineer.py not found; cannot package featurizer."
    )


def _write_feature_engineer_module(model_dir):
    steps_dir = os.path.join(model_dir, "steps")
    os.makedirs(steps_dir, exist_ok=True)

    init_path = os.path.join(steps_dir, "__init__.py")
    if not os.path.exists(init_path):
        with open(init_path, "w", encoding="utf-8") as f:
            f.write("")

    source_path = _find_feature_engineer_path()
    with open(source_path, "r", encoding="utf-8") as f:
        content = f.read()

    target_path = os.path.join(steps_dir, "feature_engineer.py")
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(content)


def _create_featurizer_tar(featurizer_model, output_path, route_stats_path=None):
    with tempfile.TemporaryDirectory() as model_dir:
        model_path = os.path.join(model_dir, "featurizer.joblib")
        joblib.dump(featurizer_model, model_path)
        _write_feature_engineer_module(model_dir)
        if route_stats_path and os.path.exists(route_stats_path):
            stats_path = os.path.join(model_dir, "route_stats.csv")
            shutil.copy2(route_stats_path, stats_path)

        with tarfile.open(output_path, "w:gz") as tar:
            tar.add(model_path, arcname="featurizer.joblib")
            if route_stats_path and os.path.exists(route_stats_path):
                tar.add(stats_path, arcname="route_stats.csv")
            tar.add(
                os.path.join(model_dir, "steps"),
                arcname="steps",
            )


def _create_xgboost_tar(booster, output_path):
    with tempfile.TemporaryDirectory() as model_dir:
        model_path = os.path.join(model_dir, "xgboost-model")
        booster.save_model(model_path)
        with tarfile.open(output_path, "w:gz") as tar:
            tar.add(model_path, arcname="xgboost-model")


def _create_featurizer_tar_from_path(
    featurizer_path,
    output_path,
    route_stats_path=None,
):
    with tempfile.TemporaryDirectory() as model_dir:
        model_path = os.path.join(model_dir, "featurizer.joblib")
        shutil.copy2(featurizer_path, model_path)
        _write_feature_engineer_module(model_dir)
        if route_stats_path and os.path.exists(route_stats_path):
            stats_path = os.path.join(model_dir, "route_stats.csv")
            shutil.copy2(route_stats_path, stats_path)

        with tarfile.open(output_path, "w:gz") as tar:
            tar.add(model_path, arcname="featurizer.joblib")
            if route_stats_path and os.path.exists(route_stats_path):
                tar.add(stats_path, arcname="route_stats.csv")
            tar.add(
                os.path.join(model_dir, "steps"),
                arcname="steps",
            )


def _create_xgboost_tar_from_path(model_path, output_path):
    with tempfile.TemporaryDirectory() as model_dir:
        target_path = os.path.join(model_dir, "xgboost-model")
        shutil.copy2(model_path, target_path)
        with tarfile.open(output_path, "w:gz") as tar:
            tar.add(target_path, arcname="xgboost-model")


def _write_featurizer_inference(source_dir):
    os.makedirs(source_dir, exist_ok=True)
    inference_path = os.path.join(source_dir, "inference.py")

    inference_code = """import io
import os
import sys

import joblib
import numpy as np
import pandas as pd


def model_fn(model_dir):
    sys.path.append(model_dir)
    global FlightFeatureEngineer
    from steps.feature_engineer import FlightFeatureEngineer as _FlightFeatureEngineer
    FlightFeatureEngineer = _FlightFeatureEngineer
    model_path = os.path.join(model_dir, "featurizer.joblib")
    model = joblib.load(model_path)
    stats_path = os.path.join(model_dir, "route_stats.csv")
    route_stats = None
    if os.path.exists(stats_path):
        try:
            route_stats = pd.read_csv(stats_path)
        except Exception:
            route_stats = None
    return {"model": model, "route_stats": route_stats}


def input_fn(request_body, request_content_type):
    if isinstance(request_body, (bytes, bytearray)):
        request_body = request_body.decode("utf-8")
    if request_content_type == "text/csv":
        return pd.read_csv(io.StringIO(request_body))
    return pd.read_json(io.StringIO(request_body))


def predict_fn(input_data, model):
    model_obj = model["model"] if isinstance(model, dict) else model
    route_stats = model.get("route_stats") if isinstance(model, dict) else None
    engineer = FlightFeatureEngineer(apply_log_to_target=False)
    if "Fare" not in input_data.columns:
        input_data = input_data.copy()
        input_data["Fare"] = 0.0
    engineered = engineer.transform(input_data, historical_df=route_stats)
    if "price" in engineered.columns:
        engineered = engineered.drop(columns=["price", "price_original"], errors="ignore")
    feats = model_obj.transform(engineered)
    return np.asarray(feats, dtype=np.float32)


def output_fn(prediction, response_content_type):
    buffer = io.StringIO()
    np.savetxt(buffer, prediction, delimiter=",")
    return buffer.getvalue(), "text/csv"
"""

    with open(inference_path, "w", encoding="utf-8") as f:
        f.write(inference_code)


def to_json_safe(obj):
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def register(
    role,
    featurizer_model,
    booster,
    bucket_name,
    model_report_dict,
    model_package_group_name,
    model_approval_status,
    experiment_name="main_experiment",
    run_id="run-01",
    featurizer_path=None,
    booster_path=None,
    route_stats_path=None,
):
    """
    Register Step
    - Two-container PipelineModel: sklearn featurizer -> xgboost regressor
    """

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_ARN"])
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_id=run_id):
        with mlflow.start_run(run_name="Register", nested=True):
            mlflow.autolog()
            _load_sagemaker()

            output_dir = os.environ.get(
                "REGISTER_OUTPUT_DIR", "/opt/ml/processing/output"
            )
            try:
                os.makedirs(output_dir, exist_ok=True)
            except PermissionError:
                output_dir = tempfile.mkdtemp(prefix="register-output-")

            eval_file_name = unique_name_from_base("evaluation")
            eval_report_s3_uri = s3_path_join(
                "s3://",
                bucket_name,
                model_package_group_name,
                f"evaluation-report/{eval_file_name}.json",
            )

            safe_report = to_json_safe(model_report_dict)
            s3_client = boto3.client("s3")
            s3_path = eval_report_s3_uri.replace("s3://", "")
            bucket = s3_path.split("/")[0]
            key = "/".join(s3_path.split("/")[1:])

            s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=json.dumps(safe_report).encode("utf-8"),
                ContentType="application/json",
            )

            if _HAS_MODEL_METRICS:
                model_metrics = ModelMetrics(
                    model_statistics=MetricsSource(
                        s3_uri=eval_report_s3_uri,
                        content_type="application/json",
                    )
                )
            else:
                model_metrics = None

            if Session is None:
                raise RuntimeError(
                    "sagemaker.session.Session is required to build container defs."
                )
            sm_session = Session()

            artifact_name = unique_name_from_base("featurizer")
            featurizer_s3_uri = s3_path_join(
                "s3://",
                bucket_name,
                model_package_group_name,
                f"artifacts/{artifact_name}.tar.gz",
            )

            xgb_artifact_name = unique_name_from_base("xgboost")
            xgb_s3_uri = s3_path_join(
                "s3://",
                bucket_name,
                model_package_group_name,
                f"artifacts/{xgb_artifact_name}.tar.gz",
            )

            tmpdir = tempfile.mkdtemp(prefix="register-src-")
            try:
                featurizer_tar = os.path.join(tmpdir, "featurizer.tar.gz")
                if featurizer_model is not None:
                    _create_featurizer_tar(
                        featurizer_model,
                        featurizer_tar,
                        route_stats_path=route_stats_path,
                    )
                elif featurizer_path:
                    _create_featurizer_tar_from_path(
                        featurizer_path,
                        featurizer_tar,
                        route_stats_path=route_stats_path,
                    )
                else:
                    raise ValueError(
                        "featurizer_model or featurizer_path is required."
                    )
                s3_client.upload_file(
                    featurizer_tar,
                    bucket_name,
                    f"{model_package_group_name}/artifacts/{artifact_name}.tar.gz",
                )

                xgb_tar = os.path.join(tmpdir, "xgboost.tar.gz")
                if booster is not None:
                    _create_xgboost_tar(booster, xgb_tar)
                elif booster_path:
                    _create_xgboost_tar_from_path(booster_path, xgb_tar)
                else:
                    raise ValueError(
                        "booster or booster_path is required."
                    )
                s3_client.upload_file(
                    xgb_tar,
                    bucket_name,
                    f"{model_package_group_name}/artifacts/{xgb_artifact_name}.tar.gz",
                )

                source_dir = os.path.join(tmpdir, "source")
                _write_featurizer_inference(source_dir)

                sklearn_image_uri = os.environ.get("SKLEARN_IMAGE_URI")
                if sklearn_image_uri:
                    try:
                        sklearn_model = SKLearnModel(
                            model_data=featurizer_s3_uri,
                            role=role,
                            image_uri=sklearn_image_uri,
                            entry_point="inference.py",
                            source_dir=source_dir,
                            sagemaker_session=sm_session,
                        )
                    except TypeError:
                        sklearn_model = SKLearnModel(
                            model_data=featurizer_s3_uri,
                            role=role,
                            image_uri=sklearn_image_uri,
                            entry_point="inference.py",
                            source_dir=source_dir,
                            sagemaker_session=sm_session,
                        )
                else:
                    try:
                        sklearn_model = SKLearnModel(
                            model_data=featurizer_s3_uri,
                            role=role,
                            framework_version="1.2-1",
                            py_version="py3",
                            entry_point="inference.py",
                            source_dir=source_dir,
                            sagemaker_session=sm_session,
                        )
                    except TypeError:
                        sklearn_model = SKLearnModel(
                            model_data=featurizer_s3_uri,
                            role=role,
                            framework_version="1.2-1",
                            entry_point="inference.py",
                            source_dir=source_dir,
                            sagemaker_session=sm_session,
                        )

                try:
                    xgboost_model = XGBoostModel(
                        model_data=xgb_s3_uri,
                        role=role,
                        framework_version="1.7-1",
                        py_version="py3",
                        sagemaker_session=sm_session,
                    )
                except TypeError:
                    xgboost_model = XGBoostModel(
                        model_data=xgb_s3_uri,
                        role=role,
                        framework_version="1.7-1",
                        sagemaker_session=sm_session,
                    )

                pipeline_model = PipelineModel(
                    name=unique_name_from_base("flight-price-pipeline-model"),
                    role=role,
                    models=[sklearn_model, xgboost_model],
                    sagemaker_session=sm_session,
                )

                pipeline_model.register(
                    content_types=["text/csv"],
                    response_types=["text/csv"],
                    model_package_group_name=model_package_group_name,
                    approval_status=model_approval_status,
                    model_metrics=model_metrics,
                )
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

            sm_client = boto3.client("sagemaker")
            response = sm_client.list_model_packages(
                MaxResults=100,
                ModelPackageGroupName=model_package_group_name,
                ModelPackageType="Versioned",
                SortBy="CreationTime",
                SortOrder="Descending",
            )

            model_package_arn = response["ModelPackageSummaryList"][0][
                "ModelPackageArn"
            ]
            print(f"Model registered: {model_package_arn}")

            desc = sm_client.describe_model_package(
                ModelPackageName=model_package_arn
            )
            model_data_s3_uri = desc["InferenceSpecification"]["Containers"][0][
                "ModelDataUrl"
            ]

            output_path = os.path.join(output_dir, "model_data_uri.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "model_data_s3_uri": model_data_s3_uri,
                        "model_package_arn": model_package_arn,
                    },
                    f,
                )

            with open(
                os.path.join(output_dir, "register_output.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    {
                        "model_package_arn": model_package_arn,
                        "model_data_s3_uri": model_data_s3_uri,
                    },
                    f,
                )

    return model_package_arn


def _resolve_single_file(dir_path, default_name):
    candidate = os.path.join(dir_path, default_name)
    if os.path.exists(candidate):
        return candidate
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Missing directory: {dir_path}")
    files = [
        os.path.join(dir_path, name)
        for name in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, name))
    ]
    if len(files) == 1:
        return files[0]
    raise FileNotFoundError(
        f"Could not resolve file in {dir_path}; found {len(files)} files."
    )


def main():
    input_root = "/opt/ml/processing/input"
    model_path = _resolve_single_file(
        os.path.join(input_root, "model"),
        "xgboost_model.bin",
    )
    featurizer_path = _resolve_single_file(
        os.path.join(input_root, "featurizer"),
        "featurizer.joblib",
    )
    metrics_path = _resolve_single_file(
        os.path.join(input_root, "metrics"),
        "test_metrics.json",
    )
    try:
        route_stats_path = _resolve_single_file(
            os.path.join(input_root, "route_stats"),
            "route_stats.csv",
        )
    except FileNotFoundError:
        route_stats_path = None

    with open(metrics_path, "r", encoding="utf-8") as f:
        model_report_dict = json.load(f)

    role = os.environ.get("SAGEMAKER_ROLE_ARN") or os.environ.get("SM_ROLE_ARN")
    if not role:
        raise RuntimeError(
            "SAGEMAKER_ROLE_ARN (or SM_ROLE_ARN) is required for register step."
        )

    bucket_name = os.environ.get("BUCKET_NAME")
    if not bucket_name:
        raise RuntimeError("BUCKET_NAME is required for register step.")

    model_package_group_name = os.environ.get("MODEL_PACKAGE_GROUP_NAME")
    if not model_package_group_name:
        raise RuntimeError(
            "MODEL_PACKAGE_GROUP_NAME is required for register step."
        )

    model_approval_status = os.environ.get(
        "MODEL_APPROVAL_STATUS", "PendingManualApproval"
    )
    experiment_name = os.environ.get("EXPERIMENT_NAME", "main_experiment")
    run_id = os.environ.get("RUN_ID")

    if not run_id:
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_ARN"])
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name="RegisterProcessingBootstrap") as run:
            run_id = run.info.run_id

    register(
        role=role,
        featurizer_model=None,
        booster=None,
        featurizer_path=featurizer_path,
        booster_path=model_path,
        route_stats_path=route_stats_path,
        bucket_name=bucket_name,
        model_report_dict=model_report_dict,
        model_package_group_name=model_package_group_name,
        model_approval_status=model_approval_status,
        experiment_name=experiment_name,
        run_id=run_id,
    )


if __name__ == "__main__":
    main()
