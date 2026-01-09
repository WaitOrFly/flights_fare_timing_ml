import os

from sagemaker.model import ModelPackage
from sagemaker.session import Session


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


def deploy(
    role: str,
    project_prefix: str,
    model_package_arn: str,
    deploy_model: bool,
    experiment_name: str,
    run_id: str,
):
    if not deploy_model:
        return None

    session = Session()
    endpoint_name = f"{project_prefix}-endpoint"

    model = ModelPackage(
        model_package_arn=model_package_arn,
        role=role,
        sagemaker_session=session,
    )

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.large",
        endpoint_name=endpoint_name,
    )

    mlflow = _get_mlflow()
    if mlflow:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_id=run_id):
            with mlflow.start_run(run_name="Deploy", nested=True):
                mlflow.autolog()
                mlflow.log_params(
                    {
                        "model_package_arn": model_package_arn,
                        "endpoint_name": endpoint_name,
                        "instance_type": "ml.m5.large",
                        "initial_instance_count": 1,
                    }
                )

    return predictor
