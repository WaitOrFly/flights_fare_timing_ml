import argparse
import os

from sagemaker.model import ModelPackage
from sagemaker.predictor import Predictor
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
    endpoint_name = unique_name_from_base(f"{project_prefix}-endpoint")
    endpoint_config_name = unique_name_from_base(f"{endpoint_name}-config")
    sm_client = session.boto_session.client("sagemaker")
    try:
        sm_client.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
        endpoint_config_name = unique_name_from_base(f"{endpoint_name}-config")
    except sm_client.exceptions.ClientError:
        pass

    model_name = unique_name_from_base(f"{project_prefix}-model")
    model = ModelPackage(
        model_package_arn=model_package_arn,
        role=role,
        sagemaker_session=session,
        name=model_name,
    )
    print(f"[Deploy] model_package_arn: {model_package_arn}")
    print(f"[Deploy] endpoint_name: {endpoint_name}")
    print(f"[Deploy] endpoint_config_name: {endpoint_config_name}")
    print("[Deploy] volume_size_in_gb: 50")

    instance_type = "ml.m5.large"
    initial_instance_count = 1
    volume_size_in_gb = 50

    print(f"[Deploy] model_name: {model_name}")
    model.create(instance_type=instance_type)

    try:
        response = sm_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": model_name,
                    "InitialInstanceCount": initial_instance_count,
                    "InstanceType": instance_type,
                    "InitialVariantWeight": 1.0,
                    "VolumeSizeInGB": volume_size_in_gb,
                }
            ],
            Tags=[{"Key": "source", "Value": "deploy_step"}],
        )
        request_id = response.get("ResponseMetadata", {}).get("RequestId")
        print(f"[Deploy] create_endpoint_config request_id: {request_id}")
        config_details = sm_client.describe_endpoint_config(
            EndpointConfigName=endpoint_config_name
        )
        print(f"[Deploy] endpoint_config_details: {config_details}")
    except Exception as exc:
        print(f"[Deploy] create_endpoint_config failed: {exc}")
        raise

    try:
        response = sm_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
        )
        request_id = response.get("ResponseMetadata", {}).get("RequestId")
        print(f"[Deploy] create_endpoint request_id: {request_id}")
    except Exception as exc:
        print(f"[Deploy] create_endpoint failed: {exc}")
        raise
    session.wait_for_endpoint(endpoint_name)

    predictor = Predictor(
        endpoint_name=endpoint_name,
        sagemaker_session=session,
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
                        "endpoint_config_name": endpoint_config_name,
                        "instance_type": instance_type,
                        "initial_instance_count": initial_instance_count,
                        "volume_size_in_gb": volume_size_in_gb,
                    }
                )

    return predictor


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy model package in SageMaker.")
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--project-prefix", required=True)
    parser.add_argument("--model-package-arn", required=True)
    parser.add_argument("--deploy-model", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    deploy(
        role=args.role_arn,
        project_prefix=args.project_prefix,
        model_package_arn=args.model_package_arn,
        deploy_model=_parse_bool(args.deploy_model),
        experiment_name=args.experiment_name,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
