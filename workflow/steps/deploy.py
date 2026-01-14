import os
import time
import json
import boto3
import mlflow

try:
    from sagemaker.utils import unique_name_from_base
except Exception:
    def unique_name_from_base(base):
        return f"{base}-{int(time.time())}"

try:
    from sagemaker.model import ModelPackage
    _HAS_MODEL_PACKAGE = True
except ModuleNotFoundError:
    _HAS_MODEL_PACKAGE = False


def _parse_s3_uri(s3_uri: str):
    s3_path = s3_uri.replace("s3://", "")
    bucket = s3_path.split("/")[0]
    key = "/".join(s3_path.split("/")[1:])
    return bucket, key


def _load_register_output(register_output_s3_uri: str):
    s3_client = boto3.client("s3")
    bucket, key_prefix = _parse_s3_uri(register_output_s3_uri)
    key = key_prefix.rstrip("/") + "/register_output.json"
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read().decode("utf-8")


def deploy(
    role,
    project_prefix,
    model_data_s3_uri=None,
    deploy_model: bool = True,
    model_package_arn=None,
    register_output_s3_uri=None,
    instance_type="ml.m5.2xlarge",
    experiment_name="main_experiment",
    run_id="run-01",
):
    """
    Deploy Step

    - SageMaker Model Registry에 등록된 ModelPackage 기반 배포
    - MLflow parent run + nested Deploy run 유지
    - deploy_model=False 인 경우 배포 스킵
    """

    # ===== MLflow 설정 =====
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_ARN"])
    mlflow.set_experiment(experiment_name)

    # sm_client = boto3.client("sagemaker")
    # sm_session = Session()

    with mlflow.start_run(run_id=run_id) as run:
        with mlflow.start_run(run_name="Deploy", nested=True):
            mlflow.autolog()

            if not deploy_model:
                print("🚫 Deploy step skipped (deploy_model=False)")
                mlflow.set_tag("deploy_status", "skipped")
                return None

            # print(f"🚀 Deploying model package: {model_package_arn}")

            # sm_client = boto3.client("sagemaker")
            # ===== ModelPackage 승인 =====
            # response = sm_client.update_model_package(
            #         ModelPackageArn=model_package_arn,
            #         ModelApprovalStatus='Approved',
            #         ApprovalDescription='Auto-approved via SageMaker Pipelines')

            # model_package = ModelPackage(
            #     role=role,
            #     model_package_arn=model_package_arn
            # )

            endpoint_name = unique_name_from_base(f"{project_prefix}-endpoint")

            print(f"🚀 Deploying model from: {model_data_s3_uri}")
            print(f"📡 Endpoint name: {endpoint_name}")

            # model_package.deploy(initial_instance_count=1,
            #                     instance_type=instance_type,
            #                     endpoint_name=endpoint_name)

            # # ===== Endpoint 존재 여부 확인 =====
            # endpoint_exists = True
            # try:
            #     sm_client.describe_endpoint(
            #         EndpointName=endpoint_name
            #     )
            #     print(f"🔁 Existing endpoint found: {endpoint_name}")
            # except ClientError as e:
            #     if "Could not find endpoint" in str(e):
            #         endpoint_exists = False
            #     else:
            #         raise

            # # ===== Deploy or Update =====
            # if endpoint_exists:
            #     print(f"🔄 Updating endpoint: {endpoint_name}")
            #     model_package.deploy(
            #         initial_instance_count=1,
            #         instance_type=instance_type,
            #         endpoint_name=endpoint_name,
            #         update_endpoint=True,
            #     )
            #     deploy_mode = "update"
            # else:
            #     print(f"📡 Creating endpoint: {endpoint_name}")
            #     model_package.deploy(
            #         initial_instance_count=1,
            #         instance_type=instance_type,
            #         endpoint_name=endpoint_name,
            #     )
            #     deploy_mode = "create"
            if register_output_s3_uri and (not model_data_s3_uri or not model_package_arn):
                register_output = _load_register_output(register_output_s3_uri)
                register_json = json.loads(register_output)
                model_data_s3_uri = model_data_s3_uri or register_json.get(
                    "model_data_s3_uri"
                )
                model_package_arn = model_package_arn or register_json.get(
                    "model_package_arn"
                )

            if not model_data_s3_uri:
                raise RuntimeError(
                    "model_data_s3_uri is required to deploy the PipelineModel."
                )
            if not model_package_arn:
                raise RuntimeError(
                    "model_package_arn is required to deploy the PipelineModel."
                )
            if not _HAS_MODEL_PACKAGE:
                raise RuntimeError(
                    "sagemaker.model.ModelPackage is not available in this SDK."
                )

            model_package = ModelPackage(
                role=role,
                model_package_arn=model_package_arn,
            )
            predictor = model_package.deploy(
                initial_instance_count=1,
                instance_type=instance_type,
                endpoint_name=endpoint_name,
            )

            # ===== MLflow Logging =====
            mlflow.log_param("endpoint_name", endpoint_name)
            mlflow.log_param("instance_type", instance_type)
            mlflow.set_tag("deploy_status", "success")

            print(f"✅ Deployment completed: {endpoint_name}")

    return endpoint_name
