import os
import boto3

from steps.preprocess import preprocess
from steps.train import train
from steps.test import test
# from steps.register import register
from steps.deploy import deploy

from sagemaker.session import Session
from sagemaker import get_execution_role
from sagemaker.workflow.function_step import step
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import (
    ParameterBoolean,
    ParameterFloat,
    ParameterInteger,
)
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.pipeline_definition_config import PipelineDefinitionConfig
from sagemaker.workflow.pipeline_context import LocalPipelineSession
from sagemaker.processing import (
    ScriptProcessor,
    ProcessingInput,
    ProcessingOutput,
)
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.properties import PropertyFile
from sagemaker.sklearn.processing import SKLearnProcessor


# =====================================================
# MLflow Tracking Server ARN
# =====================================================
def get_mlflow_server_arn():
    r = boto3.client("sagemaker").list_mlflow_tracking_servers()[
        "TrackingServerSummaries"
    ]

    if len(r) < 1:
        raise RuntimeError(
            "No running MLflow tracking server found."
        )

    mlflow_arn = r[0]["TrackingServerArn"]
    print(f"Using MLflow server: {mlflow_arn}")
    return mlflow_arn


# =====================================================
# Pipeline Steps
# =====================================================
def create_steps(
    role,
    input_data_s3_uri,
    output_data_s3_uri,
    project_prefix,
    bucket_name,
    model_package_group_name,
    model_approval_status,
    eta_parameter,
    max_depth_parameter,
    deploy_model_parameter,
    experiment_name,
    run_name,
    mlflow_arn,
):

    env_variables = {
        "MLFLOW_TRACKING_ARN": mlflow_arn,
        "SAGEMAKER_ROLE_ARN": role,
        "BUCKET_NAME": bucket_name,
        "MODEL_PACKAGE_GROUP_NAME": model_package_group_name,
        "MODEL_APPROVAL_STATUS": model_approval_status,
        "EXPERIMENT_NAME": experiment_name,
        "REGISTER_OUTPUT_DIR": "/opt/ml/processing/output",
    }

    # -------------------------
    # Preprocess
    # -------------------------
    preprocess_result = step(
        preprocess,
        name="Preprocess",
        job_name_prefix=f"{project_prefix}-Preprocess",
        keep_alive_period_in_seconds=300,
        environment_variables=env_variables,
    )(
        input_data_s3_uri=input_data_s3_uri,
        output_data_s3_uri=output_data_s3_uri,
        experiment_name=experiment_name,
        run_name=run_name,
    )

    # preprocess_result index map
    # 0  X_train
    # 1  y_train_log
    # 2  X_val
    # 3  y_val_log
    # 4  y_val_original
    # 5  X_test
    # 6  y_test_log
    # 7  y_test_original
    # 8  featurizer_model
    # 9  run_id
    # -------------------------
    # Train
    # -------------------------
    train_result = step(
        train,
        name="Train",
        job_name_prefix=f"{project_prefix}-Train",
        keep_alive_period_in_seconds=300,
        environment_variables=env_variables,
    )(
        X_train=preprocess_result[0],
        y_train=preprocess_result[1],
        X_val=preprocess_result[2],
        y_val=preprocess_result[3],
        y_val_original=preprocess_result[4],
        output_model_s3_uri=(
            f"s3://{bucket_name}/{project_prefix}/model/xgboost_model.bin"
        ),
        eta=eta_parameter,
        max_depth=max_depth_parameter,
        experiment_name=experiment_name,
        run_id=preprocess_result[8],
    )

    # train_result = (booster, metrics)

    featurizer_s3_uri = (
        f"{output_data_s3_uri}/featurizer/featurizer.joblib"
    )
    
    test_metrics_s3_uri = f"s3://{bucket_name}/{project_prefix}/metrics/test_metrics.json"

    # -------------------------
    # Test
    # -------------------------
    test_result = step(
        test,
        name="Test",
        job_name_prefix=f"{project_prefix}-Test",
        keep_alive_period_in_seconds=300,
        environment_variables=env_variables,
    )(
        featurizer_s3_uri=featurizer_s3_uri,
        model_s3_uri=train_result[0],
        X_test=preprocess_result[5],
        y_test=preprocess_result[6],
        y_test_original=preprocess_result[7],
        output_metrics_s3_uri=test_metrics_s3_uri,
        experiment_name=experiment_name,
        run_id=preprocess_result[8],
    )

    # # -------------------------
    # # Register
    # # -------------------------
    # register_result = step(
    #     register,
    #     name="Register",
    #     job_name_prefix=f"{project_prefix}-Register",
    #     keep_alive_period_in_seconds=300,
    #     environment_variables=env_variables,
    # )(
    #     role=role,
    #     featurizer_model=preprocess_result[8],
    #     booster=train_result[0],
    #     bucket_name=bucket_name,
    #     model_report_dict=test_result,
    #     model_package_group_name=model_package_group_name,
    #     model_approval_status=model_approval_status,
    #     experiment_name=experiment_name,
    #     run_id=preprocess_result[9],
    # )

    # -------------------------
    # Register (ProcessingJob)
    # -------------------------

    register_processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_type="ml.m5.2xlarge",
        instance_count=1,
        env=env_variables,
    )

    register_property = PropertyFile(
        name="RegisterOutput",
        output_name="processing-output",
        path="register_output.json",
    )

    register_script_path = os.path.join(
        os.path.dirname(__file__), "steps", "register.py"
    )

    register_result = ProcessingStep(
        name="RegisterProcessing",
        processor=register_processor,
        inputs=[
            # Train 결과 (model.bin)
            ProcessingInput(
                source=train_result[0],
                destination="/opt/ml/processing/input/model",
            ),
            # Featurizer
            ProcessingInput(
                source=featurizer_s3_uri,
                destination="/opt/ml/processing/input/featurizer",
            ),
            # Test metrics
            ProcessingInput(
                source=test_result,
                destination="/opt/ml/processing/input/metrics",
            ),
            # Route stats
            ProcessingInput(
                source=f"{output_data_s3_uri}/route_stats.csv",
                destination="/opt/ml/processing/input/route_stats",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="processing-output",  # 🔥 이름 반드시 일치
                source="/opt/ml/processing/output",
                destination=f"s3://{bucket_name}/{project_prefix}/registered",
            )
        ],
        property_files=[register_property],   # 🔥 추가
        code=register_script_path,
    )


    # -------------------------
    # Deploy
    # -------------------------
    # deploy_result = step(
    #     deploy,
    #     name="Deploy",
    #     job_name_prefix=f"{project_prefix}-Deploy",
    #     keep_alive_period_in_seconds=300,
    #     environment_variables=env_variables,
    # )(
    #     role=role,
    #     project_prefix=project_prefix,
    #     model_package_arn=register_result,
    #     deploy_model=deploy_model_parameter,
    #     experiment_name=experiment_name,
    #     run_id=preprocess_result[9],
    # )

    deploy_result = step(
        deploy,
        name="Deploy",
        job_name_prefix=f"{project_prefix}-Deploy",
        keep_alive_period_in_seconds=300,
        environment_variables=env_variables,
    )(
        role=role,
        project_prefix=project_prefix,
        register_output_s3_uri=(
            register_result
            .properties
            .ProcessingOutputConfig
            .Outputs["processing-output"]
            .S3Output
            .S3Uri
        ),
        deploy_model=deploy_model_parameter,
        experiment_name=experiment_name,
        run_id=preprocess_result[9],
    )


    # create_steps() 맨 마지막
    return [
        preprocess_result,
        train_result,
        test_result,
        register_result,
        deploy_result,
    ]



# =====================================================
# Main
# =====================================================
if __name__ == "__main__":
    # os.environ["SAGEMAKER_USER_CONFIG_OVERRIDE"] = os.getcwd()
    os.environ["SAGEMAKER_USER_CONFIG_OVERRIDE"] = (
        "/home/sagemaker-user/flights_fare_timing_ml/workflow/config.yaml"
    )

    # MLflow
    mlflow_arn = os.environ.get(
        "MLFLOW_TRACKING_ARN", get_mlflow_server_arn()
    )
    os.environ["MLFLOW_TRACKING_ARN"] = mlflow_arn

    # SageMaker context
    local_mode = os.getenv("LOCAL_MODE", False)
    role = get_execution_role()
    session = Session()
    bucket_name = session.default_bucket()

    # Project config
    project_prefix = "flight-fares-timing"
    pipeline_name = f"{project_prefix}-pipeline"
    model_package_group_name = (
        f"{project_prefix}-model-package-group"
    )
    model_approval_status = "PendingManualApproval"

    experiment_name = pipeline_name
    run_name = ExecutionVariables.PIPELINE_EXECUTION_ID

    # Pipeline parameters
    eta_parameter = ParameterFloat(
        name="eta", default_value=0.1
    )
    max_depth_parameter = ParameterInteger(
        name="max_depth", default_value=6
    )
    deploy_model_parameter = ParameterBoolean(
        name="deploy_model", default_value=True
    )

    # Data paths
    input_data_s3_uri = (
        f"s3://{bucket_name}/{project_prefix}/raw/data.csv"
    )
    output_data_s3_uri = (
        f"s3://{bucket_name}/{project_prefix}/processed"
    )

    steps = create_steps(
        role=role,
        input_data_s3_uri=input_data_s3_uri,
        output_data_s3_uri=output_data_s3_uri,
        project_prefix=project_prefix,
        bucket_name=bucket_name,
        model_package_group_name=model_package_group_name,
        model_approval_status=model_approval_status,
        eta_parameter=eta_parameter,
        max_depth_parameter=max_depth_parameter,
        deploy_model_parameter=deploy_model_parameter,
        experiment_name=experiment_name,
        run_name=run_name,
        mlflow_arn=mlflow_arn,
    )

    local_pipeline_session = LocalPipelineSession()

    extra_args = {}
    if local_mode:
        extra_args["sagemaker_session"] = local_pipeline_session

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            deploy_model_parameter,
            eta_parameter,
            max_depth_parameter,
        ],
        steps=steps,
        pipeline_definition_config=PipelineDefinitionConfig(
            use_custom_job_prefix=True
        ),
        **extra_args,
    )

    pipeline.upsert(role_arn=role)
    pipeline.start()
