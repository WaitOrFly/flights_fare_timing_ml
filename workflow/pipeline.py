import os
import tempfile
import urllib

import boto3
import joblib
import xgboost as xgb
from steps.preprocess import preprocess
from steps.train import train
from steps.test import test

from sagemaker import get_execution_role
from sagemaker import image_uris
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.s3 import S3Uploader
from sagemaker.session import Session
from sagemaker.workflow.function_step import step
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import (
    ParameterBoolean,
    ParameterFloat,
    ParameterInteger,
    ParameterString,
)

from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.pipeline_definition_config import PipelineDefinitionConfig
from sagemaker.workflow.pipeline_experiment_config import PipelineExperimentConfig
from sagemaker.workflow.pipeline_context import LocalPipelineSession, PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep


def download_data_and_upload_to_s3(bucket_name: str) -> str:
    file_name = "flight_fares.csv"
    s3_prefix = "flight-fare/data"
    s3_uri = f"s3://{bucket_name}/{s3_prefix}"

    input_data_dir = "/tmp/data/"
    input_data_path = os.path.join(input_data_dir, file_name)
    os.makedirs(os.path.dirname(input_data_path), exist_ok=True)

    dataset_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
    urllib.request.urlretrieve(dataset_url, input_data_path)

    upload_s3_uri = S3Uploader.upload(input_data_path, s3_uri)
    print("Downloading dataset and uploading to Amazon S3...")
    print(upload_s3_uri)

    return upload_s3_uri


def _normalize_s3_uri(s3_uri: str) -> str:
    if not s3_uri.startswith("s3://"):
        raise ValueError("model_artifacts_s3_uri must start with 's3://'")
    return s3_uri.rstrip("/")


def package_model_artifacts(
    featurizer_model,
    booster: xgb.Booster,
    model_artifacts_s3_uri: str,
):
    model_artifacts_s3_uri = _normalize_s3_uri(model_artifacts_s3_uri)
    with tempfile.TemporaryDirectory() as temp_dir:
        featurizer_path = os.path.join(temp_dir, "featurizer.joblib")
        booster_path = os.path.join(temp_dir, "xgboost-model.json")
        joblib.dump(featurizer_model, featurizer_path)
        booster.save_model(booster_path)

        featurizer_s3_uri = S3Uploader.upload(
            featurizer_path, f"{model_artifacts_s3_uri}/featurizer"
        )
        booster_s3_uri = S3Uploader.upload(
            booster_path, f"{model_artifacts_s3_uri}/xgboost"
        )
    return {
        "featurizer_s3_uri": featurizer_s3_uri,
        "booster_s3_uri": booster_s3_uri,
    }


def create_steps(
    role,
    input_data_s3_uri,
    output_data_s3_uri,
    project_prefix,
    bucket_name,
    model_artifacts_s3_uri,
    model_package_group_name,
    model_approval_status,
    eta_parameter,
    max_depth_parameter,
    min_child_weight_parameter,
    subsample_parameter,
    colsample_bytree_parameter,
    gamma_parameter,
    reg_lambda_parameter,
    reg_alpha_parameter,
    num_boost_round_parameter,
    early_stopping_rounds_parameter,
    deploy_model_parameter,
    experiment_name,
    run_name,
    mlflow_arn,
    pipeline_session,
):
    env_variables = {"MLFLOW_TRACKING_ARN": mlflow_arn}

    preprocess_result = step(
        preprocess,
        name="Preprocess",
        job_name_prefix=f"{project_prefix}-Preprocess",
        keep_alive_period_in_seconds=300,
        environment_variables=env_variables,
    )(input_data_s3_uri, output_data_s3_uri, experiment_name, run_name)

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
        eta=eta_parameter,
        max_depth=max_depth_parameter,
        min_child_weight=min_child_weight_parameter,
        subsample=subsample_parameter,
        colsample_bytree=colsample_bytree_parameter,
        gamma=gamma_parameter,
        reg_lambda=reg_lambda_parameter,
        reg_alpha=reg_alpha_parameter,
        num_boost_round=num_boost_round_parameter,
        early_stopping_rounds=early_stopping_rounds_parameter,
        experiment_name=experiment_name,
        run_id=run_name,
    )

    test_result = step(
        test,
        name="Evaluate",
        job_name_prefix=f"{project_prefix}-Test",
        keep_alive_period_in_seconds=300,
        environment_variables=env_variables,
    )(
        featurizer_model=preprocess_result[6],
        booster=train_result,
        X_test=preprocess_result[4],
        y_test=preprocess_result[5],
        bucket_name=bucket_name,
        model_package_group_name=model_package_group_name,
        experiment_name=experiment_name,
        run_id=run_name,
    )

    package_model_result = step(
        package_model_artifacts,
        name="PackageModels",
        job_name_prefix=f"{project_prefix}-PackageModels",
        keep_alive_period_in_seconds=300,
        environment_variables=env_variables,
    )(
        featurizer_model=preprocess_result[6],
        booster=train_result,
        model_artifacts_s3_uri=model_artifacts_s3_uri,
    )

    register_output = PropertyFile(
        name="RegisterOutput",
        output_name="register_output",
        path="model_package.json",
    )
    processing_instance_type = os.environ.get(
        "REGISTER_PROCESSING_INSTANCE_TYPE", "ml.m5.xlarge"
    )
    processing_image_uri = image_uris.retrieve(
        framework="sklearn",
        region=Session().boto_region_name,
        version="1.2-1",
        instance_type=processing_instance_type,
    )
    register_processor = ScriptProcessor(
        image_uri=processing_image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{project_prefix}-register",
        role=role,
        env=env_variables,
        sagemaker_session=pipeline_session,
    )
    register_step_args = register_processor.run(
        code=os.path.join(os.path.dirname(__file__), "steps", "register.py"),
        inputs=[
            ProcessingInput(
                source=os.path.join(
                    os.path.dirname(__file__), "requirements_inference.txt"
                ),
                destination="/opt/ml/processing/requirements",
            ),
            ProcessingInput(
                source=package_model_result["featurizer_s3_uri"],
                destination="/opt/ml/processing/featurizer",
            ),
            ProcessingInput(
                source=package_model_result["booster_s3_uri"],
                destination="/opt/ml/processing/xgboost",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="register_output",
                source="/opt/ml/processing/output",
            )
        ],
        arguments=[
            "--role-arn",
            role,
            "--featurizer-model-path",
            "/opt/ml/processing/featurizer/featurizer.joblib",
            "--xgboost-model-path",
            "/opt/ml/processing/xgboost/xgboost-model.json",
            "--model-report-path",
            test_result,
            "--bucket-name",
            bucket_name,
            "--model-package-group-name",
            model_package_group_name,
            "--model-approval-status",
            model_approval_status,
            "--experiment-name",
            experiment_name,
            "--run-id",
            run_name,
            "--output-model-package-path",
            "/opt/ml/processing/output/model_package.json",
            "--requirements-path",
            "/opt/ml/processing/requirements/requirements_inference.txt",
        ],
    )
    register_step = ProcessingStep(
        name="Register",
        step_args=register_step_args,
        property_files=[register_output],
    )
    model_package_arn = JsonGet(
        step_name=register_step.name,
        property_file=register_output,
        json_path="model_package_arn",
    )

    deploy_processor = ScriptProcessor(
        image_uri=processing_image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{project_prefix}-deploy",
        role=role,
        env=env_variables,
        sagemaker_session=pipeline_session,
    )
    deploy_step_args = deploy_processor.run(
        code=os.path.join(os.path.dirname(__file__), "steps", "deploy.py"),
        arguments=[
            "--role-arn",
            role,
            "--project-prefix",
            project_prefix,
            "--model-package-arn",
            model_package_arn,
            "--deploy-model",
            Join(on="", values=[deploy_model_parameter]),
            "--experiment-name",
            experiment_name,
            "--run-id",
            run_name,
        ],
    )
    deploy_step = ProcessingStep(
        name="Deploy",
        step_args=deploy_step_args,
        depends_on=[register_step],
    )

    return [register_step, deploy_step]


def get_mlflow_server_arn():
    r = boto3.client("sagemaker").list_mlflow_tracking_servers()["TrackingServerSummaries"]

    if len(r) < 1:
        print("You don't have any running MLflow servers. Please create an MLflow server first.")
        return ""
    mlflow_arn = r[0]["TrackingServerArn"]
    print(f"You have {len(r)} running MLflow server(s). Get the first server. Details: {r[0]}")
    return mlflow_arn


if __name__ == "__main__":
    os.environ["SAGEMAKER_USER_CONFIG_OVERRIDE"] = os.getcwd()

    mlflow_arn = os.environ.get("MLFLOW_TRACKING_ARN") or get_mlflow_server_arn()
    if mlflow_arn:
        os.environ["MLFLOW_TRACKING_ARN"] = mlflow_arn

    local_mode = os.getenv("LOCAL_MODE", "").lower() == "true"
    role = get_execution_role()

    session = Session()
    bucket_name = session.default_bucket()
    project_prefix = os.environ.get("PROJECT_PREFIX", "amzn")
    bucket_prefix = os.environ.get("BUCKET_PREFIX", f"{bucket_name}/{project_prefix}")

    pipeline_name = f"{project_prefix}-flight-fare-pipeline"
    model_package_group_name = f"{project_prefix}-flight-fare-model-package-group"
    model_approval_status = "PendingManualApproval"
    experiment_name = pipeline_name
    run_name = ExecutionVariables.PIPELINE_EXECUTION_ID

    input_data_s3_uri = os.environ.get(
        "INPUT_DATA_S3_URI", f"s3://{bucket_prefix}/data/flight_fares.csv"
    )
    output_data_s3_uri = os.environ.get(
        "OUTPUT_DATA_S3_URI", f"s3://{bucket_prefix}/processed"
    )
    model_artifacts_s3_uri = os.environ.get(
        "MODEL_ARTIFACTS_S3_URI", f"s3://{bucket_prefix}/model-artifacts"
    )

    input_data_param = ParameterString(name="input_data_s3_uri", default_value=input_data_s3_uri)
    output_data_param = ParameterString(name="output_data_s3_uri", default_value=output_data_s3_uri)
    model_artifacts_param = ParameterString(
        name="model_artifacts_s3_uri", default_value=model_artifacts_s3_uri
    )

    eta_parameter = ParameterFloat(name="eta", default_value=0.3)
    max_depth_parameter = ParameterInteger(name="max_depth", default_value=6)
    min_child_weight_parameter = ParameterFloat(name="min_child_weight", default_value=1.0)
    subsample_parameter = ParameterFloat(name="subsample", default_value=0.8)
    colsample_bytree_parameter = ParameterFloat(name="colsample_bytree", default_value=0.8)
    gamma_parameter = ParameterFloat(name="gamma", default_value=0.0)
    reg_lambda_parameter = ParameterFloat(name="reg_lambda", default_value=1.0)
    reg_alpha_parameter = ParameterFloat(name="reg_alpha", default_value=0.0)
    num_boost_round_parameter = ParameterInteger(name="num_boost_round", default_value=200)
    early_stopping_rounds_parameter = ParameterInteger(name="early_stopping_rounds", default_value=20)
    deploy_model_parameter = ParameterBoolean(name="deploy_model", default_value=True)

    pipeline_session = LocalPipelineSession() if local_mode else PipelineSession()
    steps = create_steps(
        role,
        input_data_param,
        output_data_param,
        project_prefix,
        bucket_name,
        model_artifacts_param,
        model_package_group_name,
        model_approval_status,
        eta_parameter,
        max_depth_parameter,
        min_child_weight_parameter,
        subsample_parameter,
        colsample_bytree_parameter,
        gamma_parameter,
        reg_lambda_parameter,
        reg_alpha_parameter,
        num_boost_round_parameter,
        early_stopping_rounds_parameter,
        deploy_model_parameter,
        experiment_name,
        run_name,
        mlflow_arn,
        pipeline_session,
    )

    more_params = {"sagemaker_session": pipeline_session}

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            input_data_param,
            output_data_param,
            model_artifacts_param,
            deploy_model_parameter,
            eta_parameter,
            max_depth_parameter,
            min_child_weight_parameter,
            subsample_parameter,
            colsample_bytree_parameter,
            gamma_parameter,
            reg_lambda_parameter,
            reg_alpha_parameter,
            num_boost_round_parameter,
            early_stopping_rounds_parameter,
        ],
        steps=steps,
        pipeline_definition_config=PipelineDefinitionConfig(use_custom_job_prefix=True),
        pipeline_experiment_config=PipelineExperimentConfig(
            experiment_name=experiment_name, trial_name=run_name
        ),
        **more_params,
    )

    pipeline.upsert(role_arn=role)
    pipeline.start()
