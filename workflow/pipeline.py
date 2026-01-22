import os
import urllib

import boto3

from sagemaker import get_execution_role
from sagemaker import image_uris
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.s3 import S3Uploader
from sagemaker.session import Session
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
    base_score_parameter,
    deploy_model_parameter,
    experiment_name,
    run_name,
    mlflow_arn,
    pipeline_session,
):
    env_variables = {"MLFLOW_TRACKING_ARN": mlflow_arn}

    processing_instance_type = os.environ.get(
        "PROCESSING_INSTANCE_TYPE", "ml.m5.xlarge"
    )
    processing_image_uri = image_uris.retrieve(
        framework="sklearn",
        region=Session().boto_region_name,
        version="1.2-1",
        instance_type=processing_instance_type,
    )
    steps_dir = os.path.join(os.path.dirname(__file__), "steps")

    preprocess_output_s3 = Join(
        on="/", values=[output_data_s3_uri, "processing", "preprocess"]
    )
    preprocess_processor = ScriptProcessor(
        image_uri=processing_image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{project_prefix}-preprocess",
        role=role,
        env=env_variables,
        sagemaker_session=pipeline_session,
    )
    preprocess_step_args = preprocess_processor.run(
        code=os.path.join(steps_dir, "run_preprocess.py"),
        inputs=[
            ProcessingInput(
                source=steps_dir,
                destination="/opt/ml/processing/code",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="preprocess_output",
                source="/opt/ml/processing/output",
                destination=preprocess_output_s3,
            )
        ],
        arguments=[
            "--input-data-s3-uri",
            input_data_s3_uri,
            "--output-data-s3-uri",
            output_data_s3_uri,
            "--experiment-name",
            experiment_name,
            "--run-id",
            run_name,
        ],
    )
    preprocess_step = ProcessingStep(
        name="Preprocess",
        step_args=preprocess_step_args,
    )

    train_output_s3 = Join(on="/", values=[output_data_s3_uri, "processing", "train"])
    train_processor = ScriptProcessor(
        image_uri=processing_image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{project_prefix}-train",
        role=role,
        env=env_variables,
        sagemaker_session=pipeline_session,
    )
    train_step_args = train_processor.run(
        code=os.path.join(steps_dir, "run_train.py"),
        inputs=[
            ProcessingInput(
                source=steps_dir,
                destination="/opt/ml/processing/code",
            ),
            ProcessingInput(
                source=preprocess_step.properties.ProcessingOutputConfig.Outputs[
                    "preprocess_output"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/input",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="train_output",
                source="/opt/ml/processing/output",
                destination=train_output_s3,
            )
        ],
        arguments=[
            "--input-dir",
            "/opt/ml/processing/input",
            "--output-dir",
            "/opt/ml/processing/output",
            "--eta",
            Join(on="", values=[eta_parameter]),
            "--max-depth",
            Join(on="", values=[max_depth_parameter]),
            "--min-child-weight",
            Join(on="", values=[min_child_weight_parameter]),
            "--subsample",
            Join(on="", values=[subsample_parameter]),
            "--colsample-bytree",
            Join(on="", values=[colsample_bytree_parameter]),
            "--gamma",
            Join(on="", values=[gamma_parameter]),
            "--reg-lambda",
            Join(on="", values=[reg_lambda_parameter]),
            "--reg-alpha",
            Join(on="", values=[reg_alpha_parameter]),
            "--num-boost-round",
            Join(on="", values=[num_boost_round_parameter]),
            "--early-stopping-rounds",
            Join(on="", values=[early_stopping_rounds_parameter]),
            "--base-score",
            Join(on="", values=[base_score_parameter]),
            "--model-artifacts-s3-uri",
            model_artifacts_s3_uri,
            "--experiment-name",
            experiment_name,
            "--run-id",
            run_name,
        ],
    )
    train_step = ProcessingStep(
        name="Train",
        step_args=train_step_args,
    )

    test_output_s3 = Join(on="/", values=[output_data_s3_uri, "processing", "test"])
    test_processor = ScriptProcessor(
        image_uri=processing_image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{project_prefix}-test",
        role=role,
        env=env_variables,
        sagemaker_session=pipeline_session,
    )
    test_step_args = test_processor.run(
        code=os.path.join(steps_dir, "run_test.py"),
        inputs=[
            ProcessingInput(
                source=steps_dir,
                destination="/opt/ml/processing/code",
            ),
            ProcessingInput(
                source=preprocess_step.properties.ProcessingOutputConfig.Outputs[
                    "preprocess_output"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/input",
            ),
            ProcessingInput(
                source=train_step.properties.ProcessingOutputConfig.Outputs[
                    "train_output"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/model",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="test_output",
                source="/opt/ml/processing/output",
                destination=test_output_s3,
            )
        ],
        arguments=[
            "--input-dir",
            "/opt/ml/processing/input",
            "--model-dir",
            "/opt/ml/processing/model",
            "--output-dir",
            "/opt/ml/processing/output",
            "--bucket-name",
            bucket_name,
            "--model-package-group-name",
            model_package_group_name,
            "--experiment-name",
            experiment_name,
            "--run-id",
            run_name,
        ],
    )
    test_step = ProcessingStep(
        name="Evaluate",
        step_args=test_step_args,
    )

    register_output = PropertyFile(
        name="RegisterOutput",
        output_name="register_output",
        path="model_package.json",
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
                source=preprocess_step.properties.ProcessingOutputConfig.Outputs[
                    "preprocess_output"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/preprocess",
            ),
            ProcessingInput(
                source=train_step.properties.ProcessingOutputConfig.Outputs[
                    "train_output"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/train",
            ),
            ProcessingInput(
                source=test_step.properties.ProcessingOutputConfig.Outputs[
                    "test_output"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
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
            "/opt/ml/processing/preprocess/featurizer/sklearn_model.joblib",
            "--xgboost-model-path",
            "/opt/ml/processing/train/model/xgboost_model.bin",
            "--model-report-path",
            "/opt/ml/processing/test/model_report.json",
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

    return [preprocess_step, train_step, test_step, register_step, deploy_step]


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

    eta_parameter = ParameterFloat(name="eta", default_value=0.1)
    max_depth_parameter = ParameterInteger(name="max_depth", default_value=8)
    min_child_weight_parameter = ParameterFloat(name="min_child_weight", default_value=0.5)
    subsample_parameter = ParameterFloat(name="subsample", default_value=0.9)
    colsample_bytree_parameter = ParameterFloat(name="colsample_bytree", default_value=0.9)
    gamma_parameter = ParameterFloat(name="gamma", default_value=0.0)
    reg_lambda_parameter = ParameterFloat(name="reg_lambda", default_value=1.0)
    reg_alpha_parameter = ParameterFloat(name="reg_alpha", default_value=0.0)
    num_boost_round_parameter = ParameterInteger(name="num_boost_round", default_value=800)
    early_stopping_rounds_parameter = ParameterInteger(name="early_stopping_rounds", default_value=50)
    base_score_parameter = ParameterFloat(name="base_score", default_value=-1.0)
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
        base_score_parameter,
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
            base_score_parameter,
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
