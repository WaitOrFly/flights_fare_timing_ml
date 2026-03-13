# Wait or Fly?

<img width="1884" height="3054" alt="waitorfly" src="https://github.com/user-attachments/assets/f0c82099-c700-4789-9505-ac82d7e359f5" />
<img width="1920" height="1020" alt="waitorfly pipeline" src="https://github.com/user-attachments/assets/973ec919-913c-4aa4-bd6a-f5ec7912256c" />

# Pipeline Summary (Implementation Notes)

This section describes the current pipeline wiring in `workflow/pipeline.py`, where
Register and Deploy run as ProcessingSteps and the other steps run as FunctionSteps.

## Pipeline Structure

1) Preprocess (FunctionStep: `steps/preprocess.py`)
2) Train (FunctionStep: `steps/train.py`)
3) Test/Evaluate (FunctionStep: `steps/test.py`)
4) PackageModels (FunctionStep: `package_model_artifacts`)
5) Register (ProcessingStep: `steps/register.py`)
6) Deploy (ProcessingStep: `steps/deploy.py`)

## steps/train.py

Inputs (parameters)
- X_train, y_train, X_val, y_val
- eta, max_depth, min_child_weight, subsample, colsample_bytree, gamma
- reg_lambda, reg_alpha, num_boost_round, early_stopping_rounds
- experiment_name, run_id

Output (return)
- xgboost.Booster

Algorithm
- XGBoost regressor (`objective="reg:squarederror"`)

Overfitting controls
- early_stopping_rounds on validation set

Hyperparameters
- eta, max_depth, min_child_weight, subsample, colsample_bytree, gamma
- reg_lambda, reg_alpha, num_boost_round, early_stopping_rounds

Validation metrics
- rmse, mae, r2 (on validation set)

MLflow logging
- Nested run name: "Train"
- Logs params + early_stopping_rounds
- Logs val_rmse/val_mae/val_r2

## steps/test.py

Inputs (parameters)
- featurizer_model, booster, X_test, y_test
- bucket_name, model_package_group_name
- experiment_name, run_id

Output (return)
- S3 URI to evaluation report JSON:
  `s3://{bucket_name}/{model_package_group_name}/evaluation-report/{unique}.json`

Metrics
- rmse, mae, r2 (on test set)

MLflow logging
- Nested run name: "Test"
- Logs test_rmse/test_mae/test_r2

## steps/register.py

Inputs (parameters)
- role_arn
- featurizer_model_path (S3 or local path)
- xgboost_model_path (S3 or local path)
- model_report_json or model_report_path (S3 URI supported)
- bucket_name
- model_package_group_name
- model_approval_status
- experiment_name, run_id
- requirements_path (optional)

Output (return)
- Model package ARN (written to `model_package.json`)

MLflow logging
- Nested run name: "Register"
- Logs model_package_group_name, model_approval_status, evaluation_s3_uri

## steps/deploy.py

Inputs (parameters)
- role_arn
- project_prefix
- model_package_arn
- deploy_model ("true"/"false")
- experiment_name, run_id

Output (return)
- Predictor (only used when run directly; not consumed in pipeline)

MLflow logging
- Nested run name: "Deploy"
- Logs endpoint/model config parameters

## pipeline.py

Inputs (parameters)
- input_data_s3_uri
- output_data_s3_uri
- model_artifacts_s3_uri
- deploy_model
- eta, max_depth, min_child_weight, subsample, colsample_bytree, gamma
- reg_lambda, reg_alpha, num_boost_round, early_stopping_rounds

Output (return)
- Pipeline definition only (no runtime return value)
항공권 가격은 **언제 구매하느냐**에 따라 크게 달라지지만,
사용자는 “지금 사야 할지, 며칠 기다려야 할지”를 판단하기 어렵다.

이 프로젝트는 **과거 항공권 가격의 시간적 변동 패턴을 학습해, 최적의 구매 시점을 예측하고 리포트로 제공하는 서비스**를 목표로 한다.

# 사용한 알고리즘
XGBoost Regressor

# Feature Engineering Specification

## Feature Schema (v1.0-flat)

- **Model Input Type:** `tabular_flat`
- **Target Variable:** `price` (numeric)

### Input Features

| Feature Name                  | Type                         | Encoding | Possible Values / Description                                                                                         |
| ----------------------------- | ---------------------------- | -------- | --------------------------------------------------------------------------------------------------------------------- |
| `purchase_day_of_week`        | categorical                  | one-hot  | 조회(구매)한 요일 (0=월요일, 6=일요일). 요일별 가격 리프레시 및 할인 패턴을 학습.                                     |
| `purchase_time_bucket`        | categorical                  | one-hot  | 조회 시간을 시간대 구간(`dawn`, `morning`, `afternoon`, `night`)으로 변환한 변수. 시간대별 가격 업데이트 패턴을 반영. |
| `days_until_departure`        | numeric                      | number   | 출발일까지 남은 일수. 출발일까지의 거리 정보를 연속값으로 반영.                                                       |
| `is_weekend_departure`        | boolean                      | 0/1      | 출발일이 주말인지 여부. 주말 출발 항공권의 가격 프리미엄 반영.                                                        |
| `is_holiday_season`           | boolean                      | 0/1      | 출발일이 성수기(휴가철, 연말연시 등)에 해당하는지 여부.                                                               |
| `route_hash`                  | categorical_high_cardinality | as-is    | 출발지-도착지 조합을 해시화한 값. 노선별 가격 패턴을 일반화하여 학습.                                                 |
| `stops_count`                 | numeric                      | as-is    | 경유 횟수. 경유 수 증가에 따른 가격 차이를 반영.                                                                      |
| `flight_duration_bucket`      | categorical                  | one-hot  | 비행 시간을 구간화(`short`, `medium`, `long`)한 변수. 장·단거리 항공권 가격 특성을 반영.                              |

### 🎯 Target

| Name    | Type    | Description             |
| ------- | ------- | ----------------------- |
| `price` | numeric | 항공권 가격 (예측 대상) |
