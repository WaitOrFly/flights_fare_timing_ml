"""
Flight Price Preprocessing Module

이 모듈은 항공권 가격 예측 데이터 전처리 파이프라인을 포함합니다.
"""
import os
import joblib

import pandas as pd
import numpy as np
import hashlib
import warnings
import boto3
from io import StringIO

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import mlflow
import mlflow.sklearn

from .feature_engineer import FlightFeatureEngineer, RouteHashEncoder, FlightPricePreprocessor


warnings.filterwarnings('ignore')

def upload_df_to_s3(df: pd.DataFrame, s3_uri: str, filename: str):
    """
    DataFrame을 CSV로 변환하여 S3에 업로드
    """
    if not s3_uri.startswith("s3://"):
        raise ValueError("output_data_s3_uri must start with 's3://'")

    s3_path = s3_uri.replace("s3://", "")
    bucket = s3_path.split("/")[0]
    prefix = "/".join(s3_path.split("/")[1:]).rstrip("/")

    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)

    s3 = boto3.client("s3")
    s3.put_object(
        Bucket=bucket,
        Key=f"{prefix}/{filename}",
        Body=csv_buffer.getvalue()
    )

    print(f"📤 S3 저장 완료: s3://{bucket}/{prefix}/{filename}")

def upload_file_to_s3(local_path: str, s3_uri: str):
    """
    로컬 파일을 S3에 업로드 (joblib, bin 등)
    """
    if not s3_uri.startswith("s3://"):
        raise ValueError("s3_uri must start with 's3://'")

    s3_path = s3_uri.replace("s3://", "")
    bucket = s3_path.split("/")[0]
    key = "/".join(s3_path.split("/")[1:])

    s3 = boto3.client("s3")
    s3.upload_file(local_path, bucket, key)

    print(f"📤 S3 저장 완료: {s3_uri}")

def detect_outliers_iqr(data, column, multiplier=1.5):
    """
    IQR 방법으로 outlier 탐지
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    outlier_mask = (data[column] < lower_bound) | (data[column] > upper_bound)

    return lower_bound, upper_bound, outlier_mask

def time_stratified_split_by_crawl(
    df: pd.DataFrame,
    time_col: str = "Crawl Timestamp",
    target_col: str = "Fare",
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    n_time_blocks: int = 10,
    n_price_bins: int = 5,
    random_state: int = 42,
):
    """
    시간 순서를 유지하면서 target(Fare) 분포를 안정화하는 split
    - train:test:val = 7:2:1
    - 시간 block 단위로 나눈 뒤, block 내부에서 가격 분위수 기반 분할
    - 미래 데이터 누수 없음
    """

    assert train_ratio + val_ratio < 1.0, "train_ratio + val_ratio must be < 1"

    df = df.copy()
    df["crawl_datetime"] = pd.to_datetime(
        df[time_col], utc=True
    ).dt.tz_localize(None)

    # 1️⃣ 시간 기준 정렬
    df = df.sort_values("crawl_datetime").reset_index(drop=True)

    # 2️⃣ 시간 block 생성 (순서 유지)
    df["time_block"] = pd.qcut(
        df.index,
        q=n_time_blocks,
        labels=False,
        duplicates="drop"
    )

    train_parts, val_parts, test_parts = [], [], []

    # 3️⃣ block 단위 분할
    for _, block_df in df.groupby("time_block", sort=False):
        block_df = block_df.copy()

        # block 내부 가격 분위수 bin
        block_df["price_bin"] = pd.qcut(
            block_df[target_col],
            q=min(n_price_bins, block_df[target_col].nunique()),
            duplicates="drop"
        )

        # block 내부 셔플 (시간 block 안에서만)
        block_df = block_df.sample(frac=1, random_state=random_state)

        n = len(block_df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_parts.append(block_df.iloc[:train_end])
        val_parts.append(block_df.iloc[train_end:val_end])
        test_parts.append(block_df.iloc[val_end:])

    # 4️⃣ 최종 합치기 + 시간 순 복원
    df_train = pd.concat(train_parts).sort_values("crawl_datetime").reset_index(drop=True)
    df_val   = pd.concat(val_parts).sort_values("crawl_datetime").reset_index(drop=True)
    df_test  = pd.concat(test_parts).sort_values("crawl_datetime").reset_index(drop=True)

    return df_train, df_val, df_test



def preprocess(
    input_data_s3_uri: str,
    output_data_s3_uri: str,
    experiment_name="main_experiment",
    run_name="run-01"
) -> tuple:

    # Enable autologging in MLflow
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_ARN'])
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        print(run)

        with mlflow.start_run(run_name="DataPreprocessing", nested=True):
            mlflow.autolog()

            print(f"✅ MLflow Run 시작: {run_id}")
            print(f"   Experiment: {experiment_name}")
            print(f"   Run Name: {run_name}")

            # 1. 데이터 로드
            print(f"\n📥 데이터 로드 중: {input_data_s3_uri}")

            # (참고) 아래 s3 get_object 코드는 유지. 현재는 s3fs로 읽는 방식 사용 중.
            if not input_data_s3_uri.startswith('s3://'):
                raise ValueError("input_data_s3_uri must start with 's3://'")

            df_raw = pd.read_csv(input_data_s3_uri)

            print(f"✅ 데이터 로드 완료: {df_raw.shape}")
            mlflow.log_param("raw_data_shape", str(df_raw.shape))
            mlflow.log_param("input_s3_uri", input_data_s3_uri)

            # 2. 중복 제거
            df_raw_before = df_raw.shape[0]
            df_raw = df_raw.drop_duplicates().reset_index(drop=True)
            df_raw_after = df_raw.shape[0]
            
            removed = df_raw_before - df_raw_after

            print(f"\n✅ 중복 제거 완료")
            print(f"  - 제거 전: {df_raw_before:,}개")
            print(f"  - 제거 후: {df_raw_after:,}개")
            print(f"  - 제거된 데이터: {removed:,}개")

            mlflow.log_metric("duplicates_removed", removed)
            mlflow.log_metric("data_after_dedup", df_raw_after)

            # 3. Outlier 처리 (기존 로직 유지)
            OUTLIER_METHOD = 'clip'  # 'remove', 'clip', 'log', 'none' 중 선택
            
            lower, upper, outlier_mask = detect_outliers_iqr(df_raw, 'Fare', multiplier=1.5)
            n_outliers = int(outlier_mask.sum())
            outlier_pct = (n_outliers / len(df_raw)) * 100

            print(f"\n🔍 Outlier 분석 (IQR 방법):")
            print(f"  - Lower bound: ₹{lower:,.0f}")
            print(f"  - Upper bound: ₹{upper:,.0f}")
            print(f"  - Outliers: {n_outliers:,}개 ({outlier_pct:.2f}%)")

            mlflow.log_param("outlier_method", OUTLIER_METHOD)
            mlflow.log_metric("outlier_lower_bound", float(lower))
            mlflow.log_metric("outlier_upper_bound", float(upper))
            mlflow.log_metric("n_outliers", n_outliers)
            mlflow.log_metric("outlier_percentage", float(outlier_pct))

            df_processed = df_raw.copy()

            if OUTLIER_METHOD == 'clip':
                df_processed['Fare'] = df_processed['Fare'].clip(lower=lower, upper=upper)
                print(f"✅ Outlier Clipping 완료")
                print(f"  - {n_outliers:,}개의 값이 경계값으로 대체되었습니다")

            # Route-level stats for inference (mean fare per route).
            route_stats = (
                df_processed.groupby(["Source", "Destination"], as_index=False)["Fare"]
                .mean()
            )
            route_stats["Crawl Timestamp"] = df_processed["Crawl Timestamp"].min()
            upload_df_to_s3(route_stats, output_data_s3_uri, "route_stats.csv")

            # 4. 시간 기반 + 가격 안정화 split (누수 방지)
            print(f"\n📊 (Leak-Free) crawl_timestamp 기준 데이터셋 분할 중...")

            df_train_raw, df_val_raw, df_test_raw = time_stratified_split_by_crawl(
                df_processed,
                train_ratio=0.7,
                val_ratio=0.1
            )

            print(f"✅ 데이터셋 분할 완료 (Train 70% / Validation 10% / Test 20%)")
            print(f"  - Train: {df_train_raw.shape[0]:,}개")
            print(f"  - Validation: {df_val_raw.shape[0]:,}개")
            print(f"  - Test: {df_test_raw.shape[0]:,}개")

            mlflow.log_metric("train_size", int(df_train_raw.shape[0]))
            mlflow.log_metric("val_size", int(df_val_raw.shape[0]))
            mlflow.log_metric("test_size", int(df_test_raw.shape[0]))

            # 5. Feature Engineering (split별 + 안전한 price_trend)
            print(f"\n⚙️ Feature Engineering 시작 (split별, 과거 데이터만 참조)...")
            
            APPLY_LOG_TARGET = True  # 타깃은 항상 log 사용

            engineer = FlightFeatureEngineer(apply_log_to_target=APPLY_LOG_TARGET)

            # Train: train 내부 과거만 사용 (historical_df=None → df 내부 shift(1) 기반)
            train_features = engineer.transform(df_train_raw, historical_df=None)

            # Val: 과거 = train
            val_features = engineer.transform(df_val_raw, historical_df=df_train_raw)

            # Test: 과거 = train + val
            test_features = engineer.transform(df_test_raw, historical_df=pd.concat([df_train_raw, df_val_raw], axis=0))

            # (참고) 최종 dataset 저장용 (schema 동일, 누수 제거된 버전)
            df_features_all = pd.concat([train_features, val_features, test_features], axis=0).reset_index(drop=True)

            # 중복 제거 (기존 의도 유지)
            duplicates_after_fe = int(df_features_all.duplicated().sum())
            if duplicates_after_fe > 0:
                before = df_features_all.shape[0]
                df_features_all = df_features_all.drop_duplicates().reset_index(drop=True)
                after = df_features_all.shape[0]
                removed_fe = before - after
                print(f"  - Feature Engineering 후 중복 제거: {removed_fe:,}개")
                mlflow.log_metric("duplicates_removed_after_fe", int(removed_fe))

            print(f"✅ Feature Engineering 완료")
            print(f"  - Feature 개수: {df_features_all.shape[1]}")

            mlflow.log_param("feature_count", int(df_features_all.shape[1]))
            mlflow.log_param("apply_log_to_target", bool(APPLY_LOG_TARGET))

            # 6. Target과 features 분리 (split별)
            target_col = 'price'

            drop_cols = ['price', 'price_original'] if 'price_original' in train_features.columns else ['price']

            X_train = train_features.drop(drop_cols, axis=1)
            y_train = train_features[target_col]

            X_val = val_features.drop(drop_cols, axis=1)
            y_val = val_features[target_col]

            X_test = test_features.drop(drop_cols, axis=1)
            y_test = test_features[target_col]

            # ✅ 평가용 원본 타깃은 항상 확보
            if "price_original" in val_features.columns:
                y_val_original = val_features["price_original"]
            else:
                # log를 안 쓰는 경우엔 price 자체가 원본
                y_val_original = val_features["price"]

            if "price_original" in test_features.columns:
                y_test_original = test_features["price_original"]
            else:
                y_test_original = test_features["price"]
                
            # ===== CSV 저장용 데이터 구성 & S3 저장 (기존 흐름 유지) =====
            train_df = X_train.copy()
            train_df["price"] = y_train.values

            val_df = X_val.copy()
            val_df["price"] = y_val.values

            test_df = X_test.copy()
            test_df["price"] = y_test.values

            upload_df_to_s3(train_df, output_data_s3_uri, "train.csv")
            upload_df_to_s3(val_df, output_data_s3_uri, "validation.csv")
            upload_df_to_s3(test_df, output_data_s3_uri, "test.csv")

            mlflow.log_param("output_data_s3_uri", output_data_s3_uri)

            # 7. ML Preprocessing (인코딩 & 스케일링) - train만 fit
            print(f"\n🔧 ML Preprocessing 시작...")

            featurizer_model = FlightPricePreprocessor(scale_numeric=True)

            X_train_processed = featurizer_model.fit_transform(X_train)
            X_val_processed = featurizer_model.transform(X_val)
            X_test_processed = featurizer_model.transform(X_test)

            print(f"✅ ML Preprocessing 완료")
            print(f"  - 원본 feature 수: {X_train.shape[1]}")
            print(f"  - 변환 후 feature 수: {X_train_processed.shape[1]}")

            mlflow.log_metric("original_feature_count", int(X_train.shape[1]))
            mlflow.log_metric("transformed_feature_count", int(X_train_processed.shape[1]))

            # 8. Featurizer 모델 저장 (MLflow) - 기존 형태 유지
            print(f"\n💾 Featurizer 모델 저장 중...")
            mlflow.sklearn.log_model(
                featurizer_model,
                artifact_path="featurizer"
            )
            print(f"✅ Featurizer 모델 저장 완료")

            # SageMaker 배포용 Featurizer 모델 저장
            featurizer_s3_uri = f"{output_data_s3_uri}/featurizer/featurizer.joblib"
            local_featurizer_path = "/tmp/featurizer.joblib"

            # 1️⃣ 로컬 저장
            joblib.dump(featurizer_model, local_featurizer_path)

            # 2️⃣ S3 업로드 (공통 유틸 사용)
            upload_file_to_s3(local_featurizer_path, featurizer_s3_uri)

            # 3️⃣ MLflow에는 로컬 파일만 기록
            mlflow.log_artifact(local_featurizer_path, artifact_path="featurizer")


            # ===== 최종 결과 데이터셋 S3 저장 (누수 제거 버전) =====
            upload_df_to_s3(
                df_features_all,
                output_data_s3_uri,
                "final_dataset.csv"
            )

            # 9. 통계 로깅
            mlflow.log_metric("train_mean_log_price", float(y_train.mean()))
            mlflow.log_metric("val_mean_log_price", float(y_val.mean()))
            mlflow.log_metric("test_mean_log_price", float(y_test.mean()))

            mlflow.log_metric(
                "train_mean_price_original",
                float(train_features["price_original"].mean())
            )
            
            # 10. 완료 요약
            print(f"\n" + "=" * 70)
            print(f"전처리 파이프라인 완료 (Leak-Free)")
            print(f"=" * 70)
            print(f"\n📊 최종 데이터셋:")
            print(f"  - Train: {X_train_processed.shape}")
            print(f"  - Validation: {X_val_processed.shape}")
            print(f"  - Test: {X_test_processed.shape}")
            print(f"💰 Target 통계 (log-space):")
            print(f"  - Train 평균(log): {y_train.mean():.4f}")
            print(f"  - Validation 평균(log): {y_val.mean():.4f}")
            print(f"  - Test 평균(log): {y_test.mean():.4f}")
            print(f"\n💰 Target 통계 (INR, original scale):")
            print(f"  - Train 평균: ₹{train_features['price_original'].mean():,.2f}")
            print(f"  - Validation 평균: ₹{val_features['price_original'].mean():,.2f}")
            print(f"  - Test 평균: ₹{test_features['price_original'].mean():,.2f}")
            print(f"\n✅ MLflow Run ID: {run_id}")

    return (
        X_train_processed,
        y_train.values,
        X_val_processed,
        y_val.values,
        y_val_original.values,
        X_test_processed,
        y_test.values,
        y_test_original.values,
        run_id
    )
