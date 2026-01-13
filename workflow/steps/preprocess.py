"""
Flight Price Preprocessing Module

이 모듈은 항공권 가격 예측 데이터 전처리 파이프라인을 포함합니다.
"""
import os
import joblib

import pandas as pd
import numpy as np
from datetime import datetime
import hashlib
import re
import warnings
from scipy import stats
import boto3
from io import StringIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import mlflow
import mlflow.sklearn

warnings.filterwarnings('ignore')


class FlightFeatureEngineer:
    """
    원본 데이터를 feature schema에 정의된 feature로 변환
    """
    
    def __init__(self, apply_log_to_target=False):
        # 인도 공휴일 시즌 정의
        self.holiday_months = [1, 3, 4, 5, 6, 8, 10, 11]  # Republic Day, Holi, 여름휴가, Independence Day, Diwali/Dussehra
        self.apply_log_to_target = apply_log_to_target
        self.ordinal_mapping = {
            'very_close': 0,
            'close': 1,
            'medium': 2,
            'far': 3,
        }
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        원본 데이터를 feature schema 형식으로 변환
        """
        df = df.copy()

        # 숫자형 컬럼 정제
        if 'Fare' in df.columns:
            df['Fare'] = pd.to_numeric(df['Fare'], errors='coerce').fillna(0)
        if 'Number Of Stops' in df.columns:
            df['Number Of Stops'] = df['Number Of Stops'].apply(self._parse_stops)
        
        # 날짜/시간 파싱
        df['crawl_datetime'] = pd.to_datetime(df['Crawl Timestamp'], utc=True).dt.tz_localize(None)
        df['departure_datetime'] = pd.to_datetime(df['Departure Date'] + ' ' + df['Departure Time'])
        
        features = pd.DataFrame()
        
        # 1. purchase_day_of_week: 구매(크롤링) 시점의 요일
        features['purchase_day_of_week'] = df['crawl_datetime'].dt.dayofweek
        
        # 2. purchase_time_bucket: 구매 시간대
        features['purchase_time_bucket'] = df['crawl_datetime'].dt.hour.apply(
            self._get_time_bucket
        )
        
        # 3. days_until_departure_bucket: 출발까지 남은 일수
        days_until = (df['departure_datetime'] - df['crawl_datetime']).dt.days
        days_until_bucket = days_until.apply(self._get_days_until_bucket)
        features['days_until_departure_bucket'] = days_until_bucket.map(
            self.ordinal_mapping
        )
        
        # 4. is_weekend_departure: 주말 출발 여부
        features['is_weekend_departure'] = (
            df['departure_datetime'].dt.dayofweek >= 5
        ).astype(int)
        
        # 5. is_holiday_season: 휴가 성수기 여부
        features['is_holiday_season'] = (
            df['departure_datetime'].dt.month.isin(self.holiday_months)
        ).astype(int)
        
        
        
        # 8. route_hash: 출발지-목적지 해시
        features['route_hash'] = df.apply(
            lambda row: self._hash_route(row['Source'], row['Destination']),
            axis=1
        )
        
        # 9. stops_count: 경유 횟수
        features['stops_count'] = df['Number Of Stops']
        
        # 10. flight_duration_bucket: 비행 시간 구간
        total_minutes = df['Total Time'].apply(self._parse_duration)
        features['flight_duration_bucket'] = total_minutes.apply(
            self._get_duration_bucket
        )
        
        # Target: price (log transform)
        features['price'] = np.log1p(df['Fare'])
        features['price_original'] = df['Fare']  # ?? ??
        
        return features
    
    def _get_time_bucket(self, hour: int) -> str:
        """시간을 bucket으로 변환"""
        if 0 <= hour < 6:
            return 'dawn'
        elif 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 18:
            return 'afternoon'
        else:
            return 'night'
    
    def _get_days_until_bucket(self, days: int) -> str:
        """출발까지 남은 일수를 bucket으로 변환"""
        if days < 7:
            return 'very_close'
        elif days < 14:
            return 'close'
        elif days < 30:
            return 'medium'
        else:
            return 'far'
    
    def _hash_route(self, source: str, destination: str) -> int:
        """출발지-목적지를 해시값으로 변환"""
        route_str = f"{source}_{destination}"
        return int(hashlib.md5(route_str.encode()).hexdigest()[:8], 16)

    def _parse_stops(self, value) -> int:
        """경유 횟수를 정수로 변환"""
        if pd.isna(value):
            return 0
        if isinstance(value, (int, np.integer)):
            return int(value)
        text = str(value).strip().lower()
        if "non" in text:
            return 0
        match = re.search(r"\d+", text)
        return int(match.group(0)) if match else 0
    
    def _parse_duration(self, duration_str: str) -> int:
        """
        비행 시간 문자열을 분으로 변환
        예: "2h 30m" -> 150
        """
        try:
            if pd.isna(duration_str):
                return 0
            
            hours = 0
            minutes = 0
            
            if 'h' in str(duration_str):
                parts = str(duration_str).split('h')
                hours = int(parts[0].strip())
                if len(parts) > 1 and 'm' in parts[1]:
                    minutes = int(parts[1].replace('m', '').strip())
            elif 'm' in str(duration_str):
                minutes = int(str(duration_str).replace('m', '').strip())
            
            return hours * 60 + minutes
        except:
            return 0
    
    def _get_duration_bucket(self, minutes: int) -> str:
        """비행 시간을 bucket으로 변환"""
        if minutes < 120:  # 2시간 미만
            return 'short'
        elif minutes < 360:  # 6시간 미만
            return 'medium'
        else:
            return 'long'
    
    def _calculate_price_trend(self, df: pd.DataFrame) -> pd.Series:
        """가격 추세 계산 (route별 평균 대비 변화율)"""
        df['route'] = df['Source'] + '_' + df['Destination']
        route_avg = df.groupby('route')['Fare'].transform('mean')
        trend = (df['Fare'] - route_avg) / route_avg
        return trend.fillna(0)
    
    def _calculate_price_ratio(self, df: pd.DataFrame) -> pd.Series:
        """현재 가격 vs 평균 가격 비율"""
        df['route'] = df['Source'] + '_' + df['Destination']
        route_avg = df.groupby('route')['Fare'].transform('mean')
        ratio = df['Fare'] / route_avg
        return ratio.fillna(1.0)


class FlightPricePreprocessor:
    """
    Feature schema 기반 전처리 클래스
    - Categorical: One-hot encoding
    - Ordinal: Label encoding
    - Numeric: Standardization (선택)
    - Boolean: 0/1 (그대로)
    """
    
    def __init__(self, scale_numeric: bool = True):
        self.scale_numeric = scale_numeric
        self.preprocessor = None
        self.ordinal_mapping = {
            'very_close': 0,
            'close': 1,
            'medium': 2,
            'far': 3
        }
        self._setup_preprocessor()
        
    def _setup_preprocessor(self):
        """전처리 파이프라인 설정"""
        
        # Feature 그룹 정의
        categorical_onehot_features = [
            'purchase_day_of_week',
            'purchase_time_bucket', 
            'flight_duration_bucket'
        ]
        
        boolean_features = [
            'is_weekend_departure',
            'is_holiday_season'
        ]
        
        numeric_features = [
            'stops_count',
            'days_until_departure_bucket'
        ]
        
        high_cardinality_features = ['route_hash']
        
        # ColumnTransformer 구성
        transformers = []
        
        # 1. Categorical (one-hot)
        transformers.append((
            'cat_onehot',
            OneHotEncoder(
                drop='first',
                sparse_output=False,
                handle_unknown='ignore'
            ),
            categorical_onehot_features
        ))
        
        # 2. Numeric features
        if self.scale_numeric:
            transformers.append((
                'num',
                StandardScaler(),
                numeric_features
            ))
        else:
            transformers.append((
                'num',
                'passthrough',
                numeric_features
            ))
        
        # 3. Boolean features (그대로)
        transformers.append((
            'bool',
            'passthrough',
            boolean_features
        ))
        
        # 4. High cardinality (그대로)
        transformers.append((
            'high_card',
            'passthrough',
            high_cardinality_features
        ))
        
        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )
        
    def _encode_ordinal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ordinal feature 인코딩"""
        df = df.copy()
        if 'days_until_departure_bucket' in df.columns:
            if df['days_until_departure_bucket'].dtype == object:
                df['days_until_departure_bucket'] = df['days_until_departure_bucket'].map(
                    self.ordinal_mapping
                )
        return df
    
    def fit(self, X: pd.DataFrame, y=None):
        """전처리기를 학습 데이터에 fit"""
        X_processed = self._encode_ordinal_features(X)
        self.preprocessor.fit(X_processed)
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """데이터를 전처리하여 변환"""
        X_processed = self._encode_ordinal_features(X)
        X_transformed = self.preprocessor.transform(X_processed)
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        """fit과 transform을 한 번에 수행"""
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self):
        """변환 후 feature 이름 반환"""
        try:
            return list(self.preprocessor.get_feature_names_out())
        except AttributeError:
            # Fallback for older scikit-learn versions
            return ['feature_' + str(i) for i in range(self.preprocessor.n_features_in_)]

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

def detect_outliers_iqr(data, column, multiplier=1.5):
    """
    IQR 방법으로 outlier 탐지
    
    Args:
        data: DataFrame
        column: 컬럼명
        multiplier: IQR 배수 (기본 1.5, 더 엄격하게는 3.0)
    
    Returns:
        lower_bound, upper_bound, outlier_mask
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outlier_mask = (data[column] < lower_bound) | (data[column] > upper_bound)
    
    return lower_bound, upper_bound, outlier_mask


def preprocess(input_data_s3_uri: str, output_data_s3_uri: str, experiment_name="main_experiment", run_name="run-01") -> tuple:
    """
    항공권 가격 예측 데이터 전처리 함수
    
    Args:
        input_data_s3_uri: S3에 저장된 원본 데이터 경로 (s3://bucket/path/to/file.csv)
        experiment_name: MLflow experiment 이름
        run_name: MLflow run 이름
    
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, featurizer_model, run_id
    """
    
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
            
            # 1. 데이터 로드 from S3
            print(f"\n📥 데이터 로드 중: {input_data_s3_uri}")
            
            # S3에서 데이터 읽기
            s3 = boto3.client('s3')
            
            # S3 URI 파싱
            if input_data_s3_uri.startswith('s3://'):
                s3_path = input_data_s3_uri.replace('s3://', '')
                bucket_name = s3_path.split('/')[0]
                object_key = '/'.join(s3_path.split('/')[1:])
            else:
                raise ValueError("input_data_s3_uri must start with 's3://'")
            
            # S3에서 파일 읽기
            # obj = s3.get_object(Bucket=bucket_name, Key=object_key)
            # df_raw = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
            df_raw = pd.read_csv(input_data_s3_uri)
            
            print(f"✅ 데이터 로드 완료: {df_raw.shape}")
            mlflow.log_param("raw_data_shape", str(df_raw.shape))
            mlflow.log_param("input_s3_uri", input_data_s3_uri)
            
            # 2. 중복 데이터 제거
            df_raw_before = df_raw.shape[0]
            df_raw = df_raw.drop_duplicates()
            df_raw = df_raw.reset_index(drop=True)
            df_raw_after = df_raw.shape[0]
            removed = df_raw_before - df_raw_after
            
            print(f"\n✅ 중복 제거 완료")
            print(f"  - 제거 전: {df_raw_before:,}개")
            print(f"  - 제거 후: {df_raw_after:,}개")
            print(f"  - 제거된 데이터: {removed:,}개")
            
            mlflow.log_metric("duplicates_removed", removed)
            mlflow.log_metric("data_after_dedup", df_raw_after)
            
            # 3. Outlier 처리 (Clipping 방법 사용)
            OUTLIER_METHOD = 'clip'  # 'remove', 'clip', 'log', 'none' 중 선택
            
            lower, upper, outlier_mask = detect_outliers_iqr(df_raw, 'Fare', multiplier=1.5)
            n_outliers = outlier_mask.sum()
            outlier_pct = (n_outliers / len(df_raw)) * 100
            
            print(f"\n🔍 Outlier 분석 (IQR 방법):")
            print(f"  - Lower bound: ₹{lower:,.0f}")
            print(f"  - Upper bound: ₹{upper:,.0f}")
            print(f"  - Outliers: {n_outliers:,}개 ({outlier_pct:.2f}%)")
            
            mlflow.log_param("outlier_method", OUTLIER_METHOD)
            mlflow.log_metric("outlier_lower_bound", lower)
            mlflow.log_metric("outlier_upper_bound", upper)
            mlflow.log_metric("n_outliers", n_outliers)
            mlflow.log_metric("outlier_percentage", outlier_pct)
            
            df_processed = df_raw.copy()
            
            if OUTLIER_METHOD == 'clip':
                # Outlier Clipping
                df_processed['Fare'] = df_processed['Fare'].clip(lower=lower, upper=upper)
                print(f"✅ Outlier Clipping 완료")
                print(f"  - {n_outliers:,}개의 값이 경계값으로 대체되었습니다")
            # 4. Train/Validation/Test Split (70/10/20)
            print("Split dataset (Train/Val/Test) ...")

            df_train_raw, df_temp_raw = train_test_split(
                df_processed, test_size=0.3, random_state=42
            )
            df_val_raw, df_test_raw = train_test_split(
                df_temp_raw, test_size=2/3, random_state=42
            )

            print(f"???????????? ?????? ????? (Train 70% / Validation 10% / Test 20%)")
            print(f"  - Train: {df_train_raw.shape[0]:,}???")
            print(f"  - Validation: {df_val_raw.shape[0]:,}???")
            print(f"  - Test: {df_test_raw.shape[0]:,}???")

            mlflow.log_metric("train_size", df_train_raw.shape[0])
            mlflow.log_metric("val_size", df_val_raw.shape[0])
            mlflow.log_metric("test_size", df_test_raw.shape[0])

            # 5. Feature Engineering
            print("Feature engineering...")

            apply_log = True
            engineer = FlightFeatureEngineer(apply_log_to_target=apply_log)
            df_train_features = engineer.transform(df_train_raw)
            df_val_features = engineer.transform(df_val_raw)
            df_test_features = engineer.transform(df_test_raw)

            def _dedup_features(df_features, label):
                duplicates = df_features.duplicated().sum()
                if duplicates > 0:
                    df_before = df_features.shape[0]
                    df_features = df_features.drop_duplicates().reset_index(drop=True)
                    removed = df_before - df_features.shape[0]
                    print(f"  - Feature Engineering {label} ???????? ?????: {removed:,}???")
                    mlflow.log_metric(f"duplicates_removed_after_fe_{label}", removed)
                return df_features

            df_train_features = _dedup_features(df_train_features, "train")
            df_val_features = _dedup_features(df_val_features, "val")
            df_test_features = _dedup_features(df_test_features, "test")

            df_features = pd.concat(
                [df_train_features, df_val_features, df_test_features],
                ignore_index=True,
            )

            print(f"??Feature Engineering ?????")
            print(f"  - Feature ??????: {df_train_features.shape[1]}")

            mlflow.log_param("feature_count", df_train_features.shape[1])
            mlflow.log_param("apply_log_to_target", apply_log)

            target_col = 'price'
            drop_cols = [col for col in ['price', 'price_original'] if col in df_train_features.columns]

            X_train = df_train_features.drop(columns=drop_cols)
            y_train = df_train_features[target_col]
            X_val = df_val_features.drop(columns=drop_cols)
            y_val = df_val_features[target_col]
            X_test = df_test_features.drop(columns=drop_cols)
            y_test = df_test_features[target_col]

            # ===== CSV ??????? ????????????? =====
            train_df = X_train.copy()
            train_df["price"] = y_train.values

            val_df = X_val.copy()
            val_df["price"] = y_val.values

            test_df = X_test.copy()
            test_df["price"] = y_test.values

            # ===== S3 ????=====
            upload_df_to_s3(train_df, output_data_s3_uri, "train.csv")
            upload_df_to_s3(val_df, output_data_s3_uri, "validation.csv")
            upload_df_to_s3(test_df, output_data_s3_uri, "test.csv")

            mlflow.log_param("output_data_s3_uri", output_data_s3_uri)

            # 6. ML Preprocessing
            print("Processing...")
            print(f"\n🔧 ML Preprocessing 시작...")
            
            featurizer_model = FlightPricePreprocessor(scale_numeric=True)
            
            # Train 데이터로 fit & transform
            X_train_processed = featurizer_model.fit_transform(X_train)
            
            # Validation & Test 데이터 transform
            X_val_processed = featurizer_model.transform(X_val)
            X_test_processed = featurizer_model.transform(X_test)
            
            print(f"✅ ML Preprocessing 완료")
            print(f"  - 원본 feature 수: {X_train.shape[1]}")
            print(f"  - 변환 후 feature 수: {X_train_processed.shape[1]}")
            
            mlflow.log_metric("original_feature_count", X_train.shape[1])
            mlflow.log_metric("transformed_feature_count", X_train_processed.shape[1])
            
            # 7. Featurizer 모델 저장 (MLflow)
            print(f"\n💾 Featurizer 모델 저장 중...")
            safe_name = "".join(
                ch if ch.isalnum() else "-" for ch in f"{experiment_name}-featurizer"
            ).strip("-")
            safe_name = safe_name[:57]
            featurizer_transformer = featurizer_model.preprocessor
            mlflow.sklearn.log_model(
                featurizer_transformer,
                "featurizer",
                registered_model_name=safe_name
            )
            print(f"✅ Featurizer 모델 저장 완료")

            # ===============================
            # SageMaker 배포용 Featurizer 모델 저장
            # ===============================
            model_file_path = "/opt/ml/model/sklearn_model.joblib"
            os.makedirs(os.path.dirname(model_file_path), exist_ok=True)

            joblib.dump(featurizer_transformer, model_file_path)
            mlflow.log_artifact(model_file_path, artifact_path="model")

            # ===== 최종 결과 데이터셋 S3 저장 =====
            upload_df_to_s3(
                df_features,
                output_data_s3_uri,
                "final_dataset.csv"
            )

            # 8. 통계 로깅
            mlflow.log_metric("train_mean_price", float(y_train.mean()))
            mlflow.log_metric("val_mean_price", float(y_val.mean()))
            mlflow.log_metric("test_mean_price", float(y_test.mean()))
            mlflow.log_metric("train_std_price", float(y_train.std()))
            
            # 9. 완료 요약
            print(f"\n" + "="*70)
            print(f"전처리 파이프라인 완료")
            print(f"="*70)
            print(f"\n📊 최종 데이터셋:")
            print(f"  - Train: {X_train_processed.shape}")
            print(f"  - Validation: {X_val_processed.shape}")
            print(f"  - Test: {X_test_processed.shape}")
            print(f"\n💰 Target 통계:")
            print(f"  - Train 평균: ₹{y_train.mean():,.2f}")
            print(f"  - Validation 평균: ₹{y_val.mean():,.2f}")
            print(f"  - Test 평균: ₹{y_test.mean():,.2f}")
            print(f"\n✅ MLflow Run ID: {run_id}")
            
    # numpy array로 변환 및 반환
    return (
        X_train_processed,
        y_train.values,
        X_val_processed,
        y_val.values,
        X_test_processed,
        y_test.values,
        featurizer_transformer,
        run_id
    )

