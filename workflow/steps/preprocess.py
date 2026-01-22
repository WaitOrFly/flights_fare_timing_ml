"""
Flight Price Preprocessing Module.
"""
from __future__ import annotations

import hashlib
import os
import re
import warnings
from datetime import datetime
from io import StringIO
from typing import Tuple

import boto3
import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


class FlightFeatureEngineer:
    """Convert raw data into the feature schema."""

    def __init__(self, apply_log_to_target: bool = False) -> None:
        self.holiday_months = [1, 3, 4, 5, 6, 8, 10, 11]
        self.apply_log_to_target = apply_log_to_target
        self.ordinal_mapping = {
            "very_close": 0,
            "close": 1,
            "medium": 2,
            "far": 3,
        }

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw data into feature schema."""
        df = df.copy()

        if "Fare" in df.columns:
            df["Fare"] = pd.to_numeric(df["Fare"], errors="coerce").fillna(0)
        if "Number Of Stops" in df.columns:
            df["Number Of Stops"] = df["Number Of Stops"].apply(self._parse_stops)

        df["crawl_datetime"] = (
            pd.to_datetime(df["Crawl Timestamp"], utc=True)
            .dt.tz_localize(None)
        )
        df["departure_datetime"] = pd.to_datetime(
            df["Departure Date"] + " " + df["Departure Time"]
        )

        features = pd.DataFrame()

        features["purchase_day_of_week"] = df["crawl_datetime"].dt.dayofweek
        features["purchase_time_bucket"] = df["crawl_datetime"].dt.hour.apply(
            self._get_time_bucket
        )

        days_until = (df["departure_datetime"] - df["crawl_datetime"]).dt.days
        features["days_until_departure"] = days_until

        features["is_weekend_departure"] = (
            df["departure_datetime"].dt.dayofweek >= 5
        ).astype(int)

        features["is_holiday_season"] = (
            df["departure_datetime"].dt.month.isin(self.holiday_months)
        ).astype(int)

        features["route_hash"] = df.apply(
            lambda row: self._hash_route(row["Source"], row["Destination"]),
            axis=1,
        )

        features["stops_count"] = df["Number Of Stops"]

        total_minutes = df["Total Time"].apply(self._parse_duration)
        features["flight_duration_bucket"] = total_minutes.apply(
            self._get_duration_bucket
        )

        features["price"] = np.log1p(df["Fare"])
        features["price_original"] = df["Fare"]

        return features

    def _get_time_bucket(self, hour: int) -> str:
        if 0 <= hour < 6:
            return "dawn"
        if 6 <= hour < 12:
            return "morning"
        if 12 <= hour < 18:
            return "afternoon"
        return "night"

    def _hash_route(self, source: str, destination: str) -> int:
        route_str = f"{source}_{destination}"
        return int(hashlib.md5(route_str.encode()).hexdigest()[:8], 16)

    def _parse_stops(self, value) -> int:
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
        try:
            if pd.isna(duration_str):
                return 0

            hours = 0
            minutes = 0

            if "h" in str(duration_str):
                parts = str(duration_str).split("h")
                hours = int(parts[0].strip())
                if len(parts) > 1 and "m" in parts[1]:
                    minutes = int(parts[1].replace("m", "").strip())
            elif "m" in str(duration_str):
                minutes = int(str(duration_str).replace("m", "").strip())

            return hours * 60 + minutes
        except Exception:
            return 0

    def _get_duration_bucket(self, minutes: int) -> str:
        if minutes < 120:
            return "short"
        if minutes < 360:
            return "medium"
        return "long"

    def _calculate_price_trend(self, df: pd.DataFrame) -> pd.Series:
        df["route"] = df["Source"] + "_" + df["Destination"]
        route_avg = df.groupby("route")["Fare"].transform("mean")
        trend = (df["Fare"] - route_avg) / route_avg
        return trend.fillna(0)

    def _calculate_price_ratio(self, df: pd.DataFrame) -> pd.Series:
        df["route"] = df["Source"] + "_" + df["Destination"]
        route_avg = df.groupby("route")["Fare"].transform("mean")
        ratio = df["Fare"] / route_avg
        return ratio.fillna(1.0)


class FlightPricePreprocessor:
    """Preprocessing pipeline for model features."""

    def __init__(self, scale_numeric: bool = True) -> None:
        self.scale_numeric = scale_numeric
        self.preprocessor = None
        self.ordinal_mapping = {
            "very_close": 0,
            "close": 1,
            "medium": 2,
            "far": 3,
        }
        self._setup_preprocessor()

    def _setup_preprocessor(self) -> None:
        categorical_onehot_features = [
            "purchase_day_of_week",
            "purchase_time_bucket",
            "flight_duration_bucket",
        ]
        boolean_features = ["is_weekend_departure", "is_holiday_season"]
        numeric_features = ["stops_count", "days_until_departure"]
        high_cardinality_features = ["route_hash"]

        transformers = []

        transformers.append(
            (
                "cat_onehot",
                OneHotEncoder(
                    drop="first",
                    sparse_output=False,
                    handle_unknown="ignore",
                ),
                categorical_onehot_features,
            )
        )

        if self.scale_numeric:
            transformers.append(("num", StandardScaler(), numeric_features))
        else:
            transformers.append(("num", "passthrough", numeric_features))

        transformers.append(("bool", "passthrough", boolean_features))
        transformers.append(("high_card", "passthrough", high_cardinality_features))

        self.preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    def _encode_ordinal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "days_until_departure" in df.columns:
            df["days_until_departure"] = pd.to_numeric(
                df["days_until_departure"], errors="coerce"
            )
        return df

    def fit(self, X: pd.DataFrame, y=None):
        X_processed = self._encode_ordinal_features(X)
        self.preprocessor.fit(X_processed)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_processed = self._encode_ordinal_features(X)
        return self.preprocessor.transform(X_processed)

    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self):
        try:
            return list(self.preprocessor.get_feature_names_out())
        except AttributeError:
            return ["feature_" + str(i) for i in range(self.preprocessor.n_features_in_)]


def upload_df_to_s3(df: pd.DataFrame, s3_uri: str, filename: str) -> None:
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
        Body=csv_buffer.getvalue(),
    )

    print(f"S3 upload complete: s3://{bucket}/{prefix}/{filename}")


def upload_file_to_s3(file_path: str, s3_uri: str, filename: str) -> None:
    if not s3_uri.startswith("s3://"):
        raise ValueError("output_data_s3_uri must start with 's3://'")

    s3_path = s3_uri.replace("s3://", "")
    bucket = s3_path.split("/")[0]
    prefix = "/".join(s3_path.split("/")[1:]).rstrip("/")
    key = f"{prefix}/{filename}" if prefix else filename

    s3 = boto3.client("s3")
    with open(file_path, "rb") as handle:
        s3.put_object(Bucket=bucket, Key=key, Body=handle)

    print(f"S3 upload complete: s3://{bucket}/{key}")


def detect_outliers_iqr(data: pd.DataFrame, column: str, multiplier: float = 1.5):
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    outlier_mask = (data[column] < lower_bound) | (data[column] > upper_bound)

    return lower_bound, upper_bound, outlier_mask


def preprocess(
    input_data_s3_uri: str,
    output_data_s3_uri: str,
    experiment_name: str = "main_experiment",
    run_name: str = "run-01",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, ColumnTransformer, str]:
    """Run preprocessing pipeline and return processed arrays."""
    run_id = run_name
    print(f"Run ID: {run_id}")

    print(f"\nLoading data: {input_data_s3_uri}")

    s3 = boto3.client("s3")

    if input_data_s3_uri.startswith("s3://"):
        s3_path = input_data_s3_uri.replace("s3://", "")
        bucket_name = s3_path.split("/")[0]
        object_key = "/".join(s3_path.split("/")[1:])
    else:
        raise ValueError("input_data_s3_uri must start with 's3://'")

    df_raw = pd.read_csv(input_data_s3_uri)

    print(f"Data loaded: {df_raw.shape}")

    df_raw_before = df_raw.shape[0]
    df_raw = df_raw.drop_duplicates().reset_index(drop=True)
    df_raw_after = df_raw.shape[0]
    removed = df_raw_before - df_raw_after

    print("\nDedup complete")
    print(f"  - Before: {df_raw_before:,}")
    print(f"  - After: {df_raw_after:,}")
    print(f"  - Removed: {removed:,}")

    outlier_method = "clip"
    lower, upper, outlier_mask = detect_outliers_iqr(df_raw, "Fare", multiplier=1.5)
    n_outliers = outlier_mask.sum()
    outlier_pct = (n_outliers / len(df_raw)) * 100

    print("\nOutlier analysis (IQR)")
    print(f"  - Lower bound: {lower:,.0f}")
    print(f"  - Upper bound: {upper:,.0f}")
    print(f"  - Outliers: {n_outliers:,} ({outlier_pct:.2f}%)")

    df_processed = df_raw.copy()
    if outlier_method == "clip":
        df_processed["Fare"] = df_processed["Fare"].clip(lower=lower, upper=upper)
        print("Outlier clipping applied")

    print("Split dataset (Train/Val/Test) ...")
    df_train_raw, df_temp_raw = train_test_split(
        df_processed, test_size=0.3, random_state=42
    )
    df_val_raw, df_test_raw = train_test_split(
        df_temp_raw, test_size=2 / 3, random_state=42
    )

    print("Split sizes (Train 70% / Validation 10% / Test 20%)")
    print(f"  - Train: {df_train_raw.shape[0]:,}")
    print(f"  - Validation: {df_val_raw.shape[0]:,}")
    print(f"  - Test: {df_test_raw.shape[0]:,}")

    print("Feature engineering...")

    apply_log = True
    engineer = FlightFeatureEngineer(apply_log_to_target=apply_log)
    df_train_features = engineer.transform(df_train_raw)
    df_val_features = engineer.transform(df_val_raw)
    df_test_features = engineer.transform(df_test_raw)

    def _dedup_features(df_features: pd.DataFrame, label: str) -> pd.DataFrame:
        duplicates = df_features.duplicated().sum()
        if duplicates > 0:
            df_before = df_features.shape[0]
            df_features = df_features.drop_duplicates().reset_index(drop=True)
            removed_local = df_before - df_features.shape[0]
            print(f"  - Feature dedup {label}: {removed_local:,}")
        return df_features

    df_train_features = _dedup_features(df_train_features, "train")
    df_val_features = _dedup_features(df_val_features, "val")
    df_test_features = _dedup_features(df_test_features, "test")

    df_features = pd.concat(
        [df_train_features, df_val_features, df_test_features],
        ignore_index=True,
    )

    print("Feature engineering done")
    print(f"  - Feature count: {df_train_features.shape[1]}")

    target_col = "price"
    drop_cols = [col for col in ["price", "price_original"] if col in df_train_features.columns]

    X_train = df_train_features.drop(columns=drop_cols)
    y_train = df_train_features[target_col]
    X_val = df_val_features.drop(columns=drop_cols)
    y_val = df_val_features[target_col]
    X_test = df_test_features.drop(columns=drop_cols)
    y_test = df_test_features[target_col]

    train_df = X_train.copy()
    train_df["price"] = y_train.values

    val_df = X_val.copy()
    val_df["price"] = y_val.values

    test_df = X_test.copy()
    test_df["price"] = y_test.values

    upload_df_to_s3(train_df, output_data_s3_uri, "train.csv")
    upload_df_to_s3(val_df, output_data_s3_uri, "validation.csv")
    upload_df_to_s3(test_df, output_data_s3_uri, "test.csv")

    print("Processing...")
    print("ML preprocessing start...")

    featurizer_model = FlightPricePreprocessor(scale_numeric=True)

    X_train_processed = featurizer_model.fit_transform(X_train)
    X_val_processed = featurizer_model.transform(X_val)
    X_test_processed = featurizer_model.transform(X_test)

    print("ML preprocessing done")
    print(f"  - Raw feature count: {X_train.shape[1]}")
    print(f"  - Transformed feature count: {X_train_processed.shape[1]}")

    print("Saving featurizer model...")
    featurizer_transformer = featurizer_model.preprocessor
    print("Featurizer model saved.")

    model_file_path = "/opt/ml/model/sklearn_model.joblib"
    os.makedirs(os.path.dirname(model_file_path), exist_ok=True)

    joblib.dump(featurizer_transformer, model_file_path)
    upload_file_to_s3(
        model_file_path,
        output_data_s3_uri,
        "artifacts/sklearn_model.joblib",
    )

    upload_df_to_s3(
        df_features,
        output_data_s3_uri,
        "final_dataset.csv",
    )

    print("\n" + "=" * 70)
    print("Preprocessing pipeline complete")
    print("=" * 70)
    print("\nFinal dataset shapes:")
    print(f"  - Train: {X_train_processed.shape}")
    print(f"  - Validation: {X_val_processed.shape}")
    print(f"  - Test: {X_test_processed.shape}")
    print("\nTarget summary:")
    print(f"  - Train mean: {y_train.mean():,.2f}")
    print(f"  - Validation mean: {y_val.mean():,.2f}")
    print(f"  - Test mean: {y_test.mean():,.2f}")

    return (
        X_train_processed,
        y_train.values,
        X_val_processed,
        y_val.values,
        X_test_processed,
        y_test.values,
        featurizer_transformer,
        run_id,
    )
