from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys


def _ensure_packages(requirements) -> None:
    command = [sys.executable, "-m", "pip", "install"]
    subprocess.check_call([*command, *requirements])


def _is_installed(package_name: str) -> bool:
    try:
        importlib.metadata.version(package_name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def _ensure_runtime_deps() -> None:
    if not _is_installed("sagemaker"):
        _ensure_packages(["sagemaker==2.219.0"])

    _ensure_packages(
        [
            "boto3==1.28.57",
            "botocore==1.31.85",
            "s3transfer==0.7.0",
            "scikit-learn==1.3.2",
            "s3fs",
        ]
    )

    if not _is_installed("xgboost"):
        _ensure_packages(["xgboost==1.7.6"])


_ensure_runtime_deps()

import joblib
import numpy as np


def _save_numpy(output_dir: str, name: str, array: np.ndarray) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    np.save(path, array)
    return path


def main() -> None:
    sys.path.insert(0, "/opt/ml/processing/code")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data-s3-uri", required=True)
    parser.add_argument("--output-data-s3-uri", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir",
        default="/opt/ml/processing/output",
    )
    args = parser.parse_args()

    _ensure_runtime_deps()

    from preprocess import preprocess

    outputs = preprocess(
        input_data_s3_uri=args.input_data_s3_uri,
        output_data_s3_uri=args.output_data_s3_uri,
        experiment_name=args.experiment_name,
        run_name=args.run_id,
    )

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        featurizer_model,
        run_id,
    ) = outputs

    _save_numpy(args.output_dir, "X_train.npy", X_train)
    _save_numpy(args.output_dir, "y_train.npy", y_train)
    _save_numpy(args.output_dir, "X_val.npy", X_val)
    _save_numpy(args.output_dir, "y_val.npy", y_val)
    _save_numpy(args.output_dir, "X_test.npy", X_test)
    _save_numpy(args.output_dir, "y_test.npy", y_test)

    featurizer_dir = os.path.join(args.output_dir, "featurizer")
    os.makedirs(featurizer_dir, exist_ok=True)
    featurizer_path = os.path.join(featurizer_dir, "sklearn_model.joblib")
    joblib.dump(featurizer_model, featurizer_path)

    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_id": run_id,
                "x_train_shape": list(X_train.shape),
                "x_val_shape": list(X_val.shape),
                "x_test_shape": list(X_test.shape),
            },
            handle,
        )

    print(f"[Preprocess] artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
