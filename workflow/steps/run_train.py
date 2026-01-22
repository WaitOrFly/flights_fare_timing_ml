from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys

import numpy as np


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


def _load_numpy(input_dir: str, name: str) -> np.ndarray:
    path = os.path.join(input_dir, name)
    return np.load(path)


def main() -> None:
    sys.path.insert(0, "/opt/ml/processing/code")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="/opt/ml/processing/input")
    parser.add_argument("--output-dir", default="/opt/ml/processing/output")
    parser.add_argument("--eta", type=float, required=True)
    parser.add_argument("--max-depth", type=int, required=True)
    parser.add_argument("--min-child-weight", type=float, required=True)
    parser.add_argument("--subsample", type=float, required=True)
    parser.add_argument("--colsample-bytree", type=float, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--reg-lambda", type=float, required=True)
    parser.add_argument("--reg-alpha", type=float, required=True)
    parser.add_argument("--num-boost-round", type=int, required=True)
    parser.add_argument("--early-stopping-rounds", type=int, required=True)
    parser.add_argument("--base-score", type=float, required=True)
    parser.add_argument("--model-artifacts-s3-uri", default="")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    _ensure_runtime_deps()

    import xgboost as xgb

    from train import train

    X_train = _load_numpy(args.input_dir, "X_train.npy")
    y_train = _load_numpy(args.input_dir, "y_train.npy")
    X_val = _load_numpy(args.input_dir, "X_val.npy")
    y_val = _load_numpy(args.input_dir, "y_val.npy")

    booster = train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        eta=args.eta,
        max_depth=args.max_depth,
        min_child_weight=args.min_child_weight,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        gamma=args.gamma,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        base_score=args.base_score,
        model_artifacts_s3_uri=args.model_artifacts_s3_uri,
        experiment_name=args.experiment_name,
        run_id=args.run_id,
    )

    model_dir = os.path.join(args.output_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "xgboost_model.bin")
    booster.save_model(model_path)

    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_path": model_path,
                "x_train_shape": list(X_train.shape),
                "x_val_shape": list(X_val.shape),
            },
            handle,
        )

    print(f"[Train] model saved to {model_path}")


if __name__ == "__main__":
    main()
