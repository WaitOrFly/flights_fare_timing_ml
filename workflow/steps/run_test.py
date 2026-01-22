from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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


def _load_booster(model_dir: str, xgb_module) -> object:
    model_path = os.path.join(model_dir, "model", "xgboost_model.bin")
    booster = xgb_module.Booster()
    booster.load_model(model_path)
    return booster


def _compute_metrics(preds: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    preds_original = np.expm1(preds)
    y_test_original = np.expm1(y_test)
    return {
        "rmse": float(mean_squared_error(y_test_original, preds_original, squared=False)),
        "mae": float(mean_absolute_error(y_test_original, preds_original)),
        "r2": float(r2_score(y_test_original, preds_original)),
    }


def main() -> None:
    sys.path.insert(0, "/opt/ml/processing/code")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="/opt/ml/processing/input")
    parser.add_argument("--model-dir", default="/opt/ml/processing/model")
    parser.add_argument("--output-dir", default="/opt/ml/processing/output")
    parser.add_argument("--bucket-name", required=True)
    parser.add_argument("--model-package-group-name", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    _ensure_runtime_deps()

    import xgboost as xgb

    from test import test

    X_test = _load_numpy(args.input_dir, "X_test.npy")
    y_test = _load_numpy(args.input_dir, "y_test.npy")
    booster = _load_booster(args.model_dir, xgb)

    preds = booster.predict(xgb.DMatrix(X_test))
    metrics = _compute_metrics(preds, y_test)

    report_s3_uri = None
    try:
        report_s3_uri = test(
            featurizer_model=None,
            booster=booster,
            X_test=X_test,
            y_test=y_test,
            bucket_name=args.bucket_name,
            model_package_group_name=args.model_package_group_name,
            experiment_name=args.experiment_name,
            run_id=args.run_id,
        )
    except Exception as exc:
        print(f"[Test] Warning: test() failed to upload report: {exc}")

    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, "model_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        payload = {"metrics": metrics}
        if report_s3_uri:
            payload["report_s3_uri"] = report_s3_uri
        json.dump(payload, handle)

    print(f"[Test] report saved to {report_path}")


if __name__ == "__main__":
    main()
