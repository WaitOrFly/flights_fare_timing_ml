import json
import os
from typing import Dict


from common import FEATURE_ORDER, features_to_csv_rows, invoke_endpoint


def _get_endpoint_name() -> str:
    endpoint_name = os.environ.get("SAGEMAKER_ENDPOINT_NAME")
    if not endpoint_name:
        raise ValueError("SAGEMAKER_ENDPOINT_NAME environment variable is required")
    return endpoint_name


def _extract_features(event: Dict[str, object]) -> Dict[str, object]:
    features = event.get("features")
    if not isinstance(features, dict):
        raise ValueError("event.features must be an object")
    missing = [key for key in FEATURE_ORDER if key not in features]
    if missing:
        raise ValueError(f"Missing features: {', '.join(missing)}")
    return features


def handler(event, context):
    if isinstance(event, dict) and "body" in event:
        body = event.get("body") or "{}"
        if isinstance(body, str):
            try:
                event = json.loads(body)
            except json.JSONDecodeError as exc:
                raise ValueError("event.body must be valid JSON") from exc

    features = _extract_features(event)
    print(f"[Invoke] features: {features}")
    csv_payload = features_to_csv_rows([features])
    print(f"[Invoke] csv_payload: {csv_payload}")
    preds = invoke_endpoint(_get_endpoint_name(), csv_payload)

    if preds.ndim == 0:
        preds = preds.reshape(1)

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(
            {"prediction": float(preds[0])},
            ensure_ascii=False,
        ),
    }
