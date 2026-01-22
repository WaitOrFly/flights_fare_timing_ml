import datetime as dt
import io
import hashlib
import os
from typing import Dict, Iterable, List, Tuple

import boto3
import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - only used in Lambda runtime
    plt = None

FEATURE_ORDER = [
    "purchase_day_of_week",
    "purchase_time_bucket",
    "days_until_departure",
    "is_weekend_departure",
    "is_holiday_season",
    "route_hash",
    "stops_count",
    "flight_duration_bucket",
]

TIME_BUCKETS = ["dawn", "morning", "afternoon", "night"]
HOLIDAY_MONTHS = {1, 3, 4, 5, 6, 8, 10, 11}

REQUIRED_STATIC_FIELDS = [
    "stops_count",
    "flight_duration_bucket",
]

REQUIRED_ID_FIELDS = [
    "origin",
    "destination",
    "departure_date",
    "arrival_date",
    "stops_count",
]


def parse_date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def validate_user_input(user_input: Dict[str, object]) -> None:
    missing = [key for key in REQUIRED_ID_FIELDS + REQUIRED_STATIC_FIELDS if key not in user_input]
    if missing:
        raise ValueError(f"Missing required user_input fields: {', '.join(missing)}")

    if user_input["flight_duration_bucket"] not in {"short", "medium", "long"}:
        raise ValueError("flight_duration_bucket must be one of short|medium|long")


def build_candidate_features(
    base: Dict[str, object],
    purchase_date: dt.date,
    time_bucket: str,
    departure_date: dt.date,
) -> Dict[str, object]:
    days_until = (departure_date - purchase_date).days
    return {
        "purchase_day_of_week": purchase_date.weekday(),
        "purchase_time_bucket": time_bucket,
        "days_until_departure": days_until,
        "is_weekend_departure": base["is_weekend_departure"],
        "is_holiday_season": base["is_holiday_season"],
        "route_hash": base["route_hash"],
        "stops_count": base["stops_count"],
        "flight_duration_bucket": base["flight_duration_bucket"],
    }


def compute_route_hash(origin: str, destination: str) -> int:
    route_str = f"{origin}_{destination}"
    return int(hashlib.md5(route_str.encode("utf-8")).hexdigest()[:8], 16)


def features_to_csv_rows(rows: Iterable[Dict[str, object]]) -> str:
    lines = []
    for row in rows:
        values = [row[key] for key in FEATURE_ORDER]
        lines.append(",".join(str(value) for value in values))
    return "\n".join(lines)


def invoke_endpoint(endpoint_name: str, csv_payload: str) -> np.ndarray:
    client = boto3.client("sagemaker-runtime")
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/octet-stream",
        Accept="application/x-npy",
        Body=csv_payload.encode("utf-8"),
    )
    body = response["Body"].read()
    return np.load(io.BytesIO(body))


def plot_price_trend(dates: List[str], prices: List[float], output_path: str) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required to generate plots")

    plt.figure(figsize=(10, 4))
    plt.plot(dates, prices, marker="o", linewidth=1)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, format="png", dpi=150)
    plt.close()


def upload_plot(file_path: str) -> str:
    bucket = os.environ.get("PLOT_BUCKET", "").strip()
    prefix = os.environ.get("PLOT_PREFIX", "plots").strip("/")
    public_base_url = os.environ.get("PLOT_PUBLIC_BASE_URL", "").rstrip("/")

    if not bucket:
        raise ValueError("PLOT_BUCKET environment variable is required")

    s3 = boto3.client("s3")
    file_name = os.path.basename(file_path)
    key = f"{prefix}/{file_name}" if prefix else file_name

    with open(file_path, "rb") as handle:
        s3.put_object(Bucket=bucket, Key=key, Body=handle, ContentType="image/png")

    if public_base_url:
        return f"{public_base_url}/{key}"

    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=3600,
    )
