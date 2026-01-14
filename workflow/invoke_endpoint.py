import json
import os
from datetime import datetime, timedelta
import math
from typing import Any, Dict, List, Tuple

import boto3


_RUNTIME = boto3.client("sagemaker-runtime")
_SM = boto3.client("sagemaker")

_USER_REQUIRED = [
    "origin",
    "destination",
    "departure_date",
    "arrival_date",
    "stops_count",
    "departure_time",
    "total_time",
    "fare",
]

_MODEL_COLUMNS = [
    "Source",
    "Destination",
    "Departure Date",
    "Departure Time",
    "Crawl Timestamp",
    "Number Of Stops",
    "Total Time",
    "Fare",
]

_TIME_BUCKETS: List[Tuple[str, int]] = [
    ("dawn", 2),
    ("morning", 8),
    ("afternoon", 14),
    ("night", 20),
]


def _normalize_payload(event: Dict[str, Any]) -> Dict[str, Any]:
    payload = event.get("body", event)
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8")
    if isinstance(payload, str):
        payload = json.loads(payload)
    return payload


def _validate_user_input(payload: Dict[str, Any]) -> None:
    missing = [key for key in _USER_REQUIRED if key not in payload]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")


def _build_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    today = datetime.utcnow().date()
    rows: List[Dict[str, Any]] = []

    for day_offset in range(30):
        purchase_date = today + timedelta(days=day_offset)
        for bucket, hour in _TIME_BUCKETS:
            crawl_dt = datetime(
                purchase_date.year,
                purchase_date.month,
                purchase_date.day,
                hour,
                0,
                0,
            )
            rows.append(
                {
                    "Source": payload["origin"],
                    "Destination": payload["destination"],
                    "Departure Date": payload["departure_date"],
                    "Departure Time": payload["departure_time"],
                    "Crawl Timestamp": crawl_dt.isoformat(),
                    "Number Of Stops": payload["stops_count"],
                    "Total Time": payload["total_time"],
                    "Fare": payload["fare"],
                    "_bucket": bucket,
                    "_purchase_date": purchase_date.isoformat(),
                }
            )
    return rows


def _to_csv(rows: List[Dict[str, Any]]) -> str:
    from io import StringIO
    import csv

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=_MODEL_COLUMNS)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key) for key in _MODEL_COLUMNS})
    return buffer.getvalue()


def _parse_predictions(body: bytes) -> List[float]:
    text = body.decode("utf-8").strip()
    if not text:
        return []
    lines = [line for line in text.splitlines() if line.strip()]
    return [float(line.split(",")[0]) for line in lines]


def _postprocess_predictions(
    rows: List[Dict[str, Any]],
    preds: List[float],
    fare: float,
) -> List[float]:
    if not preds:
        return preds

    base = fare if fare and fare > 0 else max(1.0, sum(preds) / len(preds))
    floor = max(1.0, base * 0.6)
    ceiling = max(floor + 100.0, base * 2.5)

    bucket_adjust = {
        "dawn": 0.95,
        "morning": 1.0,
        "afternoon": 1.05,
        "night": 1.1,
    }

    adjusted: List[float] = []
    for row, pred in zip(rows, preds):
        if pred is None or not math.isfinite(pred):
            pred = base

        # Blend with current fare to reduce extreme spikes.
        blended = (0.7 * pred) + (0.3 * base)
        blended *= bucket_adjust.get(row.get("_bucket"), 1.0)
        blended = min(max(blended, floor), ceiling)
        adjusted.append(blended)

    return adjusted


def _rank_top_3(rows: List[Dict[str, Any]], preds: List[float]) -> List[Dict[str, Any]]:
    combined = []
    for row, pred in zip(rows, preds):
        combined.append(
            {
                "date": row["_purchase_date"],
                "time_bucket": row["_bucket"],
                "predicted_price": pred,
            }
        )
    combined.sort(key=lambda x: x["predicted_price"])
    top3 = []
    for rank, item in enumerate(combined[:3], start=1):
        top3.append(
            {
                "rank": rank,
                "date": item["date"],
                "time_bucket": item["time_bucket"],
                "predicted_price": round(item["predicted_price"]),
            }
        )
    return top3


def _trend_series(rows: List[Dict[str, Any]], preds: List[float]) -> List[Dict[str, Any]]:
    daily_min: Dict[str, float] = {}
    for row, pred in zip(rows, preds):
        date_key = row["_purchase_date"]
        current = daily_min.get(date_key)
        if current is None or pred < current:
            daily_min[date_key] = pred

    return [
        {"date": date_key, "predicted_price": round(daily_min[date_key])}
        for date_key in sorted(daily_min.keys())
    ]


def _cors_headers() -> Dict[str, str]:
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST,OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }


def lambda_handler(event: Dict[str, Any], _context: Any) -> Dict[str, Any]:
    if event.get("httpMethod") == "OPTIONS":
        return {
            "statusCode": 200,
            "headers": _cors_headers(),
            "body": "",
        }

    endpoint_name = (
        os.environ.get("ENDPOINT_NAME")
        or event.get("endpoint_name")
        or event.get("EndpointName")
    )
    if not endpoint_name:
        prefix = os.environ.get("ENDPOINT_PREFIX")
        if not prefix:
            return {
                "statusCode": 400,
                "headers": _cors_headers(),
                "body": json.dumps(
                    {"error": "ENDPOINT_NAME or ENDPOINT_PREFIX is required."}
                ),
            }

        response = _SM.list_endpoints(
            NameContains=prefix,
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=1,
        )
        summaries = response.get("Endpoints", [])
        if not summaries:
            return {
                "statusCode": 404,
                "headers": _cors_headers(),
                "body": json.dumps(
                    {"error": f"No endpoints found for prefix: {prefix}"}
                ),
            }
        endpoint_name = summaries[0]["EndpointName"]

    try:
        payload = _normalize_payload(event)
        _validate_user_input(payload)

        rows = _build_rows(payload)
        payload_csv = _to_csv(rows)

        response = _RUNTIME.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="text/csv",
            Accept="text/csv",
            Body=payload_csv.encode("utf-8"),
        )
        preds_log = _parse_predictions(response["Body"].read())
        preds = [math.expm1(v) for v in preds_log]
        preds = _postprocess_predictions(rows, preds, float(payload["fare"]))

        top3 = _rank_top_3(rows, preds)
        trend = _trend_series(rows, preds)
        response_body = {
            "user_input": {
                "origin": payload["origin"],
                "destination": payload["destination"],
                "departure_date": payload["departure_date"],
                "arrival_date": payload["arrival_date"],
                "stops_count": payload["stops_count"],
                "departure_time": payload["departure_time"],
                "total_time": payload["total_time"],
                "fare": payload["fare"],
            },
            "top_3_cheapest_purchase_times": top3,
            "price_trend_30d": {
                "data": trend,
                "metadata": {
                    "horizon_days": 30,
                    "highlight": "min_price_day",
                },
            },
        }
        if os.environ.get("DEBUG_PREDICTIONS") == "1":
            response_body["debug"] = {
                "preds_log_min": min(preds_log) if preds_log else None,
                "preds_log_max": max(preds_log) if preds_log else None,
                "preds_min": min(preds) if preds else None,
                "preds_max": max(preds) if preds else None,
                "sample_preds": preds[:5],
            }

        return {
            "statusCode": 200,
            "headers": _cors_headers(),
            "body": json.dumps(response_body),
        }
    except Exception as exc:
        return {
            "statusCode": 500,
            "headers": _cors_headers(),
            "body": json.dumps({"error": str(exc)}),
        }
