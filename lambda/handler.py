import datetime as dt
import json
import os
from typing import Dict, List, Tuple

import boto3
import numpy as np

from common import (
    HOLIDAY_MONTHS,
    TIME_BUCKETS,
    build_candidate_features,
    compute_route_hash,
    parse_date,
    plot_price_trend,
    upload_plot,
    validate_user_input,
)


def _parse_event(event: Dict[str, object]) -> Dict[str, object]:
    if isinstance(event, dict) and "body" in event:
        body = event.get("body") or "{}"
        if isinstance(body, str):
            try:
                return json.loads(body)
            except json.JSONDecodeError as exc:
                raise ValueError("event.body must be valid JSON") from exc
    return event


def _build_top_k(
    candidates: List[Tuple[str, str, float]],
    k: int = 3,
) -> List[Dict[str, object]]:
    sorted_candidates = sorted(candidates, key=lambda item: item[2])[:k]
    results = []
    for rank, (date_str, time_bucket, price) in enumerate(sorted_candidates, start=1):
        results.append(
            {
                "rank": rank,
                "date": date_str,
                "time_bucket": time_bucket,
                "predicted_price": int(round(price)),
            }
        )
    return results


def _invoke_prediction_lambda(
    lambda_client,
    function_name: str,
    features: Dict[str, object],
) -> float:
    payload = json.dumps({"features": features}).encode("utf-8")
    response = lambda_client.invoke(
        FunctionName=function_name,
        InvocationType="RequestResponse",
        Payload=payload,
    )
    raw = response.get("Payload").read()
    decoded = json.loads(raw)
    body = decoded.get("body", "{}")
    result = json.loads(body)
    return float(result["prediction"])


def handler(event, context):
    event = _parse_event(event)
    user_input = event.get("user_input")
    if not isinstance(user_input, dict):
        raise ValueError("event.user_input must be an object")

    departure_date = parse_date(str(user_input["departure_date"]))

    if "route_hash" not in user_input:
        origin = user_input.get("origin")
        destination = user_input.get("destination")
        if origin and destination:
            user_input["route_hash"] = compute_route_hash(origin, destination)

    if "is_weekend_departure" not in user_input:
        user_input["is_weekend_departure"] = int(departure_date.weekday() >= 5)

    if "is_holiday_season" not in user_input:
        user_input["is_holiday_season"] = int(departure_date.month in HOLIDAY_MONTHS)

    validate_user_input(user_input)

    invoke_function = os.environ.get("INVOKE_FUNCTION_NAME")
    if not invoke_function:
        raise ValueError("INVOKE_FUNCTION_NAME environment variable is required")

    horizon_days = int(os.environ.get("HORIZON_DAYS", "30"))
    start_date_str = event.get("purchase_start_date")
    start_date = parse_date(start_date_str) if start_date_str else dt.date.today()
    candidate_rows = []
    candidate_meta = []
    for day_offset in range(horizon_days):
        purchase_date = start_date + dt.timedelta(days=day_offset)
        days_until = (departure_date - purchase_date).days
        if days_until < 0:
            continue
        for time_bucket in TIME_BUCKETS:
            row = build_candidate_features(
                user_input, purchase_date, time_bucket, departure_date
            )
            candidate_rows.append(row)
            candidate_meta.append((purchase_date.isoformat(), time_bucket))

    if not candidate_rows:
        raise ValueError("No valid purchase dates found within horizon")

    lambda_client = boto3.client("lambda")
    candidates: List[Tuple[str, str, float]] = []
    daily_min: Dict[str, float] = {}
    daily_best: Dict[str, Tuple[str, float]] = {}

    for row, (date_str, time_bucket) in zip(candidate_rows, candidate_meta):
        raw_value = _invoke_prediction_lambda(lambda_client, invoke_function, row)
        price_value = float(np.expm1(raw_value))
        candidates.append((date_str, time_bucket, price_value))
        daily_min[date_str] = min(daily_min.get(date_str, price_value), price_value)
        best = daily_best.get(date_str)
        if best is None or price_value < best[1]:
            daily_best[date_str] = (time_bucket, price_value)

    daily_candidates = [
        (date_str, time_bucket, price)
        for date_str, (time_bucket, price) in daily_best.items()
    ]
    top_3 = _build_top_k(daily_candidates, k=3)
    trend_dates = sorted(daily_min.keys())
    trend_prices = [daily_min[date_str] for date_str in trend_dates]

    plot_name = f"price_trend_30d_{dt.datetime.utcnow().strftime('%Y%m%d%H%M%S')}.png"
    plot_path = os.path.join("/tmp", plot_name)
    plot_price_trend(trend_dates, trend_prices, plot_path)
    plot_url = upload_plot(plot_path)

    response = {
        "user_input": {
            "origin": user_input["origin"],
            "destination": user_input["destination"],
            "departure_date": user_input["departure_date"],
            "arrival_date": user_input["arrival_date"],
            "stops_count": user_input["stops_count"],
        },
        "top_3_cheapest_purchase_times": top_3,
        "price_trend_30d": {
            "image": {"url": plot_url},
            "metadata": {
                "horizon_days": horizon_days,
                "highlight": "min_price_day",
            },
        },
    }

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(response, ensure_ascii=False),
    }
