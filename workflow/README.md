# Wait or Fly?
항공권 가격은 **언제 구매하느냐**에 따라 크게 달라지지만,
사용자는 “지금 사야 할지, 며칠 기다려야 할지”를 판단하기 어렵다.

이 프로젝트는 **과거 항공권 가격의 시간적 변동 패턴을 학습해, 최적의 구매 시점을 예측하고 리포트로 제공하는 서비스**를 목표로 한다.

# 사용한 알고리즘
XGBoost Regressor

# Feature Engineering Specification

## Feature Schema (v1.0-flat)

- **Model Input Type:** `tabular_flat`
- **Target Variable:** `price` (numeric)

### Input Features

| Feature Name                  | Type                         | Encoding | Possible Values / Description                                                                                         |
| ----------------------------- | ---------------------------- | -------- | --------------------------------------------------------------------------------------------------------------------- |
| `purchase_day_of_week`        | categorical                  | one-hot  | 조회(구매)한 요일 (0=월요일, 6=일요일). 요일별 가격 리프레시 및 할인 패턴을 학습.                                     |
| `purchase_time_bucket`        | categorical                  | one-hot  | 조회 시간을 시간대 구간(`dawn`, `morning`, `afternoon`, `night`)으로 변환한 변수. 시간대별 가격 업데이트 패턴을 반영. |
| `days_until_departure_bucket` | ordinal                      | label    | 출발일까지 남은 기간을 구간화한 변수 (`very_close` → `far`). 출발 임박도에 따른 가격 패턴을 안정적으로 학습.          |
| `is_weekend_departure`        | boolean                      | 0/1      | 출발일이 주말인지 여부. 주말 출발 항공권의 가격 프리미엄 반영.                                                        |
| `is_holiday_season`           | boolean                      | 0/1      | 출발일이 성수기(휴가철, 연말연시 등)에 해당하는지 여부.                                                               |
| `price_trend_7d`              | numeric                      | as-is    | 최근 7일 평균 가격 대비 현재 가격의 변화율. 단기 가격 상승/하락 추세를 나타내는 핵심 Feature.                         |
| `current_vs_historical_avg`   | numeric                      | as-is    | 현재 가격과 과거 평균 가격의 비율. 절대 가격이 아닌 상대적 가격 수준을 표현.                                          |
| `route_hash`                  | categorical_high_cardinality | as-is    | 출발지-도착지 조합을 해시화한 값. 노선별 가격 패턴을 일반화하여 학습.                                                 |
| `stops_count`                 | numeric                      | as-is    | 경유 횟수. 경유 수 증가에 따른 가격 차이를 반영.                                                                      |
| `flight_duration_bucket`      | categorical                  | one-hot  | 비행 시간을 구간화(`short`, `medium`, `long`)한 변수. 장·단거리 항공권 가격 특성을 반영.                              |

### 🎯 Target

| Name    | Type    | Description             |
| ------- | ------- | ----------------------- |
| `price` | numeric | 항공권 가격 (예측 대상) |
