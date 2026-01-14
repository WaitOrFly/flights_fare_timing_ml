import hashlib
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

class RouteHashEncoder(BaseEstimator, TransformerMixin):
    """
    'c59dc251' 같은 8자리 hex string을 숫자로 변환
    - 컬럼은 유지 가능 (원본은 별도 df에 남겨두면 됨)
    - 모델 입력은 float32로 안전해짐
    """
    def __init__(self, mod: int = 1_000_000):
        self.mod = mod

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X: (n, 1) 형태로 들어오는 경우가 많음
        arr = np.asarray(X).reshape(-1)
        out = np.empty((len(arr), 1), dtype=np.float32)

        for i, v in enumerate(arr):
            # 결측/이상값 방어
            if v is None or (isinstance(v, float) and np.isnan(v)):
                out[i, 0] = 0.0
                continue

            s = str(v)
            # hex -> int (실패하면 0 처리)
            try:
                out[i, 0] = float(int(s, 16) % self.mod)
            except Exception:
                out[i, 0] = 0.0

        return out

class FlightFeatureEngineer:
    """
    원본 데이터를 feature schema에 정의된 feature로 변환
    """

    def __init__(self, apply_log_to_target=False):
        # 인도 공휴일 시즌 정의 (도메인 룰 기반 → 누수 위험 없음)
        self.holiday_months = [1, 3, 4, 5, 6, 8, 10, 11]  # Republic Day, Holi, 여름휴가, Independence Day, Diwali/Dussehra
        self.apply_log_to_target = apply_log_to_target

    def transform(
        self,
        df: pd.DataFrame,
        historical_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        원본 데이터를 feature schema 형식으로 변환

        Args:
            df: 현재 split의 데이터
            historical_df: df 이전 시점의 과거 데이터(= 누수 방지용). None이면 df 내부에서 과거만 사용.
        """
        df = df.copy()

        # 날짜/시간 파싱
        df['crawl_datetime'] = pd.to_datetime(df['Crawl Timestamp'], utc=True).dt.tz_localize(None)
        df['departure_datetime'] = pd.to_datetime(df['Departure Date'] + ' ' + df['Departure Time'])
        df['Source'] = df['Source'].astype(str).str.strip().str.upper()
        df['Destination'] = df['Destination'].astype(str).str.strip().str.upper()

        features = pd.DataFrame(index=df.index)

        # 1. purchase_day_of_week: 구매(크롤링) 시점의 요일
        features['purchase_day_of_week'] = df['crawl_datetime'].dt.dayofweek
        features.loc[
            ~features['purchase_day_of_week'].between(0, 6),
            'purchase_day_of_week'
        ] = 0

        # 2. purchase_time_bucket: 구매 시간대
        features['purchase_time_bucket'] = df['crawl_datetime'].dt.hour.apply(
            self._get_time_bucket
        )
        features.loc[
            ~features['purchase_time_bucket'].isin(
                ['dawn', 'morning', 'afternoon', 'night']
            ),
            'purchase_time_bucket'
        ] = 'morning'

        # 3. days_until_departure_bucket: 출발까지 남은 일수
        days_until = (df['departure_datetime'] - df['crawl_datetime']).dt.days
        features['days_until_departure_bucket'] = days_until.apply(
            self._get_days_until_bucket
        )
        features.loc[
            ~features['days_until_departure_bucket'].isin(
                ['very_close', 'close', 'medium', 'far']
            ),
            'days_until_departure_bucket'
        ] = 'medium'

        # 4. is_weekend_departure: 주말 출발 여부
        features['is_weekend_departure'] = (
            df['departure_datetime'].dt.dayofweek >= 5
        ).astype(int)

        # 5. is_holiday_season: 휴가 성수기 여부
        features['is_holiday_season'] = (
            df['departure_datetime'].dt.month.isin(self.holiday_months)
        ).astype(int)

        # 6. price_trend_7d: 가격 추세 (누수 방지: 과거 데이터만)
        features['price_trend_7d'] = self._calculate_price_trend(df, historical_df)

        # 7. current_vs_historical_avg: 현재 가격 vs 평균 가격 비율 (누수 방지: 과거 데이터만)
        features['current_vs_historical_avg'] = self._calculate_price_ratio(df, historical_df)

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
        features.loc[
            ~features['flight_duration_bucket'].isin(
                ['short', 'medium', 'long']
            ),
            'flight_duration_bucket'
        ] = 'medium'

        # Target: price (optional log transform)  ※ feature schema 유지
        if self.apply_log_to_target:
            features['price'] = np.log1p(df['Fare'])
            features['price_original'] = df['Fare']  # 원본 보존
        else:
            features['price'] = df['Fare']

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

    def _hash_route(self, source: str, destination: str) -> str:
        """출발지-목적지를 해시값으로 변환"""
        route_str = f"{source}_{destination}"
        return hashlib.md5(route_str.encode()).hexdigest()[:8]

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

            duration_text = str(duration_str).strip().lower()

            if ":" in duration_text:
                parts = duration_text.split(":")
                if len(parts) >= 2:
                    hours = int(parts[0].strip() or 0)
                    minutes = int(parts[1].strip() or 0)
                    return hours * 60 + minutes

            if 'h' in duration_text:
                parts = duration_text.split('h')
                hours = int(parts[0].strip())
                if len(parts) > 1 and 'm' in parts[1]:
                    minutes = int(parts[1].replace('m', '').strip())
            elif 'm' in duration_text:
                minutes = int(duration_text.replace('m', '').strip())

            return hours * 60 + minutes
        except Exception:
            return 0

    def _get_duration_bucket(self, minutes: int) -> str:
        """비행 시간을 bucket으로 변환"""
        if minutes < 120:  # 2시간 미만
            return 'short'
        elif minutes < 360:  # 6시간 미만
            return 'medium'
        else:
            return 'long'

    # -----------------------------
    # 누수 방지 price stats 계산
    # -----------------------------
    def _prepare_history_stats(
        self,
        df_current: pd.DataFrame,
        historical_df: Optional[pd.DataFrame],
    ):
        """
        route별 '과거 평균'을 현재 row마다 계산 (미래 데이터 참조 금지)
        - historical_df가 None이면 df_current 내부에서 시간순 과거만 사용
        - historical_df가 있으면 historical_df + df_current를 시간순으로 합쳐서
          현재 row의 과거(=historical + earlier current)만 사용
        """
        cur = df_current.copy()
        cur['_is_current'] = 1
        cur['_row_id'] = np.arange(len(cur))

        # df_current는 이미 crawl_datetime 파싱 완료되어 들어옴
        if historical_df is None:
            hist = cur.copy()
            hist['_is_current'] = 1  # 동일 데이터 내에서도 shift(1)로 과거만 사용
        else:
            hist = historical_df.copy()
            hist['crawl_datetime'] = pd.to_datetime(hist['Crawl Timestamp'], utc=True).dt.tz_localize(None)
            hist['_is_current'] = 0
            hist['_row_id'] = -1

        # route 생성
        cur['route'] = cur['Source'] + '_' + cur['Destination']
        hist['route'] = hist['Source'] + '_' + hist['Destination']

        combined = pd.concat([hist, cur], axis=0, ignore_index=True)
        combined = combined.sort_values(['route', 'crawl_datetime'], kind='mergesort').reset_index(drop=True)

        # route별 expanding mean 후 shift(1) → "현재 row 이전까지" 평균
        def _expanding_past_mean(s: pd.Series) -> pd.Series:
            return s.expanding().mean().shift(1)

        combined['past_route_mean'] = combined.groupby('route')['Fare'].transform(_expanding_past_mean)

        # current row만 추출
        current_rows = combined[combined['_is_current'] == 1].copy()
        # df_current 원래 순서로 복원
        current_rows = current_rows.sort_values('_row_id')

        return current_rows['past_route_mean'].reset_index(drop=True)

    def _calculate_price_trend(
        self,
        df: pd.DataFrame,
        historical_df: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """
        가격 추세 계산 (route별 과거 평균 대비 변화율)
        - 누수 방지: 과거 시점 데이터만 사용
        """
        past_mean = self._prepare_history_stats(df, historical_df)

        # past_mean이 NaN(과거 없음) 또는 0인 경우 안전 처리
        denom = past_mean.replace(0, np.nan)
        trend = (df['Fare'].reset_index(drop=True) - past_mean) / denom
        trend = trend.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return trend

    def _calculate_price_ratio(
        self,
        df: pd.DataFrame,
        historical_df: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """
        현재 가격 vs 과거 평균 가격 비율
        - 누수 방지: 과거 시점 데이터만 사용
        """
        past_mean = self._prepare_history_stats(df, historical_df)

        denom = past_mean.replace(0, np.nan)
        ratio = df['Fare'].reset_index(drop=True) / denom
        ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        return ratio

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
        self._required_bool_features = [
            'is_weekend_departure',
            'is_holiday_season',
        ]
        self._setup_preprocessor()

    def _setup_preprocessor(self):
        """전처리 파이프라인 설정"""

        # Feature 그룹 정의 (schema 동일)
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
            'price_trend_7d',
            'current_vs_historical_avg',
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
            RouteHashEncoder(mod=1_000_000),
            high_cardinality_features
        ))

        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )

    def _encode_ordinal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ordinal feature 인코딩"""
        df = df.copy()
        for col in self._required_bool_features:
            if col not in df.columns:
                df[col] = 0
        if 'days_until_departure_bucket' in df.columns:
            df['days_until_departure_bucket'] = df['days_until_departure_bucket'].map(
                self.ordinal_mapping
            )
        return df

    def fit(self, X: pd.DataFrame, y=None):
        """전처리기를 학습 데이터에 fit"""
        X_processed = self._encode_ordinal_features(X)
        self.preprocessor.fit(X_processed)
        self._expected_columns = list(X_processed.columns)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """데이터를 전처리하여 변환"""
        X_processed = self._encode_ordinal_features(X)
        if hasattr(self, "_expected_columns"):
            X_processed = X_processed.reindex(
                columns=self._expected_columns,
                fill_value=0
            )
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
