from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from pandas import DataFrame

from freqtrade.strategy import DecimalParameter

from PerpPathTrendOnlyV1 import PerpPathTrendOnlyV1
from perppath_ml_common import ML_FEATURE_COLUMNS, add_ml_features


class PerpPathTrendOnlyV1Lev3MLFilter(PerpPathTrendOnlyV1):
    """
    Baseline Lev3 trend strategy + ML entry filter.
    The core long rules are unchanged; ML only blocks low-quality entries.
    """

    can_short = False
    ml_min_prob = DecimalParameter(0.50, 0.85, default=0.62, decimals=2, space="buy", optimize=True)
    leverage_value = 3.0

    _ml_model = None
    _ml_meta = {}
    _ml_ready = False

    @classmethod
    def _load_ml_artifacts(cls) -> None:
        if cls._ml_ready:
            return

        user_data_dir = Path(__file__).resolve().parents[1]
        model_path = user_data_dir / "models" / "perppath_lev3_ml_filter.pkl"
        meta_path = user_data_dir / "models" / "perppath_lev3_ml_filter_meta.json"

        if model_path.exists() and meta_path.exists():
            cls._ml_model = joblib.load(model_path)
            with meta_path.open("r", encoding="utf-8") as fh:
                cls._ml_meta = json.load(fh)
        else:
            cls._ml_model = None
            cls._ml_meta = {}

        cls._ml_ready = True

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        return min(float(self.leverage_value), float(max_leverage))

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = super().populate_indicators(dataframe, metadata)
        pair = metadata.get("pair", "")
        pair_id = 1.0 if pair.startswith("ETH/") else 0.0

        ml_df = add_ml_features(df, pair_id=pair_id)
        for col in ML_FEATURE_COLUMNS:
            df[col] = ml_df[col]

        self._load_ml_artifacts()
        if self._ml_model is None:
            df["ml_prob"] = np.nan
            df["ml_long_ok"] = 1
            return df

        features = self._ml_meta.get("features", ML_FEATURE_COLUMNS)
        x = df[features].replace([np.inf, -np.inf], np.nan)
        valid = x.notna().all(axis=1)

        probs = np.full(len(df), np.nan, dtype=float)
        if valid.any():
            probs[valid.to_numpy()] = self._ml_model.predict_proba(x.loc[valid].to_numpy())[:, 1]

        threshold = float(self.ml_min_prob.value)
        df["ml_prob"] = probs
        df["ml_long_ok"] = (df["ml_prob"] >= threshold).astype(int)
        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df = super().populate_entry_trend(df, metadata)

        if "enter_long" not in df.columns:
            df["enter_long"] = 0
        if "enter_tag" not in df.columns:
            df["enter_tag"] = None

        long_signal = df["enter_long"] == 1
        low_quality = long_signal & (df["ml_long_ok"] != 1)
        accepted = long_signal & (df["ml_long_ok"] == 1)

        df.loc[low_quality, ["enter_long", "enter_tag"]] = (0, None)
        df.loc[accepted, "enter_tag"] = "trend_long_ml"
        return df
