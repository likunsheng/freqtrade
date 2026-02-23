from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from pandas import DataFrame

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from perppath_ml_common import ML_FEATURE_COLUMNS, add_ml_features
from PerpPathUnifiedFactorTopK3 import PerpPathUnifiedFactorTopK3


class PerpPathUnifiedMLTopK3(PerpPathUnifiedFactorTopK3):
    """
    Unified pool strategy with ML-enhanced scoring.
    - Factor score remains as robustness anchor.
    - ML up/down probabilities dominate when model is available.
    """

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

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = super().populate_indicators(dataframe, metadata)

        pair = metadata.get("pair", "")
        pair_id = 1.0 if pair.startswith("ETH/") else 0.0
        ml_df = add_ml_features(df, pair_id=pair_id)

        self._load_ml_artifacts()
        if self._ml_model is None:
            df["ml_up_prob"] = np.nan
            df["ml_dn_prob"] = np.nan
            return df

        features = self._ml_meta.get("features", ML_FEATURE_COLUMNS)
        x = ml_df[features].replace([np.inf, -np.inf], np.nan)
        valid = x.notna().all(axis=1)

        up_prob = np.full(len(df), np.nan, dtype=float)
        if valid.any():
            up_prob[valid.to_numpy()] = self._ml_model.predict_proba(x.loc[valid].to_numpy())[:, 1]
        dn_prob = 1.0 - up_prob

        df["ml_up_prob"] = up_prob
        df["ml_dn_prob"] = dn_prob

        # Blend factor and ML probabilities.
        factor_w = 0.35
        ml_w = 0.65
        df["score_long"] = np.where(
            np.isnan(df["ml_up_prob"]),
            df["score_long"],
            (factor_w * df["score_long"]) + (ml_w * df["ml_up_prob"]),
        )
        df["score_short"] = np.where(
            np.isnan(df["ml_dn_prob"]),
            df["score_short"],
            (factor_w * df["score_short"]) + (ml_w * df["ml_dn_prob"]),
        )
        return df
