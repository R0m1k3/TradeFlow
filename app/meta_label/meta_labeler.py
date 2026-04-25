"""
MetaLabeler — LightGBM classifier that decides if a primary signal is worth acting on.

Pipeline:
  1. Primary strategy fires a signal (Dual Momentum / Cross-Sectional / Pullback)
  2. Build feature vector at signal time (volatility, regime, momentum, technical)
  3. MetaLabeler.predict_proba(features) → P(success)
  4. If P > threshold (default 0.55), execute. Else skip.

Training is done offline on historical events labeled via TripleBarrier.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from app.risk.stops import compute_atr

logger = logging.getLogger(__name__)


@dataclass
class MetaLabelConfig:
    threshold: float = 0.55          # min probability to act on a signal
    n_estimators: int = 300
    max_depth: int = 5
    learning_rate: float = 0.05
    random_state: int = 42


def build_features(df: pd.DataFrame, idx: int, regime_signal: dict | None = None) -> dict:
    """
    Build feature vector at bar `idx`. All features must be known at decision time
    (no lookahead).
    """
    close = df["close"]
    volume = df["volume"] if "volume" in df.columns else None

    if idx < 252 or idx >= len(close):
        return {}

    # Returns
    ret_1d = float(close.iloc[idx] / close.iloc[idx - 1] - 1)
    ret_5d = float(close.iloc[idx] / close.iloc[idx - 5] - 1)
    ret_21d = float(close.iloc[idx] / close.iloc[idx - 21] - 1)
    ret_63d = float(close.iloc[idx] / close.iloc[idx - 63] - 1)
    ret_252d = float(close.iloc[idx] / close.iloc[idx - 252] - 1)

    # Volatility
    log_ret = np.log(close / close.shift(1)).iloc[idx - 21:idx + 1]
    vol_20d = float(log_ret.std() * np.sqrt(252))

    log_ret_60 = np.log(close / close.shift(1)).iloc[idx - 60:idx + 1]
    vol_60d = float(log_ret_60.std() * np.sqrt(252))
    vol_ratio = vol_20d / vol_60d if vol_60d > 0 else 1.0

    # Trend
    sma20 = float(close.iloc[idx - 19:idx + 1].mean())
    sma50 = float(close.iloc[idx - 49:idx + 1].mean())
    sma200 = float(close.iloc[idx - 199:idx + 1].mean()) if idx >= 199 else float("nan")

    price_vs_sma20 = float(close.iloc[idx] / sma20 - 1)
    price_vs_sma50 = float(close.iloc[idx] / sma50 - 1)
    price_vs_sma200 = float(close.iloc[idx] / sma200 - 1) if not np.isnan(sma200) else 0.0
    sma20_vs_sma50 = sma20 / sma50 - 1 if sma50 > 0 else 0.0

    # ATR-normalized distance from recent extremes
    atr = compute_atr(df.iloc[max(0, idx - 50):idx + 1], 14).iloc[-1]
    high_20 = float(df["high"].iloc[idx - 19:idx + 1].max())
    low_20 = float(df["low"].iloc[idx - 19:idx + 1].min())
    dist_to_high = (high_20 - float(close.iloc[idx])) / atr if atr > 0 else 0.0
    dist_to_low = (float(close.iloc[idx]) - low_20) / atr if atr > 0 else 0.0

    # Volume
    vol_features = {}
    if volume is not None:
        v_avg_20 = float(volume.iloc[idx - 19:idx + 1].mean())
        v_avg_60 = float(volume.iloc[idx - 59:idx + 1].mean()) if idx >= 59 else v_avg_20
        vol_features = {
            "volume_ratio": v_avg_20 / v_avg_60 if v_avg_60 > 0 else 1.0,
        }

    # Regime context
    regime_features = {}
    if regime_signal:
        regime_features = {
            "regime_trend_up": float(regime_signal.get("trend_up", True)),
            "regime_vol_pct": float(regime_signal.get("vol_percentile", 0.5)),
            "regime_exposure": float(regime_signal.get("exposure_multiplier", 1.0)),
        }

    return {
        "ret_1d": ret_1d,
        "ret_5d": ret_5d,
        "ret_21d": ret_21d,
        "ret_63d": ret_63d,
        "ret_252d": ret_252d,
        "vol_20d": vol_20d,
        "vol_60d": vol_60d,
        "vol_ratio": vol_ratio,
        "price_vs_sma20": price_vs_sma20,
        "price_vs_sma50": price_vs_sma50,
        "price_vs_sma200": price_vs_sma200,
        "sma20_vs_sma50": sma20_vs_sma50,
        "dist_to_high_atr": dist_to_high,
        "dist_to_low_atr": dist_to_low,
        **vol_features,
        **regime_features,
    }


class MetaLabeler:
    """LightGBM-based meta-labeler. Trains offline, predicts online."""

    def __init__(self, config: MetaLabelConfig | None = None) -> None:
        self.config = config or MetaLabelConfig()
        self._model = None
        self._feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MetaLabeler":
        """
        Train on (features, binary labels).
        Labels: 1 = signal would have made money (PT hit), 0 = otherwise.
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError(
                "lightgbm not installed. Add `lightgbm>=4.3.0` to requirements.txt"
            )

        # Convert triple-barrier labels (1, 0, -1) to binary (1, 0)
        if y.isin([-1, 0, 1]).all():
            y_binary = (y == 1).astype(int)
        else:
            y_binary = y.astype(int)

        self._feature_names = list(X.columns)
        self._model = lgb.LGBMClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            random_state=self.config.random_state,
            verbose=-1,
        )
        self._model.fit(X, y_binary)
        logger.info(
            "MetaLabeler trained: n=%d, pos_rate=%.2f, features=%d",
            len(y_binary), float(y_binary.mean()), len(self._feature_names),
        )
        return self

    def predict_proba(self, features: dict | pd.DataFrame) -> float:
        """Return P(success) for a single event or batch."""
        if self._model is None:
            return 0.5  # no model → neutral
        if isinstance(features, dict):
            X = pd.DataFrame([features])[self._feature_names]
        else:
            X = features[self._feature_names]
        proba = self._model.predict_proba(X)[:, 1]
        if len(proba) == 1:
            return float(proba[0])
        return proba

    def should_act(self, features: dict) -> tuple[bool, float]:
        """Convenience: returns (act?, probability)."""
        p = self.predict_proba(features)
        if isinstance(p, np.ndarray):
            p = float(p[0])
        return bool(p >= self.config.threshold), float(p)

    def feature_importance(self) -> dict[str, float]:
        if self._model is None:
            return {}
        importances = self._model.feature_importances_
        return dict(zip(self._feature_names, importances.tolist()))

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self._model,
                "feature_names": self._feature_names,
                "config": self.config,
            }, f)
        logger.info("MetaLabeler saved to %s", path)

    @classmethod
    def load(cls, path: Path | str) -> "MetaLabeler":
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        ml = cls(config=data["config"])
        ml._model = data["model"]
        ml._feature_names = data["feature_names"]
        return ml
