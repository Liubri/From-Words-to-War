from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


def _minmax(series: pd.Series) -> pd.Series:
    min_v = float(series.min())
    max_v = float(series.max())
    if max_v - min_v < 1e-12:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - min_v) / (max_v - min_v)


def zscore_detector(df: pd.DataFrame, z_columns: List[str]) -> pd.Series:
    if not z_columns:
        return pd.Series(np.zeros(len(df)), index=df.index)
    z_abs = df[z_columns].abs().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return _minmax(z_abs.mean(axis=1))


def nlp_detector(df: pd.DataFrame) -> pd.Series:
    candidates = [
        "threat_rate",
        "threat_rate_diff1",
        "threat_rate_z",
        "seed_sentiment",
        "seed_sentiment_diff1",
        "seed_sentiment_z",
        "speech_risk_divergence",
    ]
    cols = [c for c in candidates if c in df.columns]
    if not cols:
        return pd.Series(np.zeros(len(df)), index=df.index)

    local = df[cols].copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    abs_cols = [c for c in local.columns if c.endswith("_diff1") or c.endswith("_z") or c == "speech_risk_divergence"]
    for c in abs_cols:
        local[c] = local[c].abs()
    return _minmax(local.mean(axis=1))


def residual_detector(df: pd.DataFrame) -> pd.Series:
    components = [
        df[col].abs()
        for col in [
            "instability_score_diff1",
            "social_media_sentiment_diff1",
            "protest_events_last_3m_diff1",
            "threat_rate_diff1",
        ]
        if col in df.columns
    ]
    if not components:
        return pd.Series(np.zeros(len(df)), index=df.index)
    return _minmax(pd.concat(components, axis=1).fillna(0.0).mean(axis=1))


def isolation_forest_detector(
    X: pd.DataFrame, contamination: float = 0.08, random_state: int = 42
) -> pd.Series:
    X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_scaled = RobustScaler().fit_transform(X_clean.values)
    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_scaled)
    return _minmax(pd.Series(-model.score_samples(X_scaled), index=X.index))
