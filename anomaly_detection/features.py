from typing import List

import numpy as np
import pandas as pd


BASE_FEATURES: List[str] = [
    "social_media_sentiment",
    "political_stability_index",
    "instability_score",
    "protest_events_last_3m",
    "cyber_attack_incidents",
    "border_disputes_count",
    "military_expenditure_pct_gdp",
    "arms_imports_index",
    "sanctions_active",
    "threat_rate",
    "seed_sentiment",
]


def build_features(panel: pd.DataFrame, rolling_window: int = 6) -> tuple[pd.DataFrame, List[str]]:
    df = panel.copy()

    for col in BASE_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["sanctions_active"] = df["sanctions_active"].round().clip(lower=0, upper=1)

    engineered: List[str] = []
    by_country = df.groupby("country", group_keys=False)

    for col in BASE_FEATURES:
        lag1 = f"{col}_lag1"
        diff1 = f"{col}_diff1"
        roll_mean = f"{col}_rollmean"
        roll_std = f"{col}_rollstd"
        z_col = f"{col}_z"

        df[lag1] = by_country[col].shift(1)
        df[diff1] = df[col] - df[lag1]
        df[roll_mean] = by_country[col].rolling(window=rolling_window, min_periods=3).mean().reset_index(level=0, drop=True)
        df[roll_std] = (
            by_country[col]
            .rolling(window=rolling_window, min_periods=3)
            .std(ddof=0)
            .reset_index(level=0, drop=True)
        )
        df[z_col] = (df[col] - df[roll_mean]) / df[roll_std].replace(0, np.nan)

        engineered.extend([col, lag1, diff1, roll_mean, roll_std, z_col])

    df["speech_risk_divergence"] = df["social_media_sentiment"] - df["seed_sentiment"]
    df["threat_protest_interaction"] = df["threat_rate"] * (1.0 + df["protest_events_last_3m"])
    df["instability_x_sanctions"] = df["instability_score"] * (1.0 + df["sanctions_active"])
    engineered.extend(["speech_risk_divergence", "threat_protest_interaction", "instability_x_sanctions"])

    df[engineered] = df[engineered].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df.sort_values(["country", "month"]).reset_index(drop=True), engineered
