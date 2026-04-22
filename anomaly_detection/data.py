from typing import Dict, Set

import numpy as np
import pandas as pd


THREAT_KEYWORDS: Set[str] = {
    "weapon",
    "attack",
    "military",
    "bomb",
    "terrorist",
    "rebel",
    "war",
    "conflict",
    "army",
    "violence",
    "security",
    "hostility",
    "aggression",
    "threat",
    "crisis",
    "danger",
    "missile",
    "nuclear",
    "siege",
    "invasion",
    "sanctions",
    "ceasefire",
    "blockade",
    "retaliation",
}

POSITIVE_SEEDS: Set[str] = {
    "peace",
    "cooperation",
    "dialogue",
    "agreement",
    "resolve",
    "stability",
    "diplomacy",
    "treaty",
}

NEGATIVE_SEEDS: Set[str] = {
    "war",
    "conflict",
    "attack",
    "threat",
    "violence",
    "aggression",
    "kill",
    "destroy",
    "invade",
    "terror",
    "massacre",
    "genocide",
    "bomb",
    "missile",
}

COUNTRY_NAME_MAP: Dict[str, str] = {
    "UK": "United Kingdom",
    "USA": "United States",
    "South Korea": "Korea, South",
    "North Korea": "Korea, North",
}


def _text_features(text: str) -> tuple[int, int, int, int]:
    lowered = text.lower()
    return (
        int(sum(lowered.count(t) for t in THREAT_KEYWORDS)),
        int(sum(lowered.count(t) for t in POSITIVE_SEEDS)),
        int(sum(lowered.count(t) for t in NEGATIVE_SEEDS)),
        max(len(text.split()), 1),
    )


def load_ungdc_monthly_features(ungdc_path: str | bytes | "os.PathLike[str]") -> pd.DataFrame:
    cols = ["country", "year", "text"]
    ungdc = pd.read_csv(ungdc_path, usecols=cols, low_memory=False)
    ungdc = ungdc.dropna(subset=["country", "year", "text"]).copy()
    ungdc["country"] = ungdc["country"].astype(str).str.strip()
    ungdc["year"] = pd.to_numeric(ungdc["year"], errors="coerce")
    ungdc = ungdc.dropna(subset=["year"])
    ungdc["year"] = ungdc["year"].astype(int)
    ungdc["month"] = pd.to_datetime(ungdc["year"].astype(str) + "-09-01")

    ungdc[["threat_count", "pos_seed_count", "neg_seed_count", "word_count"]] = [
        _text_features(t) for t in ungdc["text"]
    ]
    ungdc["threat_rate"] = ungdc["threat_count"] / ungdc["word_count"] * 1000.0
    ungdc["seed_sentiment"] = (ungdc["pos_seed_count"] - ungdc["neg_seed_count"]) / (
        ungdc["pos_seed_count"] + ungdc["neg_seed_count"] + 1.0
    )

    annual = (
        ungdc.groupby(["country", "month"], as_index=False)
        .agg(
            threat_count=("threat_count", "sum"),
            word_count=("word_count", "sum"),
            threat_rate=("threat_rate", "mean"),
            seed_sentiment=("seed_sentiment", "mean"),
            speech_count=("text", "count"),
        )
        .sort_values(["country", "month"])
    )
    return annual


def load_risk_dataset(risk_path: str | bytes | "os.PathLike[str]") -> pd.DataFrame:
    risk = pd.read_csv(risk_path)
    risk["month"] = pd.to_datetime(risk["month"], format="%Y-%m", errors="coerce")
    risk = risk.dropna(subset=["country", "month"]).copy()
    risk["country"] = risk["country"].replace(COUNTRY_NAME_MAP)
    risk = risk.sort_values(["country", "month"]).reset_index(drop=True)
    return risk


def build_country_month_panel(ungdc_path: str | bytes | "os.PathLike[str]", risk_path: str | bytes | "os.PathLike[str]") -> pd.DataFrame:
    ungdc_features = load_ungdc_monthly_features(ungdc_path)
    risk = load_risk_dataset(risk_path)

    panel = risk.merge(ungdc_features, on=["country", "month"], how="left")
    panel[["threat_count", "word_count", "threat_rate", "seed_sentiment", "speech_count"]] = panel[
        ["threat_count", "word_count", "threat_rate", "seed_sentiment", "speech_count"]
    ].fillna(0.0)

    numeric_cols = panel.select_dtypes(include=[np.number]).columns.tolist()
    if "conflict_escalation_6m" in numeric_cols:
        numeric_cols.remove("conflict_escalation_6m")
    panel[numeric_cols] = panel[numeric_cols].replace([np.inf, -np.inf], np.nan)

    panel = panel.sort_values(["country", "month"]).reset_index(drop=True)
    return panel

