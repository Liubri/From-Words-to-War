from pathlib import Path
from typing import Dict, List

import pandas as pd

from .config import PipelineConfig
from .data import build_country_month_panel
from .features import build_features
from .models import isolation_forest_detector, nlp_detector, residual_detector, zscore_detector


def _top_reasons(row: pd.Series, reason_cols: List[str], top_k: int = 3) -> str:
    if not reason_cols:
        return "No engineered features available"
    vals = row[reason_cols].abs().sort_values(ascending=False).head(top_k)
    return ", ".join(f"{k}={float(v):.3f}" for k, v in vals.items())


def run_pipeline(config: PipelineConfig | None = None) -> Dict[str, Path]:
    cfg = config or PipelineConfig()
    total = cfg.nlp_weight + cfg.zscore_weight + cfg.iso_weight + cfg.residual_weight
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Detector weights must sum to 1.0; got {total:.5f}")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    panel = build_country_month_panel(cfg.ungdc_path, cfg.risk_path)
    model_df, feature_cols = build_features(panel, rolling_window=cfg.rolling_window)

    min_obs = cfg.min_history_months
    counts = model_df.groupby("country")["month"].transform("count")
    model_df = model_df[counts >= min_obs].copy()
    if model_df.empty:
        raise ValueError("No country has enough history after feature engineering.")

    z_cols = [c for c in feature_cols if c.endswith("_z")]
    nlp_score = nlp_detector(model_df)
    zscore = zscore_detector(model_df, z_cols)
    residual = residual_detector(model_df)

    iso = isolation_forest_detector(
        model_df[feature_cols],
        contamination=cfg.isolation_forest_contamination,
        random_state=cfg.random_state,
    )

    model_df["score_nlp"] = nlp_score
    model_df["score_z"] = zscore
    model_df["score_iso"] = iso
    model_df["score_residual"] = residual
    model_df["anomaly_score"] = (
        cfg.nlp_weight * model_df["score_nlp"]
        + cfg.zscore_weight * model_df["score_z"]
        + cfg.iso_weight * model_df["score_iso"]
        + cfg.residual_weight * model_df["score_residual"]
    )

    global_threshold = float(model_df["anomaly_score"].quantile(cfg.alert_percentile))
    country_threshold = (
        model_df.groupby("country")["anomaly_score"]
        .transform(lambda s: s.quantile(cfg.alert_percentile))
        .astype(float)
    )
    model_df["anomaly_threshold"] = country_threshold
    model_df["is_anomaly"] = model_df["anomaly_score"] >= model_df["anomaly_threshold"]

    model_df["anomaly_reason"] = model_df.apply(lambda r: _top_reasons(r, z_cols), axis=1)

    score_path = cfg.output_dir / "anomaly_scores.csv"
    alert_path = cfg.output_dir / "anomaly_alerts.csv"
    summary_path = cfg.output_dir / "anomaly_summary.csv"

    output_cols = [
        "country",
        "region",
        "month",
        "anomaly_score",
        "anomaly_threshold",
        "is_anomaly",
        "score_nlp",
        "score_z",
        "score_iso",
        "score_residual",
        "anomaly_reason",
        "conflict_escalation_6m",
    ]
    for col in output_cols:
        if col not in model_df.columns:
            model_df[col] = 0.0

    scores = model_df[output_cols].sort_values(["country", "month"])
    alerts = scores[scores["is_anomaly"]].sort_values(["anomaly_score"], ascending=False)

    summary = (
        scores.groupby("country", as_index=False)
        .agg(
            observations=("month", "count"),
            anomalies=("is_anomaly", "sum"),
            mean_score=("anomaly_score", "mean"),
            max_score=("anomaly_score", "max"),
        )
        .sort_values("max_score", ascending=False)
    )
    summary["global_threshold"] = global_threshold

    scores.to_csv(score_path, index=False)
    alerts.to_csv(alert_path, index=False)
    summary.to_csv(summary_path, index=False)

    return {"scores": score_path, "alerts": alert_path, "summary": summary_path}
