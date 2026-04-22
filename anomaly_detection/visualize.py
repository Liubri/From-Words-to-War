from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _parse_reason_features(reason: str) -> list[str]:
    if not isinstance(reason, str) or not reason.strip():
        return []
    feats: list[str] = []
    for token in reason.split(","):
        name = token.strip().split("=")[0].strip()
        if name:
            feats.append(name)
    return feats


def generate_visualizations(
    output_dir: Path,
    nlp_weight: float = 0.40,
    zscore_weight: float = 0.25,
    iso_weight: float = 0.20,
    residual_weight: float = 0.15,
) -> dict[str, Path]:
    scores_path = output_dir / "anomaly_scores.csv"
    alerts_path = output_dir / "anomaly_alerts.csv"
    summary_path = output_dir / "anomaly_summary.csv"

    scores = pd.read_csv(scores_path)
    alerts = pd.read_csv(alerts_path)
    summary = pd.read_csv(summary_path)
    scores["month"] = pd.to_datetime(scores["month"])
    alerts["month"] = pd.to_datetime(alerts["month"])

    paths: dict[str, Path] = {}

    # top countries by max anomaly score
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    top = summary.sort_values("max_score", ascending=False).head(10).iloc[::-1]
    ax1.barh(top["country"], top["max_score"], color="#756bb1", alpha=0.9)
    ax1.set_title("Top 10 countries by max anomaly score")
    ax1.set_xlabel("Max anomaly score")
    ax1.grid(axis="x", alpha=0.25)
    fig1.tight_layout()
    p1 = output_dir / "plot_top_countries_max_score.png"
    fig1.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    paths["top_countries"] = p1

    # anomaly events per month
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    anomaly_counts = alerts.groupby("month").size().rename("count").reset_index()
    ax2.bar(anomaly_counts["month"], anomaly_counts["count"], color="#e34a33", alpha=0.85, width=20)
    ax2.set_title("Anomaly alerts per month")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Alert count")
    ax2.grid(axis="y", alpha=0.25)
    fig2.tight_layout()
    p2 = output_dir / "plot_alerts_per_month.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    paths["alerts_per_month"] = p2

    # top reason features among alerts
    fig3, ax3 = plt.subplots(figsize=(12, 7))
    reason_counts: dict[str, int] = {}
    for reason in alerts.get("anomaly_reason", pd.Series(dtype=str)):
        for feat in _parse_reason_features(reason):
            reason_counts[feat] = reason_counts.get(feat, 0) + 1
    reason_df = (
        pd.DataFrame({"feature": list(reason_counts.keys()), "count": list(reason_counts.values())})
        .sort_values("count", ascending=False)
        .head(15)
    )
    if not reason_df.empty:
        top_reasons = reason_df.iloc[::-1]
        ax3.barh(top_reasons["feature"], top_reasons["count"], color="#4daf4a", alpha=0.9)
    ax3.set_title("Most frequent anomaly reason features")
    ax3.set_xlabel("Count across alerts")
    ax3.grid(axis="x", alpha=0.25)
    fig3.tight_layout()
    p3 = output_dir / "plot_top_reason_features.png"
    fig3.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    paths["reason_features"] = p3

    # top-alert component decomposition
    fig4, ax4 = plt.subplots(figsize=(14, 7))
    top_alerts = alerts.nlargest(12, "anomaly_score").copy()
    if not top_alerts.empty:
        top_alerts["label"] = top_alerts["country"].astype(str) + " " + top_alerts["month"].dt.strftime("%Y-%m")
        top_alerts = top_alerts.sort_values("anomaly_score", ascending=True)
        w_n, w_z, w_i, w_r = nlp_weight, zscore_weight, iso_weight, residual_weight
        c0 = top_alerts["score_nlp"] * w_n if "score_nlp" in top_alerts.columns else 0.0
        c1 = top_alerts["score_z"] * w_z
        c2 = top_alerts["score_iso"] * w_i
        c3 = top_alerts["score_residual"] * w_r
        ax4.barh(top_alerts["label"], c0, color="#1b9e77", label="nlp contribution")
        ax4.barh(top_alerts["label"], c1, left=c0, color="#66c2a4", label="z-score contribution")
        ax4.barh(top_alerts["label"], c2, left=(c0 + c1), color="#fc8d62", label="isolation contribution")
        ax4.barh(top_alerts["label"], c3, left=(c0 + c1 + c2), color="#8da0cb", label="residual contribution")
    ax4.set_title("Top alert decomposition by detector contribution")
    ax4.set_xlabel("Weighted contribution to anomaly score")
    ax4.grid(axis="x", alpha=0.25)
    ax4.legend(loc="lower right")
    fig4.tight_layout()
    p4 = output_dir / "plot_top_alert_decomposition.png"
    fig4.savefig(p4, dpi=150, bbox_inches="tight")
    plt.close(fig4)
    paths["alert_decomposition"] = p4

    # anomaly score heatmap for top countries
    fig5, ax5 = plt.subplots(figsize=(14, 8))
    top_countries = (
        summary.sort_values("max_score", ascending=False)["country"].head(12).tolist()
        if not summary.empty
        else []
    )
    heat_src = scores[scores["country"].isin(top_countries)].copy()
    if not heat_src.empty:
        heat_src["ym"] = heat_src["month"].dt.strftime("%Y-%m")
        piv = heat_src.pivot_table(index="country", columns="ym", values="anomaly_score", aggfunc="mean")
        arr = piv.values
        im = ax5.imshow(arr, aspect="auto", cmap="magma")
        ax5.set_yticks(np.arange(len(piv.index)))
        ax5.set_yticklabels(piv.index)
        step = max(1, len(piv.columns) // 12)
        x_idx = np.arange(0, len(piv.columns), step)
        ax5.set_xticks(x_idx)
        ax5.set_xticklabels(piv.columns[x_idx], rotation=45, ha="right")
        cbar = plt.colorbar(im, ax=ax5)
        cbar.set_label("Anomaly score")
    ax5.set_title("Anomaly score heatmap (top countries by max score)")
    fig5.tight_layout()
    p5 = output_dir / "plot_anomaly_heatmap_top_countries.png"
    fig5.savefig(p5, dpi=150, bbox_inches="tight")
    plt.close(fig5)
    paths["heatmap"] = p5

    return paths
