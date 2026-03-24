# ============================================================
# GDR_risk_NER_model.py
# Conflict NER Analysis — UN Speeches vs GPR Index
#
# Datasets:
#   - data_gpr_export.csv  (Caldara & Iacoviello GPR index)
#   - un-general-debates.csv (UNGDC 1946-2023)
#
# Run: python GDR_risk_NER_model.py
# ============================================================

import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity

# ── CONFIG ───────────────────────────────────────────────────
GPR_FILE   = "data_gpr_export.csv"
UNGDC_FILE = "UNGDC_1946-2023.csv"   # adjust filename if needed

# Number of speeches to process — increase to len(ungdc) for full run
# 300 = ~3-7 min,  len(ungdc) = ~30-50 min
SAMPLE_N = 300

# Country columns in your GPR file use the gprc_ prefix
GPR_COUNTRY_MAP = {
    "ARG": "gprc_arg", "AUS": "gprc_aus", "BEL": "gprc_bel",
    "BRA": "gprc_bra", "CAN": "gprc_can", "CHE": "gprc_che",
    "CHL": "gprc_chl", "CHN": "gprc_chn", "COL": "gprc_col",
    "DEU": "gprc_deu", "DNK": "gprc_dnk", "EGY": "gprc_egy",
    "ESP": "gprc_esp", "FIN": "gprc_fin", "FRA": "gprc_fra",
    "GBR": "gprc_gbr", "HKG": "gprc_hkg", "HUN": "gprc_hun",
    "IDN": "gprc_idn", "IND": "gprc_ind", "ISR": "gprc_isr",
    "ITA": "gprc_ita", "JPN": "gprc_jpn", "KOR": "gprc_kor",
    "MEX": "gprc_mex", "MYS": "gprc_mys", "NLD": "gprc_nld",
    "NOR": "gprc_nor", "PER": "gprc_per", "PHL": "gprc_phl",
    "POL": "gprc_pol", "PRT": "gprc_prt", "RUS": "gprc_rus",
    "SAU": "gprc_sau", "SWE": "gprc_swe", "THA": "gprc_tha",
    "TUN": "gprc_tun", "TUR": "gprc_tur", "TWN": "gprc_twn",
    "UKR": "gprc_ukr", "USA": "gprc_usa", "VEN": "gprc_ven",
    "VNM": "gprc_vnm", "ZAF": "gprc_zaf",
}

# ─────────────────────────────────────────────────────────────
# STAGE 1 — LOAD DATA
# ─────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("STAGE 1 — Loading datasets")
print("="*55)

# ── Load GPR ─────────────────────────────────────────────────
gpr = pd.read_csv(GPR_FILE)
gpr.columns = gpr.columns.str.strip().str.lower()

# Strip metadata legend rows — real data rows have a numeric GPR value
gpr = gpr[pd.to_numeric(gpr["gpr"], errors="coerce").notna()].copy()
gpr.reset_index(drop=True, inplace=True)

# Parse month column — format is M/D/YYYY e.g. "1/1/1985"
gpr["date"]      = pd.to_datetime(gpr["month"], format="%m/%d/%Y")
gpr["year"]      = gpr["date"].dt.year
gpr["month_num"] = gpr["date"].dt.month

# Convert all data columns to numeric
numeric_cols = [c for c in gpr.columns
                if c not in ("month", "date", "var_name", "var_label")]
gpr[numeric_cols] = gpr[numeric_cols].apply(pd.to_numeric, errors="coerce")

# ── Load UNGDC ────────────────────────────────────────────────
ungdc = pd.read_csv(UNGDC_FILE)
ungdc.columns = ungdc.columns.str.strip().str.lower()
ungdc = ungdc.reset_index(drop=True)

# The CSV has both 'ccodealp' (ISO-3 code) and 'country' (full name).
# Drop the full-name column and rename the ISO code column to 'country'.
ungdc = ungdc.drop(columns=["country"], errors="ignore")
ungdc = ungdc.rename(columns={"ccodealp": "country"})

# Drop duplicates and filter
ungdc = ungdc.drop_duplicates().reset_index(drop=True)

# Keep only countries that have a GPR column and years after 1985
ungdc = ungdc[ungdc["country"].isin(GPR_COUNTRY_MAP.keys())].reset_index(drop=True)
ungdc = ungdc[ungdc["year"] >= 1986].reset_index(drop=True)

# Drop rows where text is missing or empty
ungdc = ungdc[ungdc["text"].notna() & (ungdc["text"].str.strip() != "")].reset_index(drop=True)

print(f"  GPR rows:          {len(gpr):,}  |  years: {gpr['year'].min()}–{gpr['year'].max()}")
print(f"  UNGDC speeches:    {len(ungdc):,}  |  countries: {ungdc['country'].nunique()}")
print(f"  UNGDC year range:  {ungdc['year'].min()}–{ungdc['year'].max()}")
print(f"\n  GPR sample (first 3 real rows):")
print(gpr[["date", "year", "month_num", "gpr", "gprc_rus", "gprc_usa"]].head(3).to_string())
print(f"\n  Countries found in UNGDC:")
print(ungdc["country"].value_counts().to_string())


# ─────────────────────────────────────────────────────────────
# STAGE 2 — spaCy NER SETUP
# ─────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("STAGE 2 — Loading spaCy + adding conflict patterns")
print("="*55)

nlp = spacy.load("en_core_web_lg")

ruler = nlp.add_pipe("entity_ruler", before="ner")
ruler.add_patterns([
    {"label": "ESCALATION", "pattern": [{"LOWER": {"IN": [
        "ultimatum", "retaliation", "provocation", "aggression",
        "invasion", "occupation", "annexation", "incursion", "hostilities"
    ]}}]},
    {"label": "MILITARY_OP", "pattern": [{"LOWER": {"IN": [
        "offensive", "mobilization", "deployment", "airstrike",
        "blockade", "ceasefire", "siege", "bombardment"
    ]}}]},
    {"label": "DIPLO_EVENT", "pattern": [{"LOWER": "peace"}, {"LOWER": "talks"}]},
    {"label": "DIPLO_EVENT", "pattern": [{"LOWER": "peace"}, {"LOWER": "negotiations"}]},
    {"label": "DIPLO_EVENT", "pattern": [{"LOWER": "sanctions"}]},
    {"label": "DIPLO_EVENT", "pattern": [{"LOWER": "veto"}]},
    {"label": "WEAPON", "pattern": [{"LOWER": {"IN": [
        "nuclear", "missile", "drone", "warhead", "bomb"
    ]}}]},
])

print(f"  Pipeline: {nlp.pipe_names}")

# Sanity check
test = "Iran's nuclear program triggered new sanctions after the ultimatum expired."
doc  = nlp(test)
print(f"\n  Test sentence: '{test}'")
print("  Entities found:")
for ent in doc.ents:
    print(f"    [{ent.label_:<14}]  {ent.text}")


# ─────────────────────────────────────────────────────────────
# STAGE 3 — EXTRACT ENTITIES FROM UN SPEECHES
# ─────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("STAGE 3 — Extracting entities from UN speeches")
print(f"  (Processing {SAMPLE_N} speeches — edit SAMPLE_N for full run)")
print("="*55)

ungdc = ungdc.reset_index(drop=True)
sample_df = ungdc.sample(
    n=min(SAMPLE_N, len(ungdc)), random_state=42
).reset_index(drop=True)

records = []
for i, row in sample_df.iterrows():
    if i % 50 == 0:
        print(f"  Processing speech {i+1}/{len(sample_df)}  "
              f"({row['country']} {row['year']})")

    text = row["text"][:5000]
    doc  = nlp(text)

    for ent in doc.ents:
        records.append({
            "country":   row["country"],
            "year":      row["year"],
            "ent_text":  ent.text,
            "label":     ent.label_,
            "context":   text[max(0, ent.start_char - 80): ent.end_char + 80],
            "sent_text": ent.sent.text,
        })

if not records:
    raise RuntimeError(
        "No entities extracted. Check that 'text' column exists and is not empty.\n"
        f"Columns found: {sample_df.columns.tolist()}"
    )

entities = pd.DataFrame(records)

CONFLICT_LABELS = {"ESCALATION", "MILITARY_OP", "WEAPON", "DIPLO_EVENT", "GPE"}

print(f"\n  Total entities extracted: {len(entities):,}")
print(f"\n  Entity label counts:")
print(entities["label"].value_counts().to_string())
print(f"\n  Sample conflict-related entities:")
print(entities[entities["label"].isin(CONFLICT_LABELS)]
      [["country", "year", "label", "ent_text"]].head(15).to_string())


# ─────────────────────────────────────────────────────────────
# STAGE 4 — FEATURE EXTRACTION PER COUNTRY-YEAR
# ─────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("STAGE 4 — Building feature matrix")
print("="*55)

POS_SEEDS = {"peace", "cooperation", "dialogue", "agreement",
             "resolve", "stability", "diplomacy", "treaty"}
NEG_SEEDS = {"war", "conflict", "attack", "threat", "violence",
             "aggression", "kill", "destroy", "invade", "terror",
             "massacre", "genocide", "bomb", "missile"}


def get_features(country, year, ents_df):
    curr = ents_df[(ents_df["country"] == country) & (ents_df["year"] == year)]
    prev = ents_df[(ents_df["country"] == country) & (ents_df["year"] == year - 1)]

    def conflict_ratio(df):
        return df["label"].isin(CONFLICT_LABELS).sum() / max(len(df), 1)

    pos = neg = 0
    for ctx in curr[curr["label"].isin(CONFLICT_LABELS)]["context"]:
        toks = set(ctx.lower().split())
        pos  += len(toks & POS_SEEDS)
        neg  += len(toks & NEG_SEEDS)

    sents_gpe = set(curr[curr["label"] == "GPE"]["sent_text"])
    sents_esc = set(curr[curr["label"].isin(
                    {"ESCALATION", "MILITARY_OP", "WEAPON"})]["sent_text"])
    cooc = len(sents_gpe & sents_esc)

    ratio_curr = conflict_ratio(curr)
    ratio_prev = conflict_ratio(prev)

    return {
        "conflict_ratio_curr":  ratio_curr,
        "conflict_ratio_delta": ratio_curr - ratio_prev,
        "sentiment_score":      (pos - neg) / max(pos + neg, 1),
        "neg_word_count":       neg,
        "pos_word_count":       pos,
        "hostile_cooccurrence": cooc,
        "total_entities":       len(curr),
    }


feature_rows = []
for (country, year), _ in sample_df.groupby(["country", "year"]):
    gpr_col = GPR_COUNTRY_MAP.get(country)
    if not gpr_col or gpr_col not in gpr.columns:
        continue

    mask = (
        ((gpr["year"] == year)     & (gpr["month_num"] >= 10)) |
        ((gpr["year"] == year + 1) & (gpr["month_num"] <= 9))
    )
    gpr_fwd    = gpr.loc[mask, gpr_col].mean()
    gpr_global = gpr.loc[mask, "gpr"].mean()

    if pd.isna(gpr_fwd):
        continue

    feats = get_features(country, year, entities)
    feature_rows.append({
        "country":     country,
        "year":        year,
        **feats,
        "gpr_forward": gpr_fwd,
        "gpr_global":  gpr_global,
    })

features_df = pd.DataFrame(feature_rows).dropna()

print(f"  Feature matrix shape: {features_df.shape}")
print(f"\n  Preview:")
print(features_df[["country", "year", "conflict_ratio_curr",
                    "sentiment_score", "hostile_cooccurrence",
                    "gpr_forward"]].head(10).to_string())


# ─────────────────────────────────────────────────────────────
# STAGE 5 — TRAIN CLASSIFIER
# ─────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("STAGE 5 — Training conflict-risk classifier")
print("="*55)

features_df["gpr_spike"] = (
    features_df["gpr_forward"] >
    features_df.groupby("country")["gpr_forward"].transform("quantile", 0.75)
).astype(int)

print(f"  Class distribution:")
print(features_df["gpr_spike"].value_counts().to_string())

FEATURE_COLS = [
    "conflict_ratio_curr",
    "conflict_ratio_delta",
    "sentiment_score",
    "neg_word_count",
    "pos_word_count",
    "hostile_cooccurrence",
    "total_entities",
]

X = features_df[FEATURE_COLS].fillna(0)
y = features_df["gpr_spike"]

importances = pd.Series(dtype=float)

if len(X) < 20:
    print("\n  ⚠ Too few samples for a meaningful train/test split.")
    print("  Increase SAMPLE_N to at least 500 and re-run.")
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    clf = GradientBoostingClassifier(
        n_estimators=100, max_depth=3,
        learning_rate=0.1, subsample=0.8,
        random_state=42
    )
    clf.fit(X_train, y_train)

    print("\n  Classification report:")
    print(classification_report(y_test, clf.predict(X_test)))

    importances = pd.Series(clf.feature_importances_, index=FEATURE_COLS)
    print("\n  Feature importances (highest → lowest):")
    print(importances.sort_values(ascending=False).to_string())


# ─────────────────────────────────────────────────────────────
# STAGE 6 — VISUALISATIONS (one separate figure per plot)
# ─────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("STAGE 6 — Generating plots (6 separate windows)")
print("="*55)

conflict_label_set = {"ESCALATION", "MILITARY_OP", "WEAPON", "DIPLO_EVENT"}

# ── Plot 1: Entity label distribution ────────────────────────
fig1, ax1 = plt.subplots(figsize=(11, 7))
fig1.canvas.manager.set_window_title("Plot 1 — Entity Types")
label_counts = entities["label"].value_counts().head(12)
bar_colors   = ["#c0392b" if l in conflict_label_set else "#2980b9"
                for l in label_counts.index[::-1]]
ax1.barh(label_counts.index[::-1], label_counts.values[::-1], color=bar_colors,
         edgecolor="white", linewidth=0.5)
for i, v in enumerate(label_counts.values[::-1]):
    ax1.text(v + max(label_counts.values) * 0.01, i, str(v),
             va="center", fontsize=10, color="#222222")
ax1.set_title("Entity types extracted from UN speeches\n"
              "(red bars = conflict-domain custom labels)", fontsize=13, pad=16)
ax1.set_xlabel("Count", fontsize=11)
ax1.tick_params(labelsize=11)
ax1.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("plot1_entity_types.png", dpi=150, bbox_inches="tight")
print("  Saved: plot1_entity_types.png")

# ── Plot 2: Global GPR over time ──────────────────────────────
fig2, ax2 = plt.subplots(figsize=(13, 5))
fig2.canvas.manager.set_window_title("Plot 2 — Global GPR Over Time")
gpr_annual = gpr.groupby("year")["gpr"].mean()
ax2.fill_between(gpr_annual.index, gpr_annual.values, alpha=0.2, color="#e74c3c")
ax2.plot(gpr_annual.index, gpr_annual.values, color="#c0392b", linewidth=2)
top_years = gpr_annual.nlargest(3)
for yr, val in top_years.items():
    ax2.annotate(str(int(yr)),
                 xy=(yr, val), xytext=(yr + 0.5, val + 4),
                 fontsize=9, color="#c0392b",
                 arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.8))
ax2.set_title("Global Geopolitical Risk (GPR) index — annual mean", fontsize=13, pad=16)
ax2.set_xlabel("Year", fontsize=11)
ax2.set_ylabel("GPR", fontsize=11)
ax2.tick_params(labelsize=11)
ax2.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("plot2_global_gpr.png", dpi=150, bbox_inches="tight")
print("  Saved: plot2_global_gpr.png")

# ── Plot 3: Conflict label ratio by country ───────────────────
if not features_df.empty:
    country_means = (features_df.groupby("country")["conflict_ratio_curr"]
                                .mean().sort_values(ascending=True))
    # Scale figure height to number of countries so labels never overlap
    fig_h = max(7, len(country_means) * 0.5)
    fig3, ax3 = plt.subplots(figsize=(11, fig_h))
    fig3.canvas.manager.set_window_title("Plot 3 — Conflict Ratio by Country")
    bars = ax3.barh(country_means.index, country_means.values,
                    color="#8e44ad", alpha=0.82,
                    edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, country_means.values):
        ax3.text(val + max(country_means.values) * 0.01,
                 bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va="center", fontsize=9)
    ax3.set_title("Average conflict-entity ratio by country\n"
                  "(proportion of extracted entities that are conflict-related)",
                  fontsize=13, pad=16)
    ax3.set_xlabel("Conflict entity ratio", fontsize=11)
    ax3.tick_params(labelsize=11)
    ax3.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("plot3_country_conflict_ratio.png", dpi=150, bbox_inches="tight")
    print("  Saved: plot3_country_conflict_ratio.png")

# ── Plot 4: Sentiment score vs forward GPR ────────────────────
if not features_df.empty:
    fig4, ax4 = plt.subplots(figsize=(10, 7))
    fig4.canvas.manager.set_window_title("Plot 4 — Sentiment vs Forward GPR")
    scatter = ax4.scatter(
        features_df["sentiment_score"],
        features_df["gpr_forward"],
        c=features_df["gpr_spike"],
        cmap="RdYlGn_r", alpha=0.75,
        edgecolors="k", linewidths=0.4, s=90
    )
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label("GPR spike (1 = yes, 0 = no)", fontsize=10)
    for _, row in features_df.iterrows():
        ax4.annotate(row["country"],
                     (row["sentiment_score"], row["gpr_forward"]),
                     fontsize=7, alpha=0.65,
                     xytext=(3, 3), textcoords="offset points")
    ax4.set_xlabel("Sentiment score  (positive = peaceful language)", fontsize=11)
    ax4.set_ylabel("Country GPR — forward 12 months after speech", fontsize=11)
    ax4.set_title("UN speech sentiment vs geopolitical risk in following year\n"
                  "(red = GPR spike, green = no spike)", fontsize=13, pad=16)
    ax4.tick_params(labelsize=11)
    ax4.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("plot4_sentiment_vs_gpr.png", dpi=150, bbox_inches="tight")
    print("  Saved: plot4_sentiment_vs_gpr.png")

# ── Plot 5: Feature importance ────────────────────────────────
fig5, ax5 = plt.subplots(figsize=(10, 6))
fig5.canvas.manager.set_window_title("Plot 5 — Feature Importance")
if len(X) >= 20 and not importances.empty:
    imp    = importances.sort_values()
    colors = ["#27ae60" if v >= imp.median() else "#95a5a6" for v in imp.values]
    bars   = ax5.barh(imp.index, imp.values, color=colors,
                      edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, imp.values):
        ax5.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va="center", fontsize=10)
    ax5.set_title("Feature importance — Gradient Boosting classifier\n"
                  "(green = above median importance)", fontsize=13, pad=16)
    ax5.set_xlabel("Importance score", fontsize=11)
    ax5.tick_params(labelsize=11)
    ax5.spines[["top", "right"]].set_visible(False)
else:
    ax5.text(0.5, 0.5,
             "Classifier not trained\n(increase SAMPLE_N to 500+ and re-run)",
             ha="center", va="center", transform=ax5.transAxes, fontsize=12)
    ax5.set_title("Feature importance — Gradient Boosting classifier",
                  fontsize=13, pad=16)
plt.tight_layout()
plt.savefig("plot5_feature_importance.png", dpi=150, bbox_inches="tight")
print("  Saved: plot5_feature_importance.png")

# ── Plot 6: Country GPR over time ─────────────────────────────
fig6, ax6 = plt.subplots(figsize=(13, 6))
fig6.canvas.manager.set_window_title("Plot 6 — Country GPR Over Time")
plot_countries = {
    "RUS": "#e74c3c", "CHN": "#f39c12",
    "ISR": "#2ecc71", "USA": "#3498db",
    "TUR": "#9b59b6", "IND": "#1abc9c",
}
for ctry, color in plot_countries.items():
    col = GPR_COUNTRY_MAP.get(ctry)
    if col and col in gpr.columns:
        annual = gpr.groupby("year")[col].mean().dropna()
        ax6.plot(annual.index, annual.values,
                 label=ctry, color=color, linewidth=2, alpha=0.85)
ax6.set_title("Country-level Geopolitical Risk over time\n"
              "(selected countries)", fontsize=13, pad=16)
ax6.set_xlabel("Year", fontsize=11)
ax6.set_ylabel("GPR (annual mean)", fontsize=11)
ax6.legend(fontsize=10, loc="upper left", framealpha=0.7)
ax6.tick_params(labelsize=11)
ax6.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("plot6_country_gpr_over_time.png", dpi=150, bbox_inches="tight")
print("  Saved: plot6_country_gpr_over_time.png")

# Open all 6 windows at once
plt.show()

print("\n" + "="*55)
print("DONE")
print("  6 plots saved as individual PNG files in your project folder")
print("="*55)