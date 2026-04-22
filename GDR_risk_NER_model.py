# ============================================================
# GDR_risk_NER_model_v4.py
# Fixes: class imbalance, lag modeling, weak features removal
# ============================================================

import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, f1_score

from scipy.stats import pearsonr

# ── CONFIG ───────────────────────────────────────────────────
GPR_FILE   = "data_gpr_export.csv"
UNGDC_FILE = "UNGDC_1946-2023.csv"
SAMPLE_N = 300

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
print("\n=== STAGE 1 — Loading datasets ===")

gpr = pd.read_csv(GPR_FILE)
gpr.columns = gpr.columns.str.strip().str.lower()
gpr = gpr[pd.to_numeric(gpr["gpr"], errors="coerce").notna()]
gpr["date"] = pd.to_datetime(gpr["month"])
gpr["year"] = gpr["date"].dt.year

ungdc = pd.read_csv(UNGDC_FILE)
ungdc.columns = ungdc.columns.str.strip().str.lower()
ungdc = ungdc.rename(columns={"ccodealp": "country"})
ungdc = ungdc[ungdc["text"].notna()]
ungdc = ungdc.loc[:, ~ungdc.columns.duplicated()].copy()

# ─────────────────────────────────────────────────────────────
# NER MODEL
# ─────────────────────────────────────────────────────────────
print("\n=== STAGE 2 — spaCy NER ===")

nlp = spacy.load("en_core_web_lg")

ruler = nlp.add_pipe("entity_ruler", before="ner")
ruler.add_patterns([
    {"label": "ESCALATION", "pattern": [{"LOWER": {"IN": ["invasion","aggression","conflict"]}}]},
    {"label": "MILITARY_OP", "pattern": [{"LOWER": {"IN": ["deployment","strike","bombardment"]}}]},
    {"label": "WEAPON", "pattern": [{"LOWER": {"IN": ["nuclear","missile","bomb"]}}]},
])

# ─────────────────────────────────────────────────────────────
# ENTITY EXTRACTION
# ─────────────────────────────────────────────────────────────
print("\n=== STAGE 3 — Extracting entities ===")

sample_df = ungdc.sample(n=min(SAMPLE_N, len(ungdc)), random_state=42).copy()

records = []
for _, row in sample_df.iterrows():
    doc = nlp(row["text"][:3000])

    for ent in doc.ents:
        records.append({
            "country": row["country"],
            "year": row["year"],
            "label": ent.label_,
            "text": row["text"]
        })

entities = pd.DataFrame(records)

# ─────────────────────────────────────────────────────────────
# FEATURES (SIMPLIFIED + FIXED)
# ─────────────────────────────────────────────────────────────

CONFLICT_LABELS = {"ESCALATION","MILITARY_OP","WEAPON","GPE"}

def ratio(df):
    return df.label.isin(CONFLICT_LABELS).sum() / max(len(df), 1)

def get_features(country, year):
    curr = entities[(entities.country==country)&(entities.year==year)]
    prev = entities[(entities.country==country)&(entities.year==year-1)]
    prev2 = entities[(entities.country==country)&(entities.year==year-2)]

    return {
        "conflict_ratio_curr": ratio(curr),
        "conflict_ratio_delta": ratio(curr) - ratio(prev),
        "conflict_ratio_lag2": ratio(curr) - ratio(prev2),
        "total_entities": len(curr)
    }

# ─────────────────────────────────────────────────────────────
# BUILD DATASET
# ─────────────────────────────────────────────────────────────
print("\n=== STAGE 4 — Building dataset ===")

rows = []

for (country, year), _ in sample_df.groupby(["country","year"]):
    feats = get_features(country, year)

    gpr_val = gpr[gpr["year"]==year+1]["gpr"].mean()

    rows.append({
        "country": country,
        "year": year,
        **feats,
        "gpr_forward": gpr_val
    })

features_df = pd.DataFrame(rows).dropna()
features_df = features_df.sort_values(["country","year"])

# ─────────────────────────────────────────────────────────────
# LAG FEATURE ENGINEERING (IMPORTANT FIX)
# ─────────────────────────────────────────────────────────────
print("\n=== STAGE 5 — Adding lag features ===")

features_df["lag_1"] = features_df.groupby("country")["conflict_ratio_curr"].shift(1)
features_df["lag_2"] = features_df.groupby("country")["conflict_ratio_curr"].shift(2)

features_df = features_df.dropna()

# ─────────────────────────────────────────────────────────────
# CORRELATION CHECK (sanity)
# ─────────────────────────────────────────────────────────────
print("\n=== STAGE 6 — Correlations ===")

for col in ["conflict_ratio_curr","lag_1","lag_2"]:
    r, p = pearsonr(features_df[col], features_df["gpr_forward"])
    print(f"{col}: r={r:.4f}, p={p:.4f}")

# ─────────────────────────────────────────────────────────────
# CLASSIFICATION TARGET
# ─────────────────────────────────────────────────────────────
features_df["gpr_spike"] = (
    features_df["gpr_forward"] >
    features_df["gpr_forward"].quantile(0.75)
).astype(int)

# ─────────────────────────────────────────────────────────────
# MODEL TRAINING (FIXED IMBALANCE)
# ─────────────────────────────────────────────────────────────
print("\n=== STAGE 7 — Model Training ===")

feature_cols = [
    "conflict_ratio_curr",
    "conflict_ratio_delta",
    "conflict_ratio_lag2",
    "lag_1",
    "lag_2",
    "total_entities"
]

X = features_df[feature_cols]
y = features_df["gpr_spike"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)

# handle imbalance properly
clf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42
)

clf.fit(X_train, y_train)
pred = clf.predict(X_test)

# ─────────────────────────────────────────────────────────────
# EVALUATION (FIXED METRICS)
# ─────────────────────────────────────────────────────────────
print("\n=== STAGE 8 — Evaluation ===")

print(classification_report(y_test, pred))

if len(np.unique(y_test)) > 1:
    print("ROC-AUC:", roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]))

print("F1 (class 1):", f1_score(y_test, pred))

# ─────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────
print("\n=== STAGE 9 — Plot ===")

plt.scatter(features_df["conflict_ratio_curr"], features_df["gpr_forward"])
plt.xlabel("Conflict Ratio")
plt.ylabel("Future GPR")
plt.title("Geopolitical Language vs Risk")
plt.show()

print("\n=== DONE ===")