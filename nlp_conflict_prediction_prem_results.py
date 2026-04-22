#!/usr/bin/env python3
"""
NLP-based Conflict Prediction: Preliminary Analysis
Uses both UNGDC speeches and structured geopolitical conflict data
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Download resources
print("Downloading NLTK resources...")
nltk.download('vader_lexicon', quiet=True)

# ============================================================================
# PART 1: UNGDC ANALYSIS - Sentiment and Threat Keywords
# ============================================================================
print("\n" + "="*70)
print("PART 1: UN GENERAL DEBATE CORPUS (UNGDC) ANALYSIS")
print("="*70)

print("\nLoading UNGDC dataset...")
UNGDC_CSV = "UNGDC_1946-2023.csv"

if not os.path.exists(UNGDC_CSV):
    print(f"ERROR: Dataset {UNGDC_CSV} not found in the current directory.")
    sys.exit(1)

ungdc = pd.read_csv(UNGDC_CSV, low_memory=False)
ungdc = ungdc.dropna(subset=['country', 'year', 'text'])
ungdc['year'] = pd.to_numeric(ungdc['year'], errors='coerce')
ungdc = ungdc.dropna(subset=['year'])
ungdc['year'] = ungdc['year'].astype(int)

# The UNGDC file contains both full country names and ISO-3 country codes.
# Use ISO-3 where available so it can align with the conflict dataset.
if 'ccodealp' in ungdc.columns:
  ungdc['country_code'] = ungdc['ccodealp'].astype(str).str.upper().str.strip()
else:
  ungdc['country_code'] = ungdc['country'].astype(str).str.upper().str.strip()

print(f"Loaded {len(ungdc)} speeches from {ungdc['country'].nunique()} countries")
print(f"Date range: {ungdc['year'].min()}-{ungdc['year'].max()}")

# Sentiment analysis using VADER
print("Computing sentiment...")
sia = SentimentIntensityAnalyzer()

def get_sentiment(text):
    try:
        return sia.polarity_scores(str(text))['compound']
    except Exception:
        return 0.0

ungdc['sentiment'] = ungdc['text'].apply(get_sentiment)

# Threat keyword detection
print("Counting threat keywords...")
THREAT_KEYWORDS = {
    'weapon', 'attack', 'military', 'bomb', 'terrorist', 'rebel', 'war', 
    'conflict', 'army', 'violence', 'security', 'hostility', 'aggression', 
    'threat', 'crisis', 'danger', 'missile', 'nuclear', 'siege', 'invasion'
}

def count_threat_keywords(text):
    try:
        text_lower = str(text).lower()
        return sum(text_lower.count(keyword) for keyword in THREAT_KEYWORDS)
    except Exception:
        return 0

ungdc['threat_count'] = ungdc['text'].apply(count_threat_keywords)

# Aggregate by year
print("Aggregating data...")
ungdc_agg = ungdc.groupby(['country', 'year']).agg({
    'sentiment': ['mean', 'std', 'count'],
    'threat_count': 'sum'
}).reset_index()

ungdc_agg.columns = ['country', 'year', 'sentiment_mean', 'sentiment_std', 'speech_count', 'threat_count']

print(f"Aggregated: {len(ungdc_agg)} country-year records")
print(f"Total threat mentions: {ungdc_agg['threat_count'].sum()}")

# ============================================================================
# PART 2: STRUCTURED DATASET - Escalation and Sentiment Correlation
# ============================================================================
print("\n" + "="*70)
print("PART 2: GEOPOLITICAL CONFLICT DATASET ANALYSIS")
print("="*70)

print("\nLoading conflict dataset...")
DATA_CSV = "geopolitical_conflict_risk_dataset.csv"

if not os.path.exists(DATA_CSV):
    print(f"ERROR: Dataset {DATA_CSV} not found in the current directory.")
    sys.exit(1)

df = pd.read_csv(DATA_CSV)
df = df.dropna(subset=['country', 'region', 'month', 'social_media_sentiment', 'conflict_escalation_6m'])
df['country'] = df['country'].astype(str).str.upper().str.strip()
df['month'] = pd.to_datetime(df['month'], format='%Y-%m', errors='coerce')
df = df.dropna(subset=['month'])
df = df.sort_values('month')

print(f"Loaded {len(df)} records from {df['month'].min().date()} to {df['month'].max().date()}")
print(f"Regions: {', '.join(sorted(df['region'].unique()))}")

# Sentiment by region
region_sentiment = df.groupby('region')['social_media_sentiment'].mean().sort_values()

# Correlation
correlation = df['social_media_sentiment'].corr(df['conflict_escalation_6m'])
print(f"Sentiment-escalation correlation: {correlation:.3f}")

# ============================================================================
# PART 3: COMBINED ANALYSIS AND CLASSIFICATION
# ============================================================================
print("\n" + "="*70)
print("PART 3: CLASSIFICATION AND PREDICTIVE MODELING")
print("="*70)

# Prepare features
print("\nPreparing features...")

# Build a clean country-month modeling table from conflict data.
conflict_model = (
  df[['country', 'region', 'month', 'social_media_sentiment', 'conflict_escalation_6m']]
  .drop_duplicates(subset=['country', 'month'])
  .copy()
)
conflict_model['year'] = conflict_model['month'].dt.year

# Aggregate UNGDC sentiment by country-year.
ungdc_country_year = (
  ungdc.groupby(['country_code', 'year'], as_index=False)
  .agg(ungdc_sentiment=('sentiment', 'mean'))
  .rename(columns={'country_code': 'country'})
)

# Global yearly fallback for countries/years missing in UNGDC country-level data.
ungdc_global_year = (
  ungdc.groupby('year', as_index=False)
  .agg(ungdc_global_sentiment=('sentiment', 'mean'))
)

# For each country-month conflict record, attach the most recent available
# UNGDC country-year sentiment (backward as-of merge).
conflict_model = conflict_model.sort_values('year')
ungdc_country_year = ungdc_country_year.sort_values('year')

conflict_model['year'] = conflict_model['year'].astype('int64')
ungdc_country_year['year'] = ungdc_country_year['year'].astype('int64')
features = pd.merge_asof(
  conflict_model,
  ungdc_country_year,
  on='year',
  by='country',
  direction='backward',
)
features['year'] = features['year'].astype('int64')
ungdc_global_year['year'] = ungdc_global_year['year'].astype('int64')

# Fill remaining gaps with global UN sentiment trend by year.
features = pd.merge_asof(
  features.sort_values('year'),
  ungdc_global_year.sort_values('year'),
  on='year',
  direction='backward',
)
features['ungdc_sentiment'] = features['ungdc_sentiment'].fillna(features['ungdc_global_sentiment'])

# Logistic-regression baseline with sentiment as the only feature.
# This combines sentiment signals from both datasets into one scalar feature.
features['sentiment'] = features[['social_media_sentiment', 'ungdc_sentiment']].mean(axis=1)
features = features.dropna(subset=['sentiment', 'conflict_escalation_6m'])
features = features.rename(columns={'conflict_escalation_6m': 'escalation'})
features['escalation'] = features['escalation'].astype(int)

print(f"Features: {len(features)} country-month records")
print(f"Countries matched to UNGDC sentiment: {features['country'].nunique()}")

# Train model
X = features[['sentiment']].values
y = features['escalation'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
preds = model.predict(X_test)

print(f"Training accuracy: {train_acc:.3f}")
print(f"Testing accuracy: {test_acc:.3f}")

# Compute confusion matrix and metrics
cm = confusion_matrix(y_test, preds, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

print(f"\nConfusion Matrix:")
print(f"  True Negatives: {tn}  |  False Positives: {fp}")
print(f"  False Negatives: {fn}  |  True Positives: {tp}")
print("\nClassification Report:")
print(classification_report(y_test, preds, digits=3))

# ============================================================================
# PART 4: VISUALIZATIONS
# ============================================================================
print("\nGenerating visualizations...")

fig = plt.figure(figsize=(16, 12))

# 1. UNGDC Sentiment Trend
ax1 = plt.subplot(3, 3, 1)
yearly_data = ungdc_agg.groupby('year')[['sentiment_mean']].mean()
ax1.plot(yearly_data.index, yearly_data['sentiment_mean'], linewidth=2, color='steelblue')
ax1.fill_between(yearly_data.index, yearly_data['sentiment_mean'], alpha=0.3, color='steelblue')
ax1.set_title('UNGDC Sentiment Trend Over Time', fontsize=10, fontweight='bold')
ax1.set_xlabel('Year')
ax1.set_ylabel('Avg Sentiment')
ax1.grid(True, alpha=0.3)

# 2. UNGDC Threat Keywords Trend
ax2 = plt.subplot(3, 3, 2)
yearly_threats = ungdc_agg.groupby('year')[['threat_count']].sum()
ax2.bar(yearly_threats.index, yearly_threats['threat_count'], color='coral', alpha=0.7, width=0.6)
ax2.set_title('Threat Keywords in UNGDC Speeches', fontsize=10, fontweight='bold')
ax2.set_xlabel('Year')
ax2.set_ylabel('Total Mentions')
ax2.grid(True, alpha=0.3, axis='y')

# 3. Regional Sentiment Comparison
ax3 = plt.subplot(3, 3, 3)
region_sentiment.plot(kind='barh', ax=ax3, color='teal', alpha=0.7)
ax3.set_title('Average Sentiment by Region', fontsize=10, fontweight='bold')
ax3.set_xlabel('Sentiment Score')
ax3.grid(True, alpha=0.3, axis='x')

# 4. Sentiment Over Time (Structured Data)
ax4 = plt.subplot(3, 3, 4)
monthly_sent = df.groupby('month')['social_media_sentiment'].mean()
ax4.plot(monthly_sent.index, monthly_sent.values, linewidth=1.5, color='darkgreen')
ax4.fill_between(monthly_sent.index, monthly_sent.values, alpha=0.2, color='darkgreen')
ax4.set_title('Monthly Sentiment Trend', fontsize=10, fontweight='bold')
ax4.set_xlabel('Month')
ax4.set_ylabel('Sentiment')
ax4.grid(True, alpha=0.3)
ax4.tick_params(axis='x', rotation=45)

# 5. Sentiment Distribution by Escalation
ax5 = plt.subplot(3, 3, 5)
no_esc = df[df['conflict_escalation_6m'] == 0]['social_media_sentiment']
esc = df[df['conflict_escalation_6m'] == 1]['social_media_sentiment']
ax5.hist([no_esc, esc], bins=30, label=['No Escalation', 'Escalation'], 
         color=['skyblue', 'salmon'], alpha=0.7)
ax5.set_title('Sentiment Distribution by Escalation', fontsize=10, fontweight='bold')
ax5.set_xlabel('Sentiment Score')
ax5.set_ylabel('Frequency')
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# 6. Escalation by Region
ax6 = plt.subplot(3, 3, 6)
region_escalation = df.groupby('region')['conflict_escalation_6m'].mean().sort_values()
region_escalation.plot(kind='barh', ax=ax6, color='darkred', alpha=0.7)
ax6.set_title('Escalation Rate by Region', fontsize=10, fontweight='bold')
ax6.set_xlabel('Escalation Proportion')
ax6.grid(True, alpha=0.3, axis='x')

# 7. Confusion Matrix Heatmap
ax7 = plt.subplot(3, 3, 7)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
im = ax7.imshow(cm_normalized, cmap='Blues', aspect='auto')
ax7.set_xticks([0, 1])
ax7.set_yticks([0, 1])
ax7.set_xticklabels(['No Escalation', 'Escalation'])
ax7.set_yticklabels(['No Escalation', 'Escalation'])
ax7.set_title('Confusion Matrix (Normalized)', fontsize=10, fontweight='bold')
for i in range(2):
    for j in range(2):
        text = ax7.text(j, i, f'{cm_normalized[i, j]:.2f}',
                       ha="center", va="center", color="black", fontweight='bold')
plt.colorbar(im, ax=ax7)

# 8. Model Accuracy Comparison
ax8 = plt.subplot(3, 3, 8)
accuracies = [train_acc, test_acc, 0.5]
labels = ['Train', 'Test', 'Random']
colors = ['green', 'orange', 'gray']
ax8.bar(labels, accuracies, color=colors, alpha=0.7)
ax8.set_title('Model Performance', fontsize=10, fontweight='bold')
ax8.set_ylabel('Accuracy')
ax8.set_ylim([0, 1])
ax8.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random Baseline')
ax8.grid(True, alpha=0.3, axis='y')
ax8.legend()

# 9. Escalation Timeline
ax9 = plt.subplot(3, 3, 9)
escalation_timeline = df.groupby('month')['conflict_escalation_6m'].sum()
ax9.scatter(escalation_timeline.index, escalation_timeline.values, 
           alpha=0.6, s=50, color='purple')
ax9.plot(escalation_timeline.index, escalation_timeline.values, alpha=0.3, color='purple')
ax9.set_title('Escalation Events Over Time', fontsize=10, fontweight='bold')
ax9.set_xlabel('Month')
ax9.set_ylabel('# Escalation Events')
ax9.grid(True, alpha=0.3)
ax9.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('conflict_analysis_results.png', dpi=300, bbox_inches='tight')
print("✓ Saved: conflict_analysis_results.png")

# ============================================================================
# PART 5: SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*70)
print("ANALYSIS SUMMARY")
print("="*70)

print(f"""
DATASETS:
  UNGDC: {len(ungdc):,} speeches, {ungdc['year'].min()}-{ungdc['year'].max()}
  Conflict: {len(df):,} monthly records, {df['region'].nunique()} regions

SENTIMENT:
  Overall mean: {df['social_media_sentiment'].mean():.3f}
  Correlation w/ escalation: {correlation:.3f}
  Regions with highest sentiment: {region_sentiment.idxmax()}
  Regions with lowest sentiment: {region_sentiment.idxmin()}

THREATS:
  Total keyword mentions: {int(ungdc_agg['threat_count'].sum()):,}
  Peak year: {ungdc_agg.groupby('year')['threat_count'].sum().idxmax()}

ESCALATION:
  No escalation: {(df['conflict_escalation_6m']==0).sum()} ({100*(df['conflict_escalation_6m']==0).mean():.1f}%)
  With escalation: {(df['conflict_escalation_6m']==1).sum()} ({100*(df['conflict_escalation_6m']==1).mean():.1f}%)

MODEL:
  Training accuracy: {train_acc:.1%}
  Testing accuracy: {test_acc:.1%}
  Model coefficient: {model.coef_[0][0]:.3f}
  
NEXT STEPS:
  • Add temporal lag features (t-1, t-2)
  • Include threat frequency from UNGDC
  • Try ensemble methods (Random Forest, XGBoost)
  • Improve feature engineering
  • Validate on held-out events
""")

print("="*70)
print("Analysis complete! Check 'conflict_analysis_results.png'")
print("="*70)
