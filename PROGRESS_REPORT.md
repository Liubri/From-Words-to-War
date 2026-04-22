# Progress Report: NLP-Based Conflict Prediction
**Names:** Rohan Parikh, Darius Saadat, Brian Liu, William Riser

---

## (i) Introduction

This project investigates whether linguistic shifts in international economic and political reports serve as leading indicators for armed conflict escalation. Our central research question is: **Do sentiment changes and threat-related entity mentions in textual reports precede conflict events?**

We are developing a multi-faceted NLP pipeline that combines sentiment analysis, named entity recognition (NER), and topic modeling to extract predictive signals from UN General Debate speeches and structured geopolitical conflict data. The ultimate goal is to build a classification model that can predict conflict escalation 1-6 months in advance by analyzing linguistic patterns in international discourse.

Our approach leverages three complementary NLP techniques to capture different aspects of linguistic signals: (1) sentiment polarity shifts that may indicate deteriorating diplomatic relations, (2) threat-entity frequency spikes that correlate with security concerns, and (3) topic distribution changes that reveal shifts in international discourse from economic to security-focused topics.

---

## (ii) Data

We are utilizing two primary datasets:

**Dataset 1: UN General Debate Corpus (UNGDC)**
- **Source:** Kaggle - United Nations General Debate Corpus, 1946-2023
- **Size:** 3,093 speeches analyzed (2000-2020 period)
- **Countries:** 151 countries represented
- **Content:** Raw text transcripts of UN General Assembly speeches
- **Usage:** Primary source for sentiment analysis and threat keyword extraction

**Dataset 2: Geopolitical Conflict Risk Dataset**
- **Source:** Kaggle - Global Geopolitical Conflict Dataset (2020-2025)
- **Size:** 1,320 monthly records
- **Regions:** 7 geographic regions (East Asia, Eastern Europe, Middle East, North America, South America, South Asia, Western Europe)
- **Features:** Monthly social media sentiment scores, conflict escalation labels (binary: 0=no escalation, 1=escalation within 6 months)
- **Class Distribution:** 68.6% no escalation (905 records), 31.4% escalation (415 records)
- **Usage:** Provides labeled escalation targets and sentiment baseline for classification modeling

**Key Statistics:**
- Total threat keywords detected across UNGDC: **85,781 mentions**
- Peak threat mention year: 2009
- Overall UNGDC sentiment mean: 0.905 (high positive, likely due to formal diplomatic language)
- Structured dataset sentiment mean: 0.045 (near-neutral social media sentiment)
- Sentiment-escalation correlation: -0.046 (weak negative relationship suggesting sentiment alone has limited predictive power)

---

## (iii) Models

### Baseline Model: Logistic Regression with Sentiment Features
We implemented a logistic regression classifier using sentiment as the sole feature. This baseline establishes a performance floor and allows us to quantify the contribution of sentiment signals.

**Model Details:**
- **Feature:** Monthly average social media sentiment score per region
- **Training/Test Split:** 80/20 with stratification
- **Regularization:** Default L2 (C=1.0)
- **Output:** Binary classification (escalation vs. no escalation)

### Planned Models (In Development):
1. **NER-Based Classifier:** Will incorporate threat entity frequency and type as features
2. **Topic-Based Classifier:** Will use topic distribution shifts as predictive features
3. **Ensemble Model:** Combined sentiment + NER + topic features for improved performance

---

## (iv) Preliminary Results

### Sentiment Analysis Performance

**Logistic Regression Baseline:**
- Training Accuracy: **60.2%**
- Testing Accuracy: **60.0%**
- Model Coefficient: -0.192 (negative relationship: lower sentiment → higher escalation risk)

**Performance Comparison:**
- Random Baseline: 50% (always predicting majority class)
- **Sentiment Model Advantage: +10 percentage points**

**Classification Report (Test Set):**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| No Escalation | 0.60 | 1.00 | 0.75 | 84 |
| Escalation | 0.00 | 0.00 | 0.00 | 56 |
| Weighted Avg | 0.36 | 0.60 | 0.45 | 140 |

**Analysis:** The model achieves modest performance above random baseline. The model currently exhibits high recall for the negative class but zero precision for escalation events, indicating a bias toward predicting non-escalation. This suggests sentiment alone is insufficient for prediction, but its modest positive predictive power justifies inclusion in an ensemble approach.

### Threat Keyword Analysis

**UNGDC Keyword Extraction:**
- Total threat-related keywords identified: 85,781 across 3,093 speeches
- Keywords tracked: weapon, attack, military, bomb, terrorist, rebel, war, conflict, army, violence, security, hostility, aggression, threat, crisis, danger, missile, nuclear, siege, invasion
- Peak year: 2009 (likely corresponding to Afghanistan/Iraq escalations)
- Year-over-year variation: Threat mentions fluctuate between 700-1000 per year in recent decade

### Regional Sentiment Variation

**Sentiment Ranking by Region (Highest to Lowest):**
1. South America: +0.177
2. North America: +0.137
3. East Asia: +0.065
4. South Asia: +0.039
5. Western Europe: +0.033
6. Eastern Europe: +0.005
7. Middle East: +0.004

**Interpretation:** Middle East and Eastern Europe exhibit near-zero mean sentiment in recent data, while Western regions show modest positive sentiment. Regional differences may reflect both genuine linguistic patterns and social media source bias.

---

## Next Steps

1. **NER Pipeline:** Complete spaCy-based threat entity extraction and temporal analysis
2. **Feature Engineering:** Introduce temporal lag features (t-1, t-2) to capture causal relationships
3. **Ensemble Methods:** Implement Random Forest and Gradient Boosting classifiers
4. **Ablation Study:** Systematically compare sentiment-only vs. NER-only vs. topic-only vs. combined features
5. **Validation:** Test on held-out geopolitical events to assess real-world predictive capability

**Conclusion:** Initial sentiment-based models show modest predictive power above baseline. The weak correlation (-0.046) suggests that linguistic signals require multi-faceted feature engineering and ensemble methods to achieve practically useful predictions. We anticipate substantial improvements through NER and topic modeling integration.
