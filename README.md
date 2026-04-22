# From-Words-to-War

This repository contains notebook-based analysis and Python pipelines for modeling geopolitical conflict risk from UN speech text and risk indicators.

## Requirements

- Python 3.10+ (3.11 recommended)
- pip

Install dependencies:

```bash
pip install -r requirements.txt
```

Install the spaCy English model used by `GDR_risk_NER_model.py`:

```bash
python -m spacy download en_core_web_lg
```

Download NLTK resources used by the notebooks:

```bash
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')"
```

## Data Files

Keep required CSV files in the project root.

- Already present in this repo:
	- `UNGDC_1946-2023.csv`
	- `geopolitical_conflict_risk_dataset.csv`
- Needed for GDR model script:
	- `data_gpr_export.csv`

## Running the Notebooks

For the two Python notebooks, open them in Jupyter (or the VS Code notebook editor) and run cells from top to bottom.

- `sentiment-analysis.ipynb`
- `nlp_conflict_prediction_prem_results.ipynb`

## Running the GDR Risk Script

Run from the repository root:

```bash
python GDR_risk_NER_model.py
```

Notes:

- This script expects `data_gpr_export.csv` and `UNGDC_1946-2023.csv` in the root directory.
- It loads `en_core_web_lg` and will fail if that model is not installed.

## Running the Anomaly Detection Pipeline

Run from the repository root:

```bash
python run_anomaly_detection.py
```

Default behavior:

- Reads `UNGDC_1946-2023.csv` and `geopolitical_conflict_risk_dataset.csv`
- Writes outputs to `anomaly_outputs/`
- Prints paths for generated score tables, alerts, summary outputs, and plots