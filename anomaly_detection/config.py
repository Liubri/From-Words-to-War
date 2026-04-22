from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    ungdc_path: Path = Path("UNGDC_1946-2023.csv")
    risk_path: Path = Path("geopolitical_conflict_risk_dataset.csv")
    output_dir: Path = Path("anomaly_outputs")
    min_history_months: int = 9
    rolling_window: int = 6
    nlp_weight: float = 0.40
    zscore_weight: float = 0.25
    iso_weight: float = 0.20
    residual_weight: float = 0.15
    isolation_forest_contamination: float = 0.08
    random_state: int = 42
    alert_percentile: float = 0.95
