from anomaly_detection.config import PipelineConfig
from anomaly_detection.pipeline import run_pipeline
from anomaly_detection.visualize import generate_visualizations


def main() -> None:
    config = PipelineConfig()
    outputs = run_pipeline(config)
    plots = generate_visualizations(
        output_dir=config.output_dir,
        nlp_weight=config.nlp_weight,
        zscore_weight=config.zscore_weight,
        iso_weight=config.iso_weight,
        residual_weight=config.residual_weight,
    )

    print("Anomaly detection pipeline complete.")
    print(f"Scores:  {outputs['scores']}")
    print(f"Alerts:  {outputs['alerts']}")
    print(f"Summary: {outputs['summary']}")
    print(f"Top countries plot:    {plots['top_countries']}")
    print(f"Alerts-by-month plot:  {plots['alerts_per_month']}")
    print(f"Reason-features plot:  {plots['reason_features']}")
    print(f"Alert-breakdown plot:  {plots['alert_decomposition']}")
    print(f"Heatmap plot:          {plots['heatmap']}")


if __name__ == "__main__":
    main()

