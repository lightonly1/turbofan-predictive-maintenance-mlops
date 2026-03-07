import pandas as pd
import os
from loguru import logger

def generate_drift_report():
    logger.info("Loading data for drift detection...")
    
    data_path = "data/processed/train_processed/train_processed.csv"
    
    if not os.path.exists(data_path):
        logger.error(f"Data not found: {data_path}")
        return
    
    df = pd.read_csv(data_path)
    
    split = int(len(df) * 0.7)
    reference_data = df[:split]
    current_data = df[split:]
    
    sensor_cols = [c for c in df.columns if 'sensor' in c or 'rolling' in c]
    
    logger.info(f"Reference: {len(reference_data)} rows")
    logger.info(f"Current: {len(current_data)} rows")
    logger.info(f"Sensors tracked: {len(sensor_cols)}")
    
    # Drift detection — mean shift check
    os.makedirs("logs/monitoring", exist_ok=True)
    
    results = []
    for col in sensor_cols:
        ref_mean = reference_data[col].mean()
        cur_mean = current_data[col].mean()
        drift = abs(ref_mean - cur_mean) / (ref_mean + 1e-9)
        results.append({
            "feature": col,
            "reference_mean": round(ref_mean, 4),
            "current_mean": round(cur_mean, 4),
            "drift_pct": round(drift * 100, 2),
            "drifted": drift > 0.05
        })
    
    report_df = pd.DataFrame(results)
    report_df.to_csv("logs/monitoring/drift_report.csv", index=False)
    
    drifted = report_df[report_df["drifted"] == True]
    logger.success(f"Report saved! {len(drifted)}/{len(sensor_cols)} features drifted")
    
    print(report_df.to_string(index=False))
    return "logs/monitoring/drift_report.csv"

if __name__ == "__main__":
    generate_drift_report()