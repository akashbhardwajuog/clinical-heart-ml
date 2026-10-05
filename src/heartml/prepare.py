import pandas as pd
import numpy as np
from .common import RAW, INTERIM, load_config, require

def run_prepare():
    cfg = load_config()
    df = pd.read_csv(require(RAW / "heart_failure.csv"))
    
    print(f"Loaded {df.shape[0]} patients, {df.shape[1]} features.")
    
    # 1. QC Checks
    if df.isnull().sum().sum() > 0:
        raise ValueError("Missing values detected. Imputation required.")
    
    # 2. Biological Transformation
    # Highly skewed clinical features require logarithmic scaling
    for col in cfg["features_to_log"]:
        df[f"log_{col}"] = np.log1p(df[col])
        df = df.drop(columns=[col])
        
    df.to_csv(INTERIM / "heart_cleaned.csv", index=False)
    print("Cleaned and log-transformed data saved to interim.")

if __name__ == "__main__":
    run_prepare()