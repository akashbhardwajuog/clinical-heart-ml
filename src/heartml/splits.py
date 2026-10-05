import pandas as pd
from sklearn.model_selection import train_test_split
from .common import INTERIM, PROCESSED, load_config, require

def make_split():
    cfg = load_config()
    df = pd.read_csv(require(INTERIM / "heart_cleaned.csv"))
    
    # Stratified split ensures the ratio of deaths is identical in train and test
    train, test = train_test_split(
        df, test_size=cfg["test_fraction"], 
        stratify=df[cfg["target_column"]], 
        random_state=cfg["seed"]
    )
    
    train.to_csv(PROCESSED / "train.csv", index=False)
    test.to_csv(PROCESSED / "test.csv", index=False)
    print(f"Data split locked: {len(train)} train, {len(test)} test.")

if __name__ == "__main__":
    make_split()