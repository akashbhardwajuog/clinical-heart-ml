import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report
from .common import PROCESSED, MODELS, load_config, require

def run_sklearn():
    cfg = load_config()
    target = cfg["target_column"]
    
    # Load locked splits
    train = pd.read_csv(require(PROCESSED / "train.csv"))
    test = pd.read_csv(require(PROCESSED / "test.csv"))
    
    X_train, y_train = train.drop(columns=[target]), train[target]
    X_test, y_test = test.drop(columns=[target]), test[target]
    
    # Pipeline ensures scaler is fit ONLY on training data
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(penalty="l1", solver="liblinear", random_state=cfg["seed"], max_iter=1000))
    ])
    
    pipe.fit(X_train, y_train)
    
    # Evaluate
    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)[:, 1]
    
    print("\n--- Logistic Regression Baseline ---")
    print(f"ROC-AUC: {roc_auc_score(y_test, probs):.3f}")
    print(classification_report(y_test, preds))
    
    # Save the trained model
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODELS / "logistic_model.joblib")
    print("Scikit-Learn model saved successfully.")

if __name__ == "__main__":
    run_sklearn()