import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
from sklearn.metrics import roc_curve, auc
from .common import PROCESSED, MODELS, FIGURES, TABLES, require, load_config

def run_evaluation():
    cfg = load_config()
    target = cfg["target_column"]
    
    # Ensure output directories exist
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    
    # Load data and the trained logistic model
    test = pd.read_csv(require(PROCESSED / "test.csv"))
    X_test, y_test = test.drop(columns=[target]), test[target]
    pipe = joblib.load(require(MODELS / "logistic_model.joblib"))
    
    # 1. Feature Importance Extraction
    model = pipe.named_steps['clf']
    features = X_test.columns
    coefs = model.coef_[0]
    
    importance_df = pd.DataFrame({'Biomarker': features, 'Coefficient': coefs})
    importance_df['Absolute_Weight'] = np.abs(importance_df['Coefficient'])
    importance_df = importance_df.sort_values(by='Absolute_Weight', ascending=False)
    importance_df.to_csv(TABLES / "feature_importance.csv", index=False)
    
    # Plot Feature Importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Coefficient', y='Biomarker', data=importance_df, palette='coolwarm')
    plt.title('Clinical Biomarker Importance (Predicting Heart Failure Mortality)')
    plt.tight_layout()
    plt.savefig(FIGURES / "feature_importance.png", dpi=300)
    plt.close()
    
    # 2. ROC Curve
    probs = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(FIGURES / "roc_curve.png", dpi=300)
    plt.close()
    
    print("Visualizations successfully saved to reports/figures/.")

if __name__ == "__main__":
    run_evaluation()