# Project Plan
- **Dataset:** UCI Clinical Heart Failure Records (299 patients).
- **Question:** Predict patient mortality (`DEATH_EVENT`) during the follow-up period based on clinical biomarkers.
- **Encoding:** 0 = Survived, 1 = Deceased.
- **Primary Metric:** ROC-AUC (due to class imbalance).
- **Secondary:** F1-Score, Precision, Recall.
- **Models:** L1-Regularized Logistic Regression, PyTorch Deep Neural Network.
- **Preprocessing:** Log1p transformation of skewed biomarkers (CPK, Serum Creatinine). Standard scaling applied strictly post-split to prevent data leakage.