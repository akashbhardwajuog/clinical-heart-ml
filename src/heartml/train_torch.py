import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from .common import PROCESSED, MODELS, load_config, require

class HeartNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)

def run_torch():
    cfg = load_config()
    target = cfg["target_column"]
    
    train = pd.read_csv(require(PROCESSED / "train.csv"))
    test = pd.read_csv(require(PROCESSED / "test.csv"))
    
    X_train, y_train = train.drop(columns=[target]).values, train[target].values
    X_test, y_test = test.drop(columns=[target]).values, test[target].values
    
    # Scale strictly on training distribution
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to PyTorch Tensors
    X_tr_t = torch.FloatTensor(X_train)
    y_tr_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_te_t = torch.FloatTensor(X_test)
    
    # Initialize Model
    model = HeartNet(input_dim=X_train.shape[1], hidden_dim=cfg["torch_hidden_units"])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["torch_learning_rate"])
    
    # Train Loop
    print("\n--- Training PyTorch DNN ---")
    model.train()
    for epoch in range(cfg["torch_epochs"]):
        optimizer.zero_grad()
        out = model(X_tr_t)
        loss = criterion(out, y_tr_t)
        loss.backward()
        optimizer.step()
        
    # Evaluate
    model.eval()
    with torch.no_grad():
        preds = model(X_te_t)
        auc = roc_auc_score(y_test, preds.numpy())
        print(f"Final PyTorch ROC-AUC: {auc:.3f}")
        
    # Save the PyTorch weights
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS / "heartnet.pt")
    print("PyTorch weights saved successfully.")

if __name__ == "__main__":
    run_torch()