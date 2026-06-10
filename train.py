"""
ML2 Regression — 1D CNN with MAPE-Aligned Loss + Multi-seed Ensemble
Leaderboard result: MAPE 27.57 (1st place)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from copy import deepcopy

# ── Config ────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

SEEDS = [10, 11, 12]


# ── Data ──────────────────────────────────────────────────────────────────────

def load_data(train_path="train.npy", test_path="test_x.npy"):
    train = np.load(train_path)
    test_x = np.load(test_path)

    print("train shape:", train.shape)
    print("test_x shape:", test_x.shape)

    X = train[:, :-1].astype(np.float32)              # (4310, 12288)
    y = train[:, -1].astype(np.float32).reshape(-1, 1) # (4310, 1)
    X_test = test_x.astype(np.float32)                # (4311, 12288)

    print(f"y  min={y.min():.4f}  max={y.max():.4f}  mean={y.mean():.4f}")
    return X, y, X_test


# ── Model ─────────────────────────────────────────────────────────────────────

class CNNRegressorV1Drop(nn.Module):
    """
    1D CNN regressor.
    Treats the 12288-dim feature vector as a 1D signal and extracts
    local patterns via three Conv1d layers with stride=2 downsampling.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1,  16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        # 12288 / 2^3 = 1536  →  64 * 1536 flattened
        self.fc1     = nn.Linear(64 * 1536, 256)
        self.dropout = nn.Dropout(0.2)
        self.fc2     = nn.Linear(256, 1)

    def forward(self, x):
        x = x.unsqueeze(1)          # (B, 12288) → (B, 1, 12288)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# ── Loss & Metric ─────────────────────────────────────────────────────────────

def mape_loss_for_training(y_true, y_pred, eps=0.01, lambda_mse=0.1):
    """
    Custom loss that directly optimizes the leaderboard metric (MAPE).
    MAPE term: aligns training objective with evaluation metric.
    MSE term (×0.1): stabilizes gradients when y is near zero.
    """
    mape_term = torch.mean(torch.abs((y_true - y_pred) / (y_true + eps)))
    mse_term  = torch.mean((y_true - y_pred) ** 2)
    return mape_term + lambda_mse * mse_term


def mape_metric(y_true, y_pred, eps=0.01):
    """Evaluation MAPE in percent (matches leaderboard definition)."""
    return torch.mean(torch.abs((y_true - y_pred) / (y_true + eps))) * 100


# ── Training ──────────────────────────────────────────────────────────────────

def train_one_model(X, y, random_state=42, batch_size=16, lr=3e-4,
                    weight_decay=1e-6, num_epochs=80, patience=10):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=batch_size, shuffle=False
    )

    model     = CNNRegressorV1Drop().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_mape  = float("inf")
    best_epoch     = -1
    patience_count = 0
    best_state     = None

    for epoch in range(num_epochs):
        # train
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = mape_loss_for_training(yb, model(xb))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # validate
        model.eval()
        preds_list, trues_list = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds_list.append(model(xb.to(device)).cpu())
                trues_list.append(yb)

        val_mape = mape_metric(
            torch.cat(trues_list), torch.cat(preds_list)
        ).item()

        print(f"[seed={random_state}] Epoch {epoch+1:02d}/{num_epochs} | "
              f"TrainLoss: {total_loss/len(train_loader):.5f} | ValMAPE: {val_mape:.3f}")

        if val_mape < best_val_mape:
            best_val_mape  = val_mape
            best_epoch     = epoch + 1
            patience_count = 0
            best_state     = deepcopy(model.state_dict())
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"[seed={random_state}] Early stopping at epoch {epoch+1}")
                break

    print(f"[seed={random_state}] Best Val MAPE: {best_val_mape:.3f} at epoch {best_epoch}")
    return best_state, best_val_mape


# ── Ensemble & Submission ─────────────────────────────────────────────────────

def main():
    X, y, X_test = load_data("train.npy", "test_x.npy")
    X_test_t = torch.from_numpy(X_test).float().to(device)

    state_dicts, val_mapes = [], []
    for seed in SEEDS:
        print(f"\n{'='*50}")
        print(f"Training with seed={seed}")
        print('='*50)
        state, mape = train_one_model(X, y, random_state=seed)
        state_dicts.append(state)
        val_mapes.append(mape)

    print("\n=== Ensemble Summary ===")
    for seed, mape in zip(SEEDS, val_mapes):
        print(f"  seed={seed} → Val MAPE: {mape:.3f}")

    # collect test predictions from each model
    all_preds = []
    for seed, state in zip(SEEDS, state_dicts):
        model = CNNRegressorV1Drop().to(device)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            preds = model(X_test_t).cpu().squeeze(1).numpy()
        all_preds.append(preds)
        print(f"  seed={seed} test prediction done. shape={preds.shape}")

    ensemble_preds = np.stack(all_preds, axis=0).mean(axis=0)  # (4311,)
    print(f"\nEnsemble prediction shape: {ensemble_preds.shape}")

    np.savetxt("submission.csv", ensemble_preds, delimiter=",")
    print("Saved: submission.csv")


if __name__ == "__main__":
    main()
