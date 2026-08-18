import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, n_features, hidden_sizes=(96,), dropout=0.1, batchnorm=False,
                 skip=False):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self.skip = nn.Linear(n_features, 1) if skip else None

    def forward(self, x):
        out = self.net(x).squeeze(-1)
        if self.skip is not None:
            out = out + self.skip(x).squeeze(-1)
        return out


def make_loaders(X_train, y_train, X_val, y_val, batch_size=32):
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(np.asarray(y_train), dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(np.asarray(y_val), dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def rmse_log(model, loader):
    model.eval()
    sq_errs, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb)
            sq_errs += ((pred - yb) ** 2).sum().item()
            n += len(yb)
    return (sq_errs / n) ** 0.5


def train_model(model, train_loader, val_loader, lr=1e-3, weight_decay=0.0,
                epochs=400, patience=30, scheduler=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    sched = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=8)
             if scheduler else None)

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        val_rmse = rmse_log(model, val_loader)
        if sched is not None:
            sched.step(val_rmse)

        if val_rmse < best_val:
            best_val = val_rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            break

    model.load_state_dict(best_state)
    return model, None, best_val
