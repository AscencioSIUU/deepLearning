"""MLP y training loop para la competencia de regresión (SalePrice en log-espacio).

Dataset es pequeño (~1000 filas, 223 features) -> CPU es suficiente y más simple que
gestionar dispositivo MPS/CUDA para un modelo de este tamaño.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, n_features, hidden_sizes=(128, 64), dropout=0.2, batchnorm=False):
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

    def forward(self, x):
        return self.net(x).squeeze(-1)


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
    """RMSE en el espacio en que se entrena (log1p(SalePrice))."""
    model.eval()
    sq_errs, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb)
            sq_errs += ((pred - yb) ** 2).sum().item()
            n += len(yb)
    return (sq_errs / n) ** 0.5


def rmse_dollars(model, loader):
    """RMSE en USD: invierte log1p con expm1 antes de comparar."""
    model.eval()
    sq_errs, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            pred = torch.expm1(model(xb))
            true = torch.expm1(yb)
            sq_errs += ((pred - true) ** 2).sum().item()
            n += len(yb)
    return (sq_errs / n) ** 0.5


def train_model(model, train_loader, val_loader, lr=1e-3, weight_decay=0.0,
                 epochs=200, patience=20, verbose=False):
    """Entrena con Adam + MSELoss (en log-espacio) y early stopping sobre val RMSE."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    history = {"train_rmse": [], "val_rmse": []}
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

        train_rmse = rmse_log(model, train_loader)
        val_rmse = rmse_log(model, val_loader)
        history["train_rmse"].append(train_rmse)
        history["val_rmse"].append(val_rmse)

        if val_rmse < best_val:
            best_val = val_rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if verbose and epoch % 10 == 0:
            print(f"epoch {epoch:3d}  train_rmse={train_rmse:.4f}  val_rmse={val_rmse:.4f}")

        if bad_epochs >= patience:
            break

    model.load_state_dict(best_state)
    return model, history, best_val


if __name__ == "__main__":
    # ponytail: smoke test de una config chica, no framework -- `python src/model.py`
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from preprocessing import load_and_clean, build_pipeline
    from sklearn.model_selection import train_test_split

    csv = os.path.join(os.path.dirname(__file__), "..", "train.csv")
    df = load_and_clean(csv)
    X = df.drop(columns=["SalePrice", "SalePriceLog"])
    y = df["SalePriceLog"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = build_pipeline(X_train)
    X_train_t = pipe.fit_transform(X_train)
    X_val_t = pipe.transform(X_val)

    train_loader, val_loader = make_loaders(X_train_t, y_train, X_val_t, y_val)
    model = MLP(n_features=X_train_t.shape[1], hidden_sizes=(64, 32), dropout=0.1)
    model, history, best_val = train_model(model, train_loader, val_loader, epochs=50, patience=10)

    assert best_val > 0 and best_val < 1.0, f"val RMSE (log) fuera de rango esperado: {best_val}"
    assert len(history["train_rmse"]) > 0
    print(f"OK: smoke test, {len(history['train_rmse'])} épocas, val_rmse(log)={best_val:.4f}, "
          f"val_rmse($)={rmse_dollars(model, val_loader):,.0f}")
