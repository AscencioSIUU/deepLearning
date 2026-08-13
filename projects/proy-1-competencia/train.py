"""Entrena el modelo final (config ganadora del sweep, Batch 4) sobre train+val
completo y guarda los artifacts necesarios para predecir sin pasos manuales.

Uso: python train.py
Salida: artifacts/model.pt, artifacts/pipeline.joblib, artifacts/meta.json
"""
import json
import os
import sys

import joblib
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from preprocessing import load_and_clean, build_pipeline, get_feature_lists
from model import MLP, make_loaders, train_model, rmse_log, rmse_dollars

SEED = 42
BASE_DIR = os.path.dirname(__file__)
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

# Config ganadora del sweep (Batch 4): C1_baseline
BEST_CONFIG = {
    "hidden_sizes": (64, 32),
    "dropout": 0.0,
    "weight_decay": 0.0,
    "batchnorm": False,
    "lr": 1e-3,
}


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    torch.manual_seed(SEED)

    csv_path = os.path.join(BASE_DIR, "train.csv")
    df = load_and_clean(csv_path)
    X = df.drop(columns=["SalePrice", "SalePriceLog"])
    y = df["SalePriceLog"]

    # Split held-out interno solo para reportar una métrica final honesta;
    # el pipeline y el modelo final se re-ajustan sobre TODO train.csv (más datos
    # disponibles para el held-out real del día de competencia).
    from sklearn.model_selection import train_test_split
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=SEED)

    pipe = build_pipeline(X_tr)
    X_tr_t = pipe.fit_transform(X_tr)
    X_va_t = pipe.transform(X_va)
    train_loader, val_loader = make_loaders(X_tr_t, y_tr, X_va_t, y_va, batch_size=32)

    model = MLP(n_features=X_tr_t.shape[1], hidden_sizes=BEST_CONFIG["hidden_sizes"],
                dropout=BEST_CONFIG["dropout"], batchnorm=BEST_CONFIG["batchnorm"])
    model, history, best_val_log = train_model(
        model, train_loader, val_loader, lr=BEST_CONFIG["lr"],
        weight_decay=BEST_CONFIG["weight_decay"], epochs=300, patience=25, verbose=True,
    )
    val_rmse_usd = rmse_dollars(model, val_loader)
    print(f"\nHeld-out interno (80/20) -> val RMSE log={best_val_log:.4f}  USD=${val_rmse_usd:,.0f}")

    # Reajustar pipeline y reentrenar sobre TODO train.csv para el modelo de producción
    pipe_full = build_pipeline(X)
    X_full_t = pipe_full.fit_transform(X)
    full_loader, _ = make_loaders(X_full_t, y, X_va_t, y_va, batch_size=32)  # val_loader solo para early stop signal
    final_model = MLP(n_features=X_full_t.shape[1], hidden_sizes=BEST_CONFIG["hidden_sizes"],
                       dropout=BEST_CONFIG["dropout"], batchnorm=BEST_CONFIG["batchnorm"])
    torch.manual_seed(SEED)
    # Entrena con el mismo número de épocas que el mejor punto encontrado arriba
    # (no hay val set independiente al usar todos los datos, así que se fija el
    # número de épocas al punto óptimo ya observado en el split 80/20).
    n_epochs = len(history["train_rmse"]) - 25  # época del mejor punto (antes de la paciencia)
    n_epochs = max(n_epochs, 10)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=BEST_CONFIG["lr"],
                                  weight_decay=BEST_CONFIG["weight_decay"])
    criterion = torch.nn.MSELoss()
    for epoch in range(n_epochs):
        final_model.train()
        for xb, yb in full_loader:
            optimizer.zero_grad()
            loss = criterion(final_model(xb), yb)
            loss.backward()
            optimizer.step()

    torch.save(final_model.state_dict(), os.path.join(ARTIFACTS_DIR, "model.pt"))
    joblib.dump(pipe_full, os.path.join(ARTIFACTS_DIR, "pipeline.joblib"))

    numeric_cols, ordinal_cols, nominal_cols = get_feature_lists(df)
    meta = {
        "hidden_sizes": list(BEST_CONFIG["hidden_sizes"]),
        "dropout": BEST_CONFIG["dropout"],
        "batchnorm": BEST_CONFIG["batchnorm"],
        "n_features_in": X_full_t.shape[1],
        "n_epochs_final": n_epochs,
        "holdout_val_rmse_log": best_val_log,
        "holdout_val_rmse_usd": val_rmse_usd,
        "seed": SEED,
    }
    with open(os.path.join(ARTIFACTS_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nArtifacts guardados en {ARTIFACTS_DIR}/")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
