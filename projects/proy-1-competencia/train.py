"""Entrena el modelo final (config ganadora del sweep, Batch 4) sobre train+val
completo y guarda los artifacts necesarios para predecir sin pasos manuales.

Uso: python train.py
Salida: artifacts/model.pt, artifacts/pipeline.pkl, artifacts/meta.json
"""
import json
import os
import pickle
import sys

import torch

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from preprocessing import load_and_clean, build_pipeline, get_feature_lists
from model import MLP, make_loaders, train_model, rmse_log, rmse_dollars, cross_validate_config

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

    # 5-fold CV de la config ganadora: confirma que el resultado del split 80/20 no
    # es ruido de un único split (limitación señalada en el reporte, Batch 4/6).
    print("\nCorriendo 5-fold cross-validation de la config ganadora...")
    cv_results = cross_validate_config(X, y, build_pipeline, BEST_CONFIG, k=5, seed=SEED)
    cv_rmse_log = [r["val_rmse_log"] for r in cv_results]
    cv_rmse_usd = [r["val_rmse_usd"] for r in cv_results]
    cv_mean_log, cv_std_log = float(np.mean(cv_rmse_log)), float(np.std(cv_rmse_log))
    cv_mean_usd, cv_std_usd = float(np.mean(cv_rmse_usd)), float(np.std(cv_rmse_usd))
    print(f"CV (k=5) val RMSE log:  {cv_mean_log:.4f} ± {cv_std_log:.4f}  (por fold: {[round(v,4) for v in cv_rmse_log]})")
    print(f"CV (k=5) val RMSE USD:  ${cv_mean_usd:,.0f} ± ${cv_std_usd:,.0f}")

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
    with open(os.path.join(ARTIFACTS_DIR, "pipeline.pkl"), "wb") as f:
        pickle.dump(pipe_full, f)

    numeric_cols, ordinal_cols, nominal_cols = get_feature_lists(df)
    meta = {
        "hidden_sizes": list(BEST_CONFIG["hidden_sizes"]),
        "dropout": BEST_CONFIG["dropout"],
        "batchnorm": BEST_CONFIG["batchnorm"],
        "n_features_in": X_full_t.shape[1],
        "n_epochs_final": n_epochs,
        "holdout_val_rmse_log": best_val_log,
        "holdout_val_rmse_usd": val_rmse_usd,
        "cv_k": 5,
        "cv_val_rmse_log_mean": cv_mean_log,
        "cv_val_rmse_log_std": cv_std_log,
        "cv_val_rmse_usd_mean": cv_mean_usd,
        "cv_val_rmse_usd_std": cv_std_usd,
        "seed": SEED,
    }
    with open(os.path.join(ARTIFACTS_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nArtifacts guardados en {ARTIFACTS_DIR}/")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
