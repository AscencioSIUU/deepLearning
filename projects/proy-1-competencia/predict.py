"""Carga el ensemble de CV entrenado y genera predicciones sobre un CSV nuevo
(el held-out del día de competencia). Si el CSV trae SalePrice, también calcula RMSE.

Uso: python predict.py <ruta_al_csv>
Salida: <ruta_al_csv sin extensión>_predictions.csv con columnas Id, SalePrice_pred
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from model import MLP
from preprocessing import engineer_features

BASE_DIR = os.path.dirname(__file__)
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")


def load_artifacts():
    with open(os.path.join(ARTIFACTS_DIR, "meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "ensemble.pkl"), "rb") as f:
        ensemble_raw = pickle.load(f)

    ensemble = []
    for member in ensemble_raw:
        n_features = member["state_dict"]["net.0.weight"].shape[1]  # cada fold puede
        # tener un ancho de one-hot distinto si a su 80% de train le faltó alguna
        # categoría rara -- se infiere del propio checkpoint, no de meta.json.
        model = MLP(n_features=n_features, hidden_sizes=tuple(meta["hidden_sizes"]),
                    dropout=meta["dropout"], batchnorm=meta["batchnorm"])
        model.load_state_dict(member["state_dict"])
        model.eval()
        ensemble.append({"pipeline": member["pipeline"], "model": model})
    return ensemble, meta


def predict(csv_path):
    ensemble, meta = load_artifacts()
    df = pd.read_csv(csv_path)

    has_target = "SalePrice" in df.columns
    y_true = df["SalePrice"].values if has_target else None

    ids = df["Id"].values if "Id" in df.columns else np.arange(len(df))

    X = df.drop(columns=["SalePrice"], errors="ignore")
    X = engineer_features(X)  # mismas features derivadas usadas en entrenamiento

    fold_preds = []
    for member in ensemble:
        X_t = member["pipeline"].transform(X)
        with torch.no_grad():
            fold_preds.append(member["model"](torch.tensor(X_t, dtype=torch.float32)).numpy())
    pred_log = np.mean(fold_preds, axis=0)
    pred_usd = np.expm1(pred_log)

    # Clip de seguridad: evita que una fila fuera del rango de entrenamiento
    # (extrapolación) dispare una predicción absurda (ver reporte, sección de
    # discusión) -- ponytail: clip fijo por percentil de train, no una
    # calibración aprendida; subir a un modelo de incertidumbre si el held-out
    # sigue mostrando outliers extremos tras esto.
    pred_usd = np.clip(pred_usd, meta["clip_min_usd"], meta["clip_max_usd"])

    out = pd.DataFrame({"Id": ids, "SalePrice_pred": pred_usd})
    out_path = os.path.splitext(csv_path)[0] + "_predictions.csv"
    out.to_csv(out_path, index=False)
    print(f"Predicciones guardadas en {out_path} (ensemble de {len(ensemble)} modelos)")

    if has_target:
        rmse = float(np.sqrt(np.mean((pred_usd - y_true) ** 2)))
        rmse_log_val = float(np.sqrt(np.mean((np.log1p(pred_usd) - np.log1p(y_true)) ** 2)))
        print(f"RMSE (USD):  ${rmse:,.2f}")
        print(f"RMSE (log):  {rmse_log_val:.4f}")
        return rmse, rmse_log_val
    return None, None


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python predict.py <ruta_al_csv>")
        sys.exit(1)
    predict(sys.argv[1])
