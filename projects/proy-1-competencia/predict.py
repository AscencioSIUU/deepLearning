"""Carga los artifacts entrenados y genera predicciones sobre un CSV nuevo
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

BASE_DIR = os.path.dirname(__file__)
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")


def load_artifacts():
    with open(os.path.join(ARTIFACTS_DIR, "meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "pipeline.pkl"), "rb") as f:
        pipe = pickle.load(f)
    model = MLP(n_features=meta["n_features_in"], hidden_sizes=tuple(meta["hidden_sizes"]),
                dropout=meta["dropout"], batchnorm=meta["batchnorm"])
    model.load_state_dict(torch.load(os.path.join(ARTIFACTS_DIR, "model.pt")))
    model.eval()
    return model, pipe, meta


def predict(csv_path):
    model, pipe, meta = load_artifacts()
    df = pd.read_csv(csv_path)

    has_target = "SalePrice" in df.columns
    y_true = df["SalePrice"].values if has_target else None

    X = df.drop(columns=[c for c in ["Id", "SalePrice"] if c in df.columns], errors="ignore")
    # el pipeline espera las mismas columnas que en fit (incluye Id como no-feature
    # ya excluida por preprocessing.get_feature_lists); reincorporamos Id solo para
    # el output final, no para el transform.
    if "Id" in df.columns:
        ids = df["Id"].values
    else:
        ids = np.arange(len(df))

    X_t = pipe.transform(df.drop(columns=["SalePrice"], errors="ignore"))
    with torch.no_grad():
        pred_log = model(torch.tensor(X_t, dtype=torch.float32)).numpy()
    pred_usd = np.expm1(pred_log)

    out = pd.DataFrame({"Id": ids, "SalePrice_pred": pred_usd})
    out_path = os.path.splitext(csv_path)[0] + "_predictions.csv"
    out.to_csv(out_path, index=False)
    print(f"Predicciones guardadas en {out_path}")

    if has_target:
        rmse = float(np.sqrt(np.mean((pred_usd - y_true) ** 2)))
        rmse_log_val = float(np.sqrt(np.mean((pred_log - np.log1p(y_true)) ** 2)))
        print(f"RMSE (USD):  ${rmse:,.2f}")
        print(f"RMSE (log):  {rmse_log_val:.4f}")
        return rmse, rmse_log_val
    return None, None


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python predict.py <ruta_al_csv>")
        sys.exit(1)
    predict(sys.argv[1])
