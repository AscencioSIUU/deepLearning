import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from ensemble import ensemble_predict_log
from preprocessing import engineer_features

BASE_DIR = os.path.dirname(__file__)
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
PREDICTIONS_DIR = os.path.join(BASE_DIR, "predictions")


def load_artifacts():
    with open(os.path.join(ARTIFACTS_DIR, "meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "ensemble.pkl"), "rb") as f:
        ensemble = pickle.load(f)
    return ensemble, meta


def predict(csv_path):
    ensemble, meta = load_artifacts()
    df = pd.read_csv(csv_path)

    has_target = "SalePrice" in df.columns
    y_true = df["SalePrice"].values if has_target else None
    ids = df["Id"].values if "Id" in df.columns else np.arange(len(df))

    X = engineer_features(df.drop(columns=["SalePrice"], errors="ignore"))
    pred_log = ensemble_predict_log(ensemble["folds"], X, dropout=ensemble.get("dropout", 0.0))
    pred_usd = np.expm1(pred_log)
    pred_usd = np.clip(pred_usd, meta["clip_min_usd"], meta["clip_max_usd"])

    out = pd.DataFrame({"Id": ids, "Prediction": np.rint(pred_usd).astype(int)})
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    out_path = os.path.join(
        PREDICTIONS_DIR, os.path.basename(os.path.splitext(csv_path)[0]) + "_predictions.csv")
    out.to_csv(out_path, index=False)
    print(f"Predicciones guardadas en {out_path} (ensemble de {len(ensemble['folds'])} MLPs)")

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
