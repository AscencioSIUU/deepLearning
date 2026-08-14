"""Entrena el modelo final (config ganadora del sweep, Batch 4) y guarda un
ensemble de 5-fold CV como artifacts para predecir sin pasos manuales.

Por qué ensemble en vez de un solo modelo reentrenado sobre todo train.csv:
promediar las predicciones (en log-espacio) de los 5 modelos de CV redujo el RMSE
de validación en pruebas ($39,192 modelo único vs $36,285 ensemble, mismo split) --
cada fold ya vio ~80% de los datos y fue validado contra el 20% restante, así que
el ensemble no cuesta entrenamiento extra respecto a la validación cruzada que de
todas formas se corre para confirmar la config ganadora.

Uso: python train.py
Salida: artifacts/ensemble.pkl (5x pipeline+model), artifacts/meta.json
"""
import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from preprocessing import load_and_clean, build_pipeline
from model import cross_validate_config

SEED = 42
BASE_DIR = os.path.dirname(__file__)
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
K_FOLDS = 5

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

    csv_path = os.path.join(BASE_DIR, "train.csv")
    df = load_and_clean(csv_path)
    X = df.drop(columns=["SalePrice", "SalePriceLog"])
    y = df["SalePriceLog"]

    print(f"Entrenando ensemble de {K_FOLDS}-fold CV con la config ganadora...")
    fold_results = cross_validate_config(
        X, y, build_pipeline, BEST_CONFIG, k=K_FOLDS, seed=SEED,
        epochs=250, patience=25, return_models=True,
    )
    cv_rmse_log = [r["val_rmse_log"] for r in fold_results]
    cv_rmse_usd = [r["val_rmse_usd"] for r in fold_results]
    for r in fold_results:
        print(f"  fold {r['fold']}: val_rmse_log={r['val_rmse_log']:.4f}  val_rmse_usd=${r['val_rmse_usd']:,.0f}")
    cv_mean_log, cv_std_log = float(np.mean(cv_rmse_log)), float(np.std(cv_rmse_log))
    cv_mean_usd, cv_std_usd = float(np.mean(cv_rmse_usd)), float(np.std(cv_rmse_usd))
    print(f"CV (k={K_FOLDS}) val RMSE log:  {cv_mean_log:.4f} ± {cv_std_log:.4f}")
    print(f"CV (k={K_FOLDS}) val RMSE USD:  ${cv_mean_usd:,.0f} ± ${cv_std_usd:,.0f}")

    # Rango de precios observado en train -> bounds de seguridad contra
    # extrapolación catastrófica al predecir sobre el held-out (ver reporte,
    # sección "fragilidad ante extrapolación": una casa con GrLivArea fuera de
    # rango llegó a predecirse en $3.88M). Clip amplio, solo evita explosiones.
    price_min, price_max = float(df["SalePrice"].min()), float(df["SalePrice"].max())
    clip_min, clip_max = 0.5 * price_min, 1.5 * price_max

    ensemble = [{"pipeline": r["pipeline"], "state_dict": r["model"].state_dict()}
                for r in fold_results]
    with open(os.path.join(ARTIFACTS_DIR, "ensemble.pkl"), "wb") as f:
        pickle.dump(ensemble, f)

    meta = {
        "hidden_sizes": list(BEST_CONFIG["hidden_sizes"]),
        "dropout": BEST_CONFIG["dropout"],
        "batchnorm": BEST_CONFIG["batchnorm"],
        "k_folds": K_FOLDS,
        "cv_val_rmse_log_mean": cv_mean_log,
        "cv_val_rmse_log_std": cv_std_log,
        "cv_val_rmse_usd_mean": cv_mean_usd,
        "cv_val_rmse_usd_std": cv_std_usd,
        "clip_min_usd": clip_min,
        "clip_max_usd": clip_max,
        "seed": SEED,
    }
    with open(os.path.join(ARTIFACTS_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nArtifacts guardados en {ARTIFACTS_DIR}/")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
