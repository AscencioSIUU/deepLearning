import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from preprocessing import load_and_clean, build_pipeline
from ensemble import train_ensemble, rmse_usd

BASE_DIR = os.path.dirname(__file__)
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
SWEEP_SEEDS = [42, 1, 7]
FINAL_SEEDS = [42, 1, 7, 11, 23]
K_FOLDS = 5


def _cfg(hidden, dropout=0.0, weight_decay=0.0, skip=False):
    return {"hidden_sizes": hidden, "dropout": dropout, "weight_decay": weight_decay,
            "batchnorm": False, "lr": 1e-3, "skip": skip}


CANDIDATES = [
    _cfg((64,), skip=True),
    _cfg((64,), dropout=0.1, skip=True),
    _cfg((48,), skip=True),
    _cfg((80,), dropout=0.1, skip=True),
    _cfg((96,), dropout=0.1, skip=True),
]


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    df = load_and_clean(os.path.join(BASE_DIR, "train.csv"))
    X = df.drop(columns=["SalePrice", "SalePriceLog"])
    y = df["SalePriceLog"]
    y_arr = np.asarray(y)

    print("Sweep de arquitecturas MLP (CV multi-seed USD):")
    best = None
    for cfg in CANDIDATES:
        _, oof = train_ensemble(X, y, build_pipeline, cfg, seeds=SWEEP_SEEDS, k=K_FOLDS)
        rmse = rmse_usd(oof, y_arr)
        arch = "->".join(map(str, cfg["hidden_sizes"]))
        print(f"  {arch:<6} dropout={cfg['dropout']} skip={cfg['skip']}  CV USD=${rmse:,.0f}")
        if best is None or rmse < best["rmse"]:
            best = {"rmse": rmse, "cfg": cfg}

    arch = "->".join(map(str, best["cfg"]["hidden_sizes"]))
    print(f"\nMejor: {arch} dropout={best['cfg']['dropout']} skip={best['cfg']['skip']}")

    print(f"Entrenando ensemble final ({len(FINAL_SEEDS)} seeds x {K_FOLDS} folds)...")
    folds, oof = train_ensemble(X, y, build_pipeline, best["cfg"],
                                seeds=FINAL_SEEDS, k=K_FOLDS)
    final_rmse = rmse_usd(oof, y_arr)
    print(f"  ensemble final ({len(folds)} MLPs) CV USD=${final_rmse:,.0f}")

    price_min, price_max = float(df["SalePrice"].min()), float(df["SalePrice"].max())
    ensemble = {"folds": folds, "hidden_sizes": list(best["cfg"]["hidden_sizes"]),
                "dropout": best["cfg"]["dropout"], "skip": best["cfg"]["skip"]}
    with open(os.path.join(ARTIFACTS_DIR, "ensemble.pkl"), "wb") as f:
        pickle.dump(ensemble, f)

    meta = {
        "model": "mlp-ensemble",
        "hidden_sizes": list(best["cfg"]["hidden_sizes"]),
        "skip": best["cfg"]["skip"],
        "dropout": best["cfg"]["dropout"],
        "target_standardized": True,
        "lr_scheduler": "ReduceLROnPlateau",
        "k_folds": K_FOLDS,
        "seeds": FINAL_SEEDS,
        "n_mlps": len(folds),
        "cv_rmse_usd": final_rmse,
        "clip_min_usd": 0.5 * price_min,
        "clip_max_usd": 1.5 * price_max,
    }
    with open(os.path.join(ARTIFACTS_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nArtifacts en {ARTIFACTS_DIR}/")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
