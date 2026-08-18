import numpy as np
import torch
from sklearn.model_selection import KFold

from model import MLP, make_loaders, train_model


def rmse_usd(pred_log, y_log):
    return float(np.sqrt(np.mean((np.expm1(pred_log) - np.expm1(y_log)) ** 2)))


def train_ensemble(X, y, build_pipeline_fn, mlp_config, seeds=(42, 1, 7, 11, 23),
                   k=5, epochs=400, patience=30):
    X = X.reset_index(drop=True)
    target = np.asarray(y).astype(np.float32)

    oof_sum = np.zeros(len(X))
    folds = []
    for s in seeds:
        kf = KFold(n_splits=k, shuffle=True, random_state=s)
        for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]

            mu = float(target[tr_idx].mean())
            sigma = float(target[tr_idx].std()) or 1.0
            t_tr = (target[tr_idx] - mu) / sigma
            t_va = (target[va_idx] - mu) / sigma

            pipe = build_pipeline_fn(X_tr)
            X_tr_t = pipe.fit_transform(X_tr, target[tr_idx])
            X_va_t = pipe.transform(X_va)

            train_loader, val_loader = make_loaders(X_tr_t, t_tr, X_va_t, t_va)
            torch.manual_seed(s + fold)
            mlp = MLP(n_features=X_tr_t.shape[1], hidden_sizes=mlp_config["hidden_sizes"],
                      dropout=mlp_config["dropout"], batchnorm=mlp_config["batchnorm"],
                      skip=mlp_config.get("skip", False))
            mlp, _, _ = train_model(mlp, train_loader, val_loader, lr=mlp_config["lr"],
                                    weight_decay=mlp_config["weight_decay"],
                                    epochs=epochs, patience=patience)
            mlp.eval()
            with torch.no_grad():
                pred_std = mlp(torch.tensor(X_va_t, dtype=torch.float32)).numpy()
            oof_sum[va_idx] += pred_std * sigma + mu
            folds.append({"pipeline": pipe, "mlp_state": mlp.state_dict(),
                          "mu": mu, "sigma": sigma})

    return folds, oof_sum / len(seeds)


def _mlp_from_state(state, dropout=0.0):
    n_features = state["net.0.weight"].shape[1]
    hidden = [state[key].shape[0] for key in state
              if key.endswith(".weight") and key.startswith("net.")][:-1]
    skip = "skip.weight" in state
    model = MLP(n_features=n_features, hidden_sizes=tuple(hidden), dropout=dropout,
                batchnorm=False, skip=skip)
    model.load_state_dict(state)
    model.eval()
    return model


def ensemble_predict_log(folds, X_new, dropout=0.0):
    acc = np.zeros(len(X_new))
    for fold in folds:
        X_t = fold["pipeline"].transform(X_new)
        mlp = _mlp_from_state(fold["mlp_state"], dropout=dropout)
        with torch.no_grad():
            pred_std = mlp(torch.tensor(X_t, dtype=torch.float32)).numpy()
        acc += pred_std * fold.get("sigma", 1.0) + fold.get("mu", 0.0)
    return acc / len(folds)
