"""Preprocesamiento del dataset Ames Housing para el MLP de la competencia.

Uso:
    from preprocessing import load_and_clean, build_pipeline

    df = load_and_clean("train.csv")
    X, y = df.drop(columns=["SalePrice", "SalePriceLog"]), df["SalePriceLog"]
    pipe = build_pipeline(X)
    X_t = pipe.fit_transform(X)

El mismo `pipe` (serializado con pickle) se reutiliza en train.py/predict.py para
garantizar que el held-out reciba exactamente las mismas transformaciones.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

# ---------------------------------------------------------------- columnas

# Categóricas donde NA significa "no tiene esa característica" -> imputar "None"
NONE_FILL_COLS = [
    "Alley", "MasVnrType", "BsmtQual", "BsmtCond", "BsmtExposure",
    "BsmtFinType1", "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish",
    "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature",
]

# Ordinales de calidad, mismo orden Po<Fa<TA<Gd<Ex (+ None cuando no aplica)
QUALITY_ORDER = ["None", "Po", "Fa", "TA", "Gd", "Ex"]
QUALITY_COLS = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
    "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC",
]

# Otras ordinales con orden propio
ORDINAL_MAPS = {
    "BsmtExposure": ["None", "No", "Mn", "Av", "Gd"],
    "BsmtFinType1": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFinType2": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "GarageFinish": ["None", "Unf", "RFn", "Fin"],
    "Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
    "Fence": ["None", "MnWw", "GdWo", "MnPrv", "GdPrv"],
    "LandSlope": ["Sev", "Mod", "Gtl"],
    "LotShape": ["IR3", "IR2", "IR1", "Reg"],
    "PavedDrive": ["N", "P", "Y"],
    "Utilities": ["NoSeWa", "AllPub"],
}
for c in QUALITY_COLS:
    ORDINAL_MAPS[c] = QUALITY_ORDER

ORDINAL_COLS = list(ORDINAL_MAPS.keys())

# Numérica con LotFrontage: imputar mediana. GarageYrBlt: mediana (0 rompería la escala).
NUMERIC_MEDIAN_COLS_HINT = ["LotFrontage", "GarageYrBlt", "MasVnrArea"]

DROP_COLS = ["Id"]


def load_and_clean(csv_path):
    """Carga el CSV, elimina outliers conocidos y agrega el target en escala log."""
    df = pd.read_csv(csv_path)
    if "SalePrice" in df.columns:
        outlier_mask = (df["GrLivArea"] > 4000) & (df["SalePrice"] < 300000)
        df = df.loc[~outlier_mask].reset_index(drop=True)
        df["SalePriceLog"] = np.log1p(df["SalePrice"])
    return df


def get_feature_lists(df):
    """Separa columnas en numéricas / ordinales / nominales, excluyendo target e Id."""
    exclude = set(DROP_COLS) | {"SalePrice", "SalePriceLog"}
    ordinal_cols = [c for c in ORDINAL_COLS if c in df.columns]
    numeric_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c not in exclude and c not in ordinal_cols
    ]
    nominal_cols = [
        c for c in df.select_dtypes(include=["object", "str"]).columns
        if c not in exclude and c not in ordinal_cols
    ]
    return numeric_cols, ordinal_cols, nominal_cols


def build_pipeline(df):
    """Construye el ColumnTransformer a partir de las columnas presentes en df."""
    numeric_cols, ordinal_cols, nominal_cols = get_feature_lists(df)

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    ordinal_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="None")),
        ("encode", OrdinalEncoder(
            categories=[ORDINAL_MAPS[c] for c in ordinal_cols],
            handle_unknown="use_encoded_value", unknown_value=-1,
        )),
    ])

    nominal_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="None")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    pre = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("ord", ordinal_pipe, ordinal_cols),
        ("nom", nominal_pipe, nominal_cols),
    ])
    return pre


if __name__ == "__main__":
    # ponytail: smoke test, no framework -- corre `python src/preprocessing.py`
    import os
    csv = os.path.join(os.path.dirname(__file__), "..", "train.csv")
    df = load_and_clean(csv)
    X = df.drop(columns=["SalePrice", "SalePriceLog"])
    y = df["SalePriceLog"]
    pipe = build_pipeline(X)
    X_t = pipe.fit_transform(X)
    assert X_t.shape[0] == X.shape[0]
    assert not np.isnan(X_t).any(), "NaNs remain after preprocessing"
    n_num, n_ord, n_nom = (len(v) for v in get_feature_lists(df))
    print(f"OK: {X.shape} -> {X_t.shape} (num={n_num} ord={n_ord} nom={n_nom} -> one-hot expande nom)")
    print("y range (log):", y.min(), y.max())
