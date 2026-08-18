import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

QUALITY_ORDER = ["None", "Po", "Fa", "TA", "Gd", "Ex"]
QUALITY_COLS = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
    "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC",
]

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
DROP_COLS = ["Id"]


def engineer_features(df):
    df = df.copy()
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
    df["TotalBath"] = (df["FullBath"] + 0.5 * df["HalfBath"]
                       + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"])
    df["Qual_x_TotalSF"] = df["OverallQual"] * df["TotalSF"]
    df["OverallGrade"] = df["OverallQual"] * df["OverallCond"]
    df["TotalPorchSF"] = (df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"]
                          + df["ScreenPorch"] + df["WoodDeckSF"])
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    df["HasBsmt"] = (df["TotalBsmtSF"] > 0).astype(int)
    df["Has2ndFloor"] = (df["2ndFlrSF"] > 0).astype(int)
    df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
    df["IsRemodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)
    for c in ["LotArea", "GrLivArea", "TotalSF", "1stFlrSF", "TotalBsmtSF",
              "GarageArea", "BsmtFinSF1", "Qual_x_TotalSF", "TotalPorchSF"]:
        df[c + "_log"] = np.log1p(df[c].clip(lower=0))
    return df


def load_and_clean(csv_path):
    df = pd.read_csv(csv_path)
    if "SalePrice" in df.columns:
        outlier_mask = (df["GrLivArea"] > 4000) & (df["SalePrice"] < 300000)
        df = df.loc[~outlier_mask].reset_index(drop=True)
        df["SalePriceLog"] = np.log1p(df["SalePrice"])
    df = engineer_features(df)
    return df


def get_feature_lists(df):
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

    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("ord", ordinal_pipe, ordinal_cols),
        ("nom", nominal_pipe, nominal_cols),
    ])
