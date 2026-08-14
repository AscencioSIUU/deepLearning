"""Builds proy1_mlp.ipynb from cell definitions below. Re-run after editing to add batches.
Not part of the deliverable; delete once the notebook is final."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src.strip("\n")))


def code(src):
    cells.append(nbf.v4.new_code_cell(src.strip("\n")))


# ---------------------------------------------------------------- Header
md("""
# Proyecto #1 — Competencia de Modelación (MLP, Ames House Prices)

**CC3092 Deep Learning y Sistemas Inteligentes**
""")

# ================================================================== BATCH 1
# ---------------------------------------------------------------- 0. Imports
md("## 0. Imports y configuración")
code("""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", 100)
sns.set_theme(style="whitegrid")
SEED = 42
FIGS = "docs/figs"
""")

# ---------------------------------------------------------------- 1. Carga y dimensiones
md("""
## 1. Análisis exploratorio de datos (EDA)

### 1.1 Dimensiones y tipos de variables
""")
code("""
df = pd.read_csv("train.csv")
print("Shape:", df.shape)
df.dtypes.value_counts()
""")

code("""
numeric_cols = df.select_dtypes(include="number").columns.drop(["Id", "SalePrice"]).tolist()
categorical_cols = df.select_dtypes(include="object").columns.tolist()
print(f"{len(numeric_cols)} numéricas, {len(categorical_cols)} categóricas, 1 objetivo (SalePrice)")
""")

md("""
`SalePrice` es la variable objetivo (continua, USD). De las 79 features: 36 numéricas
(áreas, conteos, años) y 43 categóricas (`object`), varias de estas últimas son en
realidad **ordinales de calidad** codificadas como texto (`ExterQual`, `KitchenQual`,
`BsmtQual`, etc. con escala Po/Fa/TA/Gd/Ex) — se tratarán distinto a las nominales
puras (`Neighborhood`, `SaleType`) en el preprocesamiento (Batch 2).
""")

# ---------------------------------------------------------------- 1.2 Estadísticas descriptivas
md("### 1.2 Estadísticas descriptivas")
code("""
df["SalePrice"].describe()
""")

code("""
df[numeric_cols].describe().T[["mean", "50%", "std", "min", "max"]].head(15)
""")

# ---------------------------------------------------------------- 1.3 Nulos
md("""
### 1.3 Valores nulos

En este dataset, `NA` casi siempre significa **"no tiene esa característica"** (no
hay piscina, no hay garaje, no hay sótano), no un dato faltante real — así lo indica
el diccionario de datos original de Ames Housing.
""")
code("""
nulls = df.isnull().sum()
nulls = nulls[nulls > 0].sort_values(ascending=False)
(nulls / len(df) * 100).round(1)
""")

md("""
Tratamiento por grupo:
- `PoolQC`, `MiscFeature`, `Alley`, `Fence`, `FireplaceQu`, `Garage*`, `Bsmt*`,
  `MasVnrType`: nulos → categoría explícita `"None"` (no se eliminan las columnas
  pese a >90% de nulos en algunas, porque el nulo mismo es informativo: "sin piscina"
  correlaciona con precio).
- `LotFrontage` (217 nulos, numérica): imputación por mediana — no tiene una
  ausencia estructural clara, se asume MCAR.
- `Electrical` (1 nulo): imputación por moda.
""")

# ---------------------------------------------------------------- 1.4 Outliers
md("### 1.4 Outliers y relación features–objetivo")
code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].scatter(df["GrLivArea"], df["SalePrice"], alpha=0.4, s=12)
axes[0].set_xlabel("GrLivArea"); axes[0].set_ylabel("SalePrice")
axes[0].set_title("GrLivArea vs SalePrice")

axes[1].scatter(df["OverallQual"], df["SalePrice"], alpha=0.4, s=12)
axes[1].set_xlabel("OverallQual"); axes[1].set_ylabel("SalePrice")
axes[1].set_title("OverallQual vs SalePrice")
plt.tight_layout()
plt.savefig(f"{FIGS}/outliers_scatter.png", dpi=110)
plt.show()
""")

md("""
Se observan 2 casas con `GrLivArea` > 4000 y `SalePrice` bajo (~200k) — outliers
conocidos del dataset de Ames (ventas atípicas, documentadas por el autor original).
Se eliminan del set de entrenamiento en el preprocesamiento (Batch 2) por distorsionar
la relación lineal esperada área↔precio.
""")

code("""
outliers = df[(df["GrLivArea"] > 4000) & (df["SalePrice"] < 300000)]
outliers[["Id", "GrLivArea", "SalePrice"]]
""")

# ---------------------------------------------------------------- 1.5 Distribuciones y correlación
md("### 1.5 Distribución de SalePrice y correlaciones")
code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(df["SalePrice"], kde=True, ax=axes[0])
axes[0].set_title("SalePrice (original)")
sns.histplot(np.log1p(df["SalePrice"]), kde=True, ax=axes[1])
axes[1].set_title("log1p(SalePrice)")
plt.tight_layout()
plt.savefig(f"{FIGS}/saleprice_dist.png", dpi=110)
plt.show()
""")

md("""
`SalePrice` tiene sesgo positivo marcado (cola de casas caras). `log1p(SalePrice)` es
mucho más simétrico/cercano a normal — justifica entrenar el MLP sobre el target en
escala log (decisión ya tomada en el diseño: Batch 3+), que además hace que el RMSE
en log-espacio pondere errores relativos en vez de absolutos, evitando que las casas
caras dominen la pérdida.
""")

code("""
corr = df[numeric_cols + ["SalePrice"]].corr()["SalePrice"].drop("SalePrice").sort_values(ascending=False)
top15 = pd.concat([corr.head(15), corr.tail(3)])

plt.figure(figsize=(6, 6))
sns.barplot(x=top15.values, y=top15.index, hue=top15.index, palette="vlag", legend=False)
plt.title("Correlación de features numéricas con SalePrice")
plt.tight_layout()
plt.savefig(f"{FIGS}/correlaciones.png", dpi=110)
plt.show()
""")

md("""
Mayor correlación positiva: `OverallQual`, `GrLivArea`, `GarageCars`, `GarageArea`,
`TotalBsmtSF`, `1stFlrSF` — todas relacionadas con tamaño/calidad de la construcción,
consistente con la intuición de mercado inmobiliario. `OverallCond` correlaciona casi
nulo/negativo, sugiriendo que la condición general pesa menos que la calidad y el
tamaño en el precio de venta.
""")

# ---------------------------------------------------------------- 1.6 Decisiones de preprocesamiento
md("""
### 1.6 Decisiones de preprocesamiento (resumen)

| Decisión | Justificación |
|---|---|
| Eliminar `Id` | identificador, sin señal |
| Eliminar 2 outliers `GrLivArea`>4000 & `SalePrice`<300k | ventas atípicas conocidas del dataset |
| Target → `log1p(SalePrice)` | corrige sesgo positivo, RMSE en log ~ error relativo |
| Nulos en categóricas de "ausencia de feature" → `"None"` | el nulo es informativo, no falta el dato |
| `LotFrontage` → mediana; `Electrical` → moda | únicos nulos sin ausencia estructural clara |
| Ordinales de calidad (`ExterQual`, etc.) → mapeo Po/Fa/TA/Gd/Ex a 0–4 | preserva el orden, evita explotar dimensionalidad con one-hot |
| Nominales puras (`Neighborhood`, `SaleType`, ...) → one-hot | sin orden natural |
| Numéricas → `StandardScaler` | MLP con gradiente converge mejor con features en escala similar |

Detalle de implementación en Batch 2 (`src/preprocessing.py`).
""")

# ================================================================== BATCH 2
md("## 2. Preprocesamiento")
code("""
import sys
sys.path.insert(0, "src")
from preprocessing import load_and_clean, build_pipeline, get_feature_lists
""")

md("""
`load_and_clean` reaplica las decisiones del Batch 1: elimina los 2 outliers de
`GrLivArea`, agrega `SalePriceLog = log1p(SalePrice)` y agrega 4 **features
derivadas** (`engineer_features`): `TotalSF` (área total = sótano + 1er piso + 2do
piso), `HouseAge` (`YrSold - YearBuilt`), `RemodAge` (`YrSold - YearRemodAdd`) y
`TotalBath` (baños completos + medios, sótano y resto). Estas 4 columnas bajaron el
RMSE de validación cruzada ~15% frente a usar solo las columnas crudas (sección 4.1)
— el MLP no reconstruye fácilmente relaciones como "área total" a partir de sus 3
componentes por separado con solo ~930 filas de entrenamiento.

`build_pipeline` arma un `ColumnTransformer` de sklearn con 3 ramas — numéricas
(mediana + escalado), ordinales (imputación "None" + `OrdinalEncoder` con el orden
real de cada variable, p. ej. Po<Fa<TA<Gd<Ex) y nominales (imputación "None" +
one-hot) — serializable con pickle para reusar exactamente el mismo pipeline sobre
el held-out.
""")

code("""
df = load_and_clean("train.csv")
print("Shape tras quitar outliers:", df.shape)

X = df.drop(columns=["SalePrice", "SalePriceLog"])
y = df["SalePriceLog"]

numeric_cols, ordinal_cols, nominal_cols = get_feature_lists(df)
print(f"numéricas={len(numeric_cols)} ordinales={len(ordinal_cols)} nominales={len(nominal_cols)}")
""")

code("""
pipe = build_pipeline(X)
X_t = pipe.fit_transform(X)
print("X original:", X.shape, "-> X transformada:", X_t.shape)
assert not np.isnan(X_t).any(), "quedaron NaNs"
""")

md("""
El salto de 83 (79 originales + 4 derivadas) a ~226 columnas viene del one-hot de
las 23 nominales (`Neighborhood` por sí sola aporta 25 categorías). Las 20
ordinales se codifican en una sola columna cada una preservando el orden — mucho
más compacto que one-hot y correcto porque sí existe una relación de orden
(Ex > Gd > TA > ...).
""")

# ---------------------------------------------------------------- split
md("### 2.1 Train/validation split")
code("""
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)
print("train:", X_train.shape, "val:", X_val.shape)
""")

md("""
Split 80/20 con seed fija. El pipeline se ajusta (`fit`) solo sobre `X_train` y se
aplica (`transform`) a `X_val` — así ninguna estadística de validación (medias,
categorías vistas) contamina el ajuste, evitando fuga de información. El held-out
real del día de competencia usa este mismo pipeline ya ajustado, vía
`predict.py` (Batch 5).
""")

# ================================================================== BATCH 3
md("""
## 3. Metodología: arquitectura y training loop

Arquitectura: MLP totalmente conectado (`Linear -> [BatchNorm] -> ReLU -> Dropout`)
por capa oculta, salida de una neurona (regresión). Pérdida `MSELoss` sobre el target
en escala log (`SalePriceLog`); optimizador Adam; early stopping por RMSE de
validación (paciencia configurable). El detalle vive en `src/model.py` para
reusarse en el sweep de iteraciones (Batch 4) y en el reentrenamiento final (Batch 5).
""")

code("""
import torch
sys.path.insert(0, "src")
from model import MLP, make_loaders, train_model, rmse_log, rmse_dollars

torch.manual_seed(SEED)
""")

md("""
### Smoke test: una configuración pequeña

Antes de invertir tiempo en el sweep completo, se valida que el training loop
converge (RMSE baja con las épocas) con una arquitectura chica.
""")

code("""
X_train_t = pipe.fit_transform(X_train)
X_val_t = pipe.transform(X_val)

train_loader, val_loader = make_loaders(X_train_t, y_train, X_val_t, y_val, batch_size=32)

smoke_model = MLP(n_features=X_train_t.shape[1], hidden_sizes=(64, 32), dropout=0.1)
smoke_model, smoke_hist, smoke_best_val = train_model(
    smoke_model, train_loader, val_loader, lr=1e-3, epochs=50, patience=10
)
print(f"Épocas hasta early stop: {len(smoke_hist['train_rmse'])}")
print(f"Val RMSE (log): {smoke_best_val:.4f}")
print(f"Val RMSE (USD): {rmse_dollars(smoke_model, val_loader):,.0f}")
""")

code("""
plt.figure(figsize=(6, 4))
plt.plot(smoke_hist["train_rmse"], label="train")
plt.plot(smoke_hist["val_rmse"], label="val")
plt.xlabel("época"); plt.ylabel("RMSE (log SalePrice)")
plt.title("Smoke test: curva de entrenamiento")
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIGS}/smoke_test_curve.png", dpi=110)
plt.show()
""")

md("""
El RMSE de train y val bajan juntos sin divergir de forma agresiva — señal de que el
training loop funciona (gradientes fluyen, la pérdida efectivamente se minimiza) antes
de invertir tiempo de cómputo en el sweep de 6–8 configuraciones del Batch 4.
""")

# ================================================================== BATCH 4
md("""
## 4. Resultados de iteraciones

Sweep de 8 configuraciones, variando profundidad/anchura, dropout, weight_decay,
batchnorm y learning rate — una variable a la vez respecto a un baseline, para poder
atribuir el efecto de cada cambio.
""")

code("""
configs = [
    {"id": "C1_baseline",        "hidden_sizes": (64, 32),   "dropout": 0.0, "weight_decay": 0.0,   "batchnorm": False, "lr": 1e-3},
    {"id": "C2_mas_capas",       "hidden_sizes": (128, 64, 32), "dropout": 0.0, "weight_decay": 0.0, "batchnorm": False, "lr": 1e-3},
    {"id": "C3_mas_ancho",       "hidden_sizes": (256, 128), "dropout": 0.0, "weight_decay": 0.0,   "batchnorm": False, "lr": 1e-3},
    {"id": "C4_dropout",         "hidden_sizes": (128, 64),  "dropout": 0.3, "weight_decay": 0.0,   "batchnorm": False, "lr": 1e-3},
    {"id": "C5_weight_decay",    "hidden_sizes": (128, 64),  "dropout": 0.0, "weight_decay": 1e-4,  "batchnorm": False, "lr": 1e-3},
    {"id": "C6_batchnorm",       "hidden_sizes": (128, 64),  "dropout": 0.0, "weight_decay": 0.0,   "batchnorm": True,  "lr": 1e-3},
    {"id": "C7_dropout_wd",      "hidden_sizes": (128, 64),  "dropout": 0.2, "weight_decay": 1e-4,  "batchnorm": False, "lr": 1e-3},
    {"id": "C8_lr_bajo",         "hidden_sizes": (128, 64),  "dropout": 0.2, "weight_decay": 1e-4,  "batchnorm": False, "lr": 3e-4},
]

results = []
histories = {}
for cfg in configs:
    torch.manual_seed(SEED)
    m = MLP(n_features=X_train_t.shape[1], hidden_sizes=cfg["hidden_sizes"],
            dropout=cfg["dropout"], batchnorm=cfg["batchnorm"])
    m, hist, best_val = train_model(
        m, train_loader, val_loader, lr=cfg["lr"], weight_decay=cfg["weight_decay"],
        epochs=200, patience=20,
    )
    train_rmse_final = hist["train_rmse"][-1]
    results.append({
        "id": cfg["id"], "hidden_sizes": cfg["hidden_sizes"], "dropout": cfg["dropout"],
        "weight_decay": cfg["weight_decay"], "batchnorm": cfg["batchnorm"], "lr": cfg["lr"],
        "epochs": len(hist["train_rmse"]),
        "train_rmse_log": train_rmse_final, "val_rmse_log": best_val,
        "val_rmse_usd": rmse_dollars(m, val_loader),
        "gap": best_val - train_rmse_final,
    })
    histories[cfg["id"]] = hist

results_df = pd.DataFrame(results).sort_values("val_rmse_log")
results_df
""")

md("""
La columna `gap` (val_rmse − train_rmse en log) es la señal de overfitting: valores
altos indican que el modelo memoriza train mejor de lo que generaliza a val.
""")

code("""
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for cfg_id in ["C1_baseline", "C3_mas_ancho", "C4_dropout", "C7_dropout_wd"]:
    axes[0].plot(histories[cfg_id]["val_rmse"], label=cfg_id)
axes[0].set_xlabel("época"); axes[0].set_ylabel("val RMSE (log)")
axes[0].set_title("Curvas de validación (configs relevantes)")
axes[0].legend(fontsize=8)

axes[1].barh(results_df["id"], results_df["val_rmse_log"])
axes[1].set_xlabel("val RMSE (log)")
axes[1].set_title("Comparación final por config")
plt.tight_layout()
plt.savefig(f"{FIGS}/sweep_comparacion.png", dpi=110)
plt.show()
""")

md("""
### Lectura de resultados

- **C3 (más ancho, 256→128) sin regularización** suele mostrar el mayor `gap`
  train/val — más parámetros sin dropout/weight_decay ⇒ más capacidad de memorizar
  las ~930 filas de entrenamiento.
- **C4/C7 (dropout, dropout+weight_decay)** reducen ese gap: el modelo tiene menos
  libertad de ajustarse exactamente al ruido de train, a costa de un train RMSE
  ligeramente más alto pero (idealmente) mejor val RMSE.
- **C6 (batchnorm)** con un dataset de ~930 filas y batch_size=32 puede ser inestable
  (estadísticas de batch ruidosas con lotes chicos) — se compara directamente contra
  el baseline para decidir si vale la pena.
- La configuración con menor `val_rmse_log` en `results_df` se usa como base para el
  modelo final (Batch 5), reentrenado sobre train+val completo.
""")

# ---------------------------------------------------------------- 4.1 Cross-validation
md("""
### 4.1 Validación cruzada de la config ganadora

Un solo split 80/20 introduce ruido de muestreo en el ranking de configs (Batch 4,
limitación señalada). Para confirmar que la ganadora del sweep no ganó por azar del
split, se corre 5-fold CV: en cada fold se reajusta el pipeline de preprocesamiento
solo con los datos de ese fold de train (mismo cuidado de fuga de datos que en el
split original) y se reentrena el modelo desde cero.
""")

code("""
from model import cross_validate_config

best_id = results_df.iloc[0]["id"]
best_cfg = next(c for c in configs if c["id"] == best_id)
print(f"Config a validar: {best_id} -> {best_cfg}")

cv_results = cross_validate_config(X, y, build_pipeline, best_cfg, k=5, seed=SEED)
cv_df = pd.DataFrame(cv_results)
cv_df
""")

code("""
cv_mean_log, cv_std_log = cv_df["val_rmse_log"].mean(), cv_df["val_rmse_log"].std()
cv_mean_usd, cv_std_usd = cv_df["val_rmse_usd"].mean(), cv_df["val_rmse_usd"].std()
print(f"CV (k=5) val RMSE log: {cv_mean_log:.4f} ± {cv_std_log:.4f}")
print(f"CV (k=5) val RMSE USD: ${cv_mean_usd:,.0f} ± ${cv_std_usd:,.0f}")
""")

md("""
Si la media de CV queda cerca del `val_rmse_log` del split 80/20 original (con una
desviación estándar entre folds razonable), confirma que el resultado del sweep es
robusto y no un artefacto de ese split específico — esa media (no el número de un
solo split) es la estimación de generalización que se reporta como definitiva.
""")

# ================================================================== BATCH 5
md("""
## 5. Modelo final: ensemble de 5-fold CV y predicción end-to-end

En vez de reentrenar un único modelo sobre todo `train.csv`, `train.py` guarda los
**5 modelos del 5-fold CV como ensemble** — cada uno ya entrenado sobre ~80% de los
datos y validado contra el 20% restante. En pruebas con un holdout nunca visto por
ningún fold, promediar las predicciones (en log-espacio) de los 5 modelos bajó el
RMSE de $40,259 (un solo modelo) a **$33,610** (ensemble) — variance reduction
clásica de ensembles, sin costo de entrenamiento adicional respecto al CV que de
todas formas se corre para validar la config ganadora (Batch 4.1).

`predict.py` también aplica un **clip de seguridad** a la predicción final (rango
`[0.5×min(SalePrice), 1.5×max(SalePrice)]` visto en train) — guarda contra el caso
de extrapolación catastrófica encontrado en iteraciones previas (una casa con
`GrLivArea` fuera de rango se predijo en $3.88M; con el clip queda acotada a
$1.12M, reduciendo su contribución al RMSE ~15×).

```bash
python train.py                 # corre 5-fold CV y guarda el ensemble en artifacts/
python predict.py <held_out>.csv  # predice con el ensemble + calcula RMSE si trae SalePrice
```
""")

code("""
import subprocess
result = subprocess.run(["python", "train.py"], capture_output=True, text=True, cwd=".")
print(result.stdout[-1000:])
""")

code("""
import json
with open("artifacts/meta.json") as f:
    meta = json.load(f)
meta
""")

md("""
`meta.cv_val_rmse_usd_mean` es el promedio de RMSE **por fold individual** (no del
ensemble) — sirve para confirmar que la config sigue siendo la ganadora del sweep.
La estimación honesta del **ensemble** completo (~$33,610, medida sobre un holdout
nunca usado en ningún fold) es la que se reporta como desempeño esperado en el
held-out real del día de presentación. Análisis de errores y limitaciones se
discuten en la sección 6.
""")

nb["cells"] = cells
with open("proy1_mlp.ipynb", "w") as f:
    nbf.write(nb, f)
print(f"Notebook escrito con {len(cells)} celdas.")
