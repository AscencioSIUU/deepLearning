# Proyecto #1 — Competencia de Modelación (MLP, Ames House Prices)

CC3092 Deep Learning y Sistemas Inteligentes. MLP en PyTorch que predice `SalePrice`
sobre el dataset Ames Housing (`train.csv`), evaluado por RMSE sobre un held-out.

**Modelo final**: ensemble de 25 MLPs (5-fold CV × 5 semillas). Cada MLP tiene una capa
oculta de 96 neuronas con ReLU, dropout 0.1 y una skip connection lineal; se entrena
sobre `log1p(SalePrice)` estandarizado, con Adam + `ReduceLROnPlateau` y early stopping.
`val RMSE (5-fold CV multi-seed) = $18,795 USD`; RMSE sobre el dataset de prueba de la
competencia = $20,881 USD. Ver `reporte.md` para el detalle.

## Estructura

```
proy-1-competencia/
├── train.csv                 # dataset de entrenamiento (1168×81)
├── test_features-clase.csv   # dataset de prueba de la competencia (features)
├── src/preprocessing.py      # limpieza, feature engineering y ColumnTransformer
├── src/model.py              # arquitectura MLP y training loop
├── src/ensemble.py           # entrenamiento y predicción del ensemble por CV
├── train.py                  # corre el sweep + CV, guarda el ensemble en artifacts/
├── predict.py                # carga el ensemble, predice sobre un CSV nuevo, calcula RMSE
├── artifacts/                # ensemble.pkl y meta.json (generados por train.py)
├── docs/figs/                # figuras del EDA
└── reporte.md / reporte.pdf
```

## Reproducir

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r ../../requirements.txt

# entrenar el modelo final (regenera artifacts/ensemble.pkl y artifacts/meta.json)
python train.py

# predecir sobre un dataset de prueba
python predict.py test_features-clase.csv
```

## Predecir

```bash
python predict.py <ruta_al_csv_de_prueba>
```

- El CSV de entrada debe traer las mismas columnas de features que `train.csv`
  (con o sin `SalePrice`; la columna `Id` es opcional, si falta se usa el índice).
- Genera `predictions/<nombre>_predictions.csv` con columnas **`Id,Prediction`**
  — mismos `Id` que el archivo de entrada.
- Si el CSV trae `SalePrice`, además imprime el RMSE (USD y log) en consola.
- No hace falta reentrenar: usa el ensemble ya guardado en `artifacts/`.
