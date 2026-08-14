# Proyecto #1 — Competencia de Modelación (MLP, Ames House Prices)

CC3092 Deep Learning y Sistemas Inteligentes. MLP en PyTorch que predice `SalePrice`
sobre el dataset Ames Housing (`train.csv`), evaluado por RMSE en un held-out entregado
el día de presentación.

## Estructura

```
proy-1-competencia/
├── train.csv              # dataset de entrenamiento (1168×81)
├── build_nb.py            # genera proy1_mlp.ipynb (no es entregable)
├── proy1_mlp.ipynb        # EDA, preprocesamiento, modelo, iteraciones
├── src/preprocessing.py   # ColumnTransformer reutilizado por notebook y predict.py
├── train.py                # entrena el modelo final, guarda artifacts/
├── predict.py              # carga artifacts/, predice sobre un CSV nuevo, calcula RMSE
├── artifacts/              # model.pt, pipeline.pkl, meta.json (generados por train.py)
├── docs/                   # explicadores HTML por batch + figuras
└── reporte.md / reporte.pdf
```

## Reproducir

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r ../../requirements.txt

# EDA (regenera el notebook desde build_nb.py)
python build_nb.py
jupyter nbconvert --to notebook --execute --inplace proy1_mlp.ipynb

# entrenar el modelo final
python train.py

# predecir sobre el held-out del día de presentación
python predict.py <ruta_al_csv_de_prueba>
```
