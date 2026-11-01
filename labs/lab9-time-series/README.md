# Laboratorio 9 — Series de tiempo y forecasting

Ernesto Ascencio · Carné 23009 · CC3092 Deep Learning y Sistemas Inteligentes

Pronóstico del consumo eléctrico horario (kWh) de las próximas 24 h para 100 clientes del dataset
[ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014) (UCI).
Se comparan 17 modelos, desde un pronóstico ingenuo hasta un modelo fundacional, todos con el mismo protocolo de evaluación.

## Qué se hizo

1. **Datos**: registros de 15 min pasados a kWh por hora, 100 clientes elegidos con semilla 42, corrección de los
   cambios de horario y z-score por cliente con estadísticos de entrenamiento.
2. **Split cronológico**: train hasta 2014-06-30, validación jul–sep 2014 y prueba oct–dic 2014. Los asserts del
   notebook verifican que no haya fugas de información (escalado, early stopping y covariables).
3. **Evaluación única** `backtest(modelo)`: 92 orígenes diarios en prueba; cada día el modelo ve los datos hasta las
   23:00 y pronostica las 24 h siguientes. Métricas: MAE, RMSE, WAPE, MAPE, MASE, MASE de la semana de Navidad,
   pinball y cobertura P10–P90.
4. **Modelos**:
   - Baselines: Naive, Seasonal Naive 24 h y 168 h, MSTL (StatsForecast).
   - LightGBM global con rezagos y calendario.
   - Propios en PyTorch: MLP, DLinear, NLinear, LSTM (autorregresiva y multi-salida) y Seq2seq.
   - NeuralForecast: NHITS, TCN, PatchTST, TFT y DeepAR.
   - Chronos-Bolt zero-shot, sin entrenamiento.
5. **Experimentos**:
   - 23 iteraciones de hiperparámetros: ventana L, meses de historia, normalización, tamaño, lr y batch.
   - Estrategias multi-paso y modelo local contra global.
   - Reentrenamiento mensual, 3 semillas y efecto de las covariables de calendario.

## Resultados (prueba, horizonte de 24 h)

MASE menor a 1 significa que el modelo supera al Seasonal Naive de 24 h dentro del entrenamiento.

| Modelo | MASE | MAE (kWh) | Entrenamiento |
|---|--:|--:|--:|
| Chronos-Bolt (zero-shot) | 0.785 | 24.3 | — |
| NHITS | 0.830 | 26.1 | 207 s |
| TFT | 0.849 | 26.8 | 3.1 h |
| MLP | 0.876 | 27.9 | 26 s |
| LightGBM | 0.878 | 27.6 | 222 s |
| Seasonal Naive 24 h | 1.040 | 32.5 | — |

La tabla completa de los 17 modelos está en `results/comparacion.csv`.

## Archivos

| Archivo | Contenido |
|---|---|
| `notebook.ipynb` | Notebook completo, comentado y ya ejecutado (se puede leer sin correrlo) |
| `build_nb.py` | Genera `notebook.ipynb`; las celdas se editan aquí |
| `run_nb.sh` | Regenera el notebook y lo ejecuta |
| `requirements.txt` | Dependencias |
| `results/pronosticos_prueba.parquet` | Pronósticos de prueba de cada modelo: `unique_id, ds, modelo, y_hat` |
| `results/*.csv` | Comparación, iteraciones, MAE por paso y métricas por cliente |
| `figs/` | Gráficas del notebook |
| `data/` | No se versiona: dataset descargado y caché de entrenamientos |

## Cómo correrlo

Requiere Python 3.12. Probado en un Mac con Apple Silicon: usa la GPU (MPS) si existe y, si no, la CPU.

```sh
cd labs/lab9-time-series
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name lab9
./run_nb.sh
```

- La primera ejecución descarga el dataset de UCI (≈250 MB) en `data/`.
- Desde cero tarda varias horas (TFT solo tarda ≈3 h).
- Cada modelo entrenado se guarda en `data/cache/`. Si la ejecución se corta, `./run_nb.sh` sigue desde donde quedó.
- `run_nb.sh` fija `DYLD_LIBRARY_PATH` para que LightGBM y PyTorch compartan una sola `libomp` en macOS (con dos, el
  kernel falla con segfault).
