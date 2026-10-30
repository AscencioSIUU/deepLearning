"""Builds notebook.ipynb from the cell definitions below. Re-run after editing to add batches.
Not part of the deliverable; delete once the notebook is final."""
import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
BATCH = 1                     # UPTO=n en el entorno genera solo las celdas de los batches <= n


def batch(n):
    global BATCH
    BATCH = n


def md(src):
    cells.append((BATCH, nbf.v4.new_markdown_cell(src.strip("\n"))))


def code(src):
    cells.append((BATCH, nbf.v4.new_code_cell(src.strip("\n"))))


# ================================================================== Header
md("""
# Laboratorio #9 — Series de tiempo y forecasting

**CC3092 Deep Learning y Sistemas Inteligentes**

Pronóstico *day-ahead* (H = 24 h) del consumo eléctrico horario de clientes de Portugal
(UCI ElectricityLoadDiagrams20112014). Pipeline: datos crudos → series horarias limpias → ventanas →
tensores → modelos (baselines, LightGBM, redes propias en PyTorch, NeuralForecast, Chronos) evaluados
con una sola función `backtest()` sobre 92 orígenes diarios de prueba.

- **Periodo:** 2012-01-01 a 2014-12-31 · **Train** 2012-01 → 2014-06 · **Val** 2014-07 → 2014-09 · **Test** 2014-10 → 2014-12
- **Panel:** 100 clientes elegidos con semilla fija entre los que tienen datos completos desde 2012-01-01
- **Hardware:** Apple M5 Pro (24 GB, sin CUDA): redes propias en MPS, librerías en CPU/MPS
- **Ejecución:** `./run_nb.sh` (macOS: exporta `DYLD_LIBRARY_PATH` a `torch/lib` para que LightGBM y PyTorch compartan
  una sola libomp; con dos runtimes de OpenMP el kernel hace segfault)
""")

# ================================================================== 0. Imports
md("## 0. Imports y configuración")
code("""
import os, time, json, pickle, random, warnings, zipfile, urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import holidays
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DATA, FIGS, RESULTS = Path("data"), Path("figs"), Path("results")
CACHE = DATA / "cache"
for p in (DATA, FIGS, RESULTS, CACHE):
    p.mkdir(exist_ok=True)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
HARDWARE = "Apple M5 Pro (24 GB), " + ("MPS" if DEVICE.type == "mps" else "CPU")

L, H = 168, 24                                    # ventana de entrada y horizonte (horas)
TRAIN_START = pd.Timestamp("2012-01-01")
VAL_START, TEST_START = pd.Timestamp("2014-07-01"), pd.Timestamp("2014-10-01")
END = pd.Timestamp("2014-12-31 23:00")
TRAIN_END, VAL_END = VAL_START - pd.Timedelta("1h"), TEST_START - pd.Timedelta("1h")
N_CLIENTS = 100

def cached(name, fn):
    \"\"\"Ejecuta fn() una sola vez y guarda su resultado en data/cache (re-ejecutar el notebook es rápido).\"\"\"
    p = CACHE / f"{name}.pkl"
    if p.exists():
        return pickle.load(open(p, "rb"))
    out = fn()
    pickle.dump(out, open(p, "wb"))
    return out

print("torch", torch.__version__, "| device", DEVICE)
""")

# ================================================================== 1. Datos
md("""
## 1. Datos

`LD2011_2014.txt`: 140,256 registros de 15 min × 370 clientes, en kW. El consumo de un registro en kWh es
kW × 0.25 h, así que el consumo horario en kWh es la suma de los 4 registros dividida entre 4.
""")
code("""
zip_path, txt_path, raw_path = DATA / "electricity.zip", DATA / "LD2011_2014.txt", DATA / "raw.parquet"
if not raw_path.exists():
    if not txt_path.exists():
        url = ("https://archive.ics.uci.edu/static/public/321/"
               "electricityloaddiagrams20112014.zip")
        if not zip_path.exists():
            urllib.request.urlretrieve(url, zip_path)
        zipfile.ZipFile(zip_path).extract("LD2011_2014.txt", DATA)
    pd.read_csv(txt_path, sep=";", decimal=",", index_col=0, parse_dates=True).to_parquet(raw_path)
raw = pd.read_parquet(raw_path)
print(raw.shape, "|", raw.index[0], "→", raw.index[-1])
""")

md("""
### 1.1 Agregación horaria y cambios de horario

Las marcas de tiempo del archivo son ambiguas (empiezan en 00:15 y terminan en 2015-01-01 00:00, lo que sugiere
"fin de intervalo"), pero la anomalía documentada de los cambios de horario cae exactamente en los registros
01:00–01:45: en marzo (día de 23 h) esos registros valen 0 y en octubre (día de 25 h) acumulan dos horas. Por eso
se agrupa por la hora de la etiqueta (`resample("h")`, registros h:00–h:45), y así la anomalía queda contenida en
una sola hora por día de cambio.
""")
code("""
DST_DAYS = ["2012-03-25", "2012-10-28", "2013-03-31", "2013-10-27", "2014-03-30", "2014-10-26"]
print(raw.loc["2012-03-25 00:30":"2012-03-25 02:15", ["MT_010", "MT_100"]].T.round(1))
print(raw.loc["2012-10-28 00:30":"2012-10-28 02:15", ["MT_010", "MT_100"]].T.round(1))
hourly = raw.resample("h").sum().div(4).loc[TRAIN_START:END]       # kWh por hora
print("serie horaria:", hourly.shape)
""")

md("""
### 1.2 Filtro de clientes y panel de trabajo

Un cliente con "datos completos desde 2012-01-01" es uno cuyo primer registro distinto de cero es anterior o igual
al primer registro de 2012 (00:15; antes de darse de alta, el archivo rellena con ceros) y que no tiene ningún día completo en cero entre
2012 y 2014 (cortes o bajas). Del panel resultante se toman 100 clientes con semilla fija, el mismo panel para todos
los modelos.
""")
code("""
first_nz = raw.ne(0).idxmax()
started = first_nz <= TRAIN_START + pd.Timedelta("15min")   # 160 clientes arrancan justo en el registro 00:15 de 2012
zero_days = hourly.resample("D").sum().eq(0).sum()
complete = started & (zero_days == 0)
print(f"clientes totales: {raw.shape[1]} | con datos desde 2012-01-01: {started.sum()} "
      f"| sin días en cero: {complete.sum()}")

rng = np.random.default_rng(SEED)
CLIENTS = sorted(rng.choice(complete[complete].index, N_CLIENTS, replace=False))
Y = hourly[CLIENTS].copy()
print(f"panel: {Y.shape[1]} clientes × {Y.shape[0]:,} horas = {Y.size:,} observaciones horarias")
""")

md("""
**Cambios de horario.** Para los 6 días de cambio (último domingo de marzo y de octubre) la hora 01:00 se marca como
faltante y se interpola linealmente entre 00:00 y 02:00. Es más simple y robusto que dividir entre 2 en octubre:
no todos los clientes muestran el doble conteo (algunos medidores ya lo corrigen) y así se trata igual a todos.
La primera hora del panel también se rellena: a los clientes que arrancan en el registro 00:15 de 2012 les falta
un cuarto de hora. Son 7 de 26,304 horas por serie, por lo que el efecto sobre el entrenamiento es despreciable, pero sin la corrección
aparecen picos y valles artificiales que el modelo intentaría aprender.
""")
code("""
dst_hours = pd.DatetimeIndex([pd.Timestamp(d) + pd.Timedelta("1h") for d in DST_DAYS])
before = Y.loc[dst_hours[:2]].iloc[:, :4].round(1)
Y.loc[dst_hours] = np.nan
Y.iloc[0] = np.nan            # 2012-01-01 00:00 queda incompleta (sin el registro 00:00) en los clientes que arrancan ahí
Y = Y.interpolate(method="linear").bfill()
assert not Y.isna().any().any()
print(pd.concat({"antes": before, "después": Y.loc[dst_hours[:2]].iloc[:, :4].round(1)}))
T = len(Y)
t_val, t_test = Y.index.get_loc(VAL_START), Y.index.get_loc(TEST_START)
print(f"T = {T:,} | train {t_val:,} h | val {t_test - t_val:,} h | test {T - t_test:,} h")
""")

# ================================================================== 2. Exploración
md("""
## 2. Exploración y preparación de la serie

### 2.1 Perfiles por hora y por día de la semana

Para que los clientes grandes no dominen, cada serie se divide entre su media de entrenamiento antes de promediar.
""")
code("""
pt_hol = holidays.Portugal(years=[2012, 2013, 2014])
rel = Y / Y.loc[:TRAIN_END].mean()
panel_rel = rel.mean(axis=1)                       # consumo relativo promedio del panel
days = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
prof = panel_rel.groupby([panel_rel.index.dayofweek, panel_rel.index.hour]).mean().unstack(0)
for d in range(7):
    ax[0].plot(prof.index, prof[d], label=days[d], lw=2 if d >= 5 else 1)
ax[0].set(xlabel="hora del día", ylabel="consumo / media del cliente", title="Perfil horario por día")
ax[0].legend(ncol=4, fontsize=7)
by_dow = panel_rel.groupby(panel_rel.index.dayofweek).mean()
ax[1].bar(days, by_dow, color=["C0"] * 5 + ["C3"] * 2)
ax[1].set(ylim=(0.7, 1.1), title="Consumo medio por día de la semana")
plt.tight_layout(); plt.savefig(FIGS / "perfiles.png", dpi=150); plt.show()
print(prof.round(2).loc[[4, 12, 19]])
""")

md("### 2.2 Feriados de Portugal, Navidad y Año Nuevo")
code("""
daily = panel_rel.resample("D").mean()
hol_days = pd.DatetimeIndex([d for d in pt_hol if pd.Timestamp(d) in daily.index])
fig, ax = plt.subplots(2, 1, figsize=(11, 5.5))
ax[0].plot(daily.index, daily, lw=0.7)
ax[0].scatter(hol_days, daily.loc[hol_days], color="C3", s=14, zorder=3, label="feriado PT")
for s in (VAL_START, TEST_START):
    ax[0].axvline(s, color="k", ls="--", lw=0.8)
ax[0].set(title="Consumo diario relativo del panel (train | val | test)", ylabel="relativo"); ax[0].legend()
for y0 in (2012, 2013):
    w = daily.loc[f"{y0}-12-15":f"{y0 + 1}-01-10"]
    ax[1].plot(np.arange(len(w)), w.values, marker=".", label=f"{y0}/{y0 + 1 - 2000:02d}")
ax[1].set_xticks(range(0, 27, 2), [f"{d:%d-%b}" for d in pd.date_range("2013-12-15", periods=27)[::2]], fontsize=7)
ax[1].axvline(10, color="C3", ls=":"); ax[1].axvline(17, color="C3", ls=":")
ax[1].set(title="Navidad y Año Nuevo (15-dic a 10-ene)", ylabel="relativo"); ax[1].legend()
plt.tight_layout(); plt.savefig(FIGS / "feriados.png", dpi=150); plt.show()

def vs_same_weekday(day):
    \"\"\"Consumo del día / media del mismo día de la semana en las 3 semanas previas.\"\"\"
    d = pd.Timestamp(day)
    return daily.loc[d] / daily.loc[[d - pd.Timedelta(weeks=k) for k in (1, 2, 3)]].mean()

rows = {d: vs_same_weekday(d) for d in ["2012-12-24", "2012-12-25", "2012-12-31", "2013-01-01",
                                       "2013-12-24", "2013-12-25", "2013-12-31", "2014-01-01"]}
print(pd.Series(rows, name="ratio vs mismo día previo").round(3))
print("feriado (no fin de semana) vs mismo día previo:",
      round(np.mean([vs_same_weekday(d) for d in hol_days if d.dayofweek < 5 and d > pd.Timestamp("2012-01-22")]), 3))
""")

md("""
### 2.3 Escalas entre clientes y normalización
""")
code("""
means = Y.loc[:TRAIN_END].mean()
print(f"consumo medio por cliente (kWh/h): min {means.min():.2f} | mediana {means.median():.1f} "
      f"| max {means.max():,.0f} | max/min = {means.max() / means.min():,.0f}×")
print(means.describe().round(1).to_dict())
""")
md("""
Las escalas difieren en varios órdenes de magnitud, así que una normalización **global** (un solo μ, σ) deja a los
clientes chicos en ≈0 y la pérdida la dominan los grandes. Se usa **z-score por serie con estadísticos de
entrenamiento** (μᵢ, σᵢ calculados solo hasta 2014-06-30): todas las series quedan comparables para un modelo global y
no se usa información futura. La normalización **por ventana** (RevIN) se evalúa como iteración en §5.2: adapta el
nivel a cada ventana y ayuda ante cambios de nivel, a cambio de perder el nivel absoluto de la serie. Las métricas
siempre se calculan después de invertir la transformación, en kWh.
""")
code("""
def fit_scaler(Y_train, kind="series"):
    \"\"\"Estadísticos SOLO del bloque de entrenamiento. kind='series': z-score por cliente; 'global': uno para todos.\"\"\"
    if kind == "global":
        return np.full(Y_train.shape[1], Y_train.values.mean()), np.full(Y_train.shape[1], Y_train.values.std())
    return Y_train.mean().values, Y_train.std().values

MU, SD = fit_scaler(Y.loc[:TRAIN_END])
Z = ((Y.values - MU) / SD).astype(np.float32)          # (T, N) panel escalado
print("Z train: media", Z[:t_val].mean().round(3), "std", Z[:t_val].std().round(3),
      "| Z test: media", Z[t_test:].mean().round(3))
""")

md("""
### 2.4 Covariables de calendario (conocidas a futuro)

Hora, día de la semana y mes se codifican con **seno/coseno** (`sin(2πk/P)`, `cos(2πk/P)`): 2 columnas por ciclo en
vez de 24+7+12 one-hot, y la distancia entre 23:00 y 00:00 (o diciembre y enero) es la misma que entre horas
consecutivas. El feriado es binario (1 todo el día). Las 7 columnas dependen solo del reloj y del calendario oficial,
así que se conocen para cualquier hora futura al emitir el pronóstico.
""")
code("""
def calendar(idx):
    hol = np.array([d in pt_hol for d in idx.date], dtype=np.float32)
    feats = {}
    for name, k, P in [("hour", idx.hour, 24), ("dow", idx.dayofweek, 7), ("month", idx.month - 1, 12)]:
        feats[f"{name}_sin"] = np.sin(2 * np.pi * k / P)
        feats[f"{name}_cos"] = np.cos(2 * np.pi * k / P)
    feats["holiday"] = hol
    return pd.DataFrame(feats, index=idx).astype(np.float32)

CAL = calendar(Y.index)
C = CAL.values                                          # (T, 7)
print(CAL.loc["2013-12-25 00:00":"2013-12-25 02:00"].round(3))
""")

md("""
### 2.5 Dataset de ventanas deslizantes

`WindowDataset` indexa pares (serie, inicio del objetivo `t0`). Para cada par:
- `X`: `(L, 1 + 7)` = y escalado en `[t0-L, t0)` + calendario de esas horas
- `y`: `(H,)` = y escalado en `[t0, t0+H)`
- `cal_f`: `(H, 7)` = calendario de las horas objetivo (para el decoder del seq2seq y la MLP)

Los splits se definen por la **fecha del objetivo**: una ventana es de validación si sus H horas objetivo caen en
validación (su entrada puede venir de train, que es pasado). `__getitems__` arma el batch completo con indexado
vectorizado de NumPy, mucho más rápido que llamar `__getitem__` 512 veces.
""")
code("""
def target_starts(split, L=L, H=H, stride=1, months=None):
    \"\"\"Índices t0 cuyo objetivo [t0, t0+H) cae completo dentro del split. months: historia de train a usar.\"\"\"
    lo, hi = {"train": (0, t_val), "val": (t_val, t_test), "test": (t_test, T)}[split]
    if split == "train" and months:
        lo = Y.index.get_loc(VAL_START - pd.DateOffset(months=months))
    first = lo + L if split == "train" else lo        # en train la entrada también debe caer en la historia usada
    return np.arange(first, hi - H + 1, stride)


class WindowDataset(Dataset):
    \"\"\"Ventanas deslizantes sobre todas las series del panel.\"\"\"

    def __init__(self, Z, C, t0s, L=L, H=H, series=None):
        self.Z, self.C, self.t0s, self.L, self.H = Z, C, np.asarray(t0s), L, H
        self.series = np.arange(Z.shape[1]) if series is None else np.asarray(series)

    def __len__(self):
        return len(self.series) * len(self.t0s)

    def __getitems__(self, idx):
        idx = np.asarray(idx)
        n, t0 = self.series[idx // len(self.t0s)], self.t0s[idx % len(self.t0s)]
        past = t0[:, None] + np.arange(-self.L, 0)
        fut = t0[:, None] + np.arange(self.H)
        x = np.concatenate([self.Z[past, n[:, None]][..., None], self.C[past]], axis=-1)
        return (torch.from_numpy(x), torch.from_numpy(self.Z[fut, n[:, None]]),
                torch.from_numpy(self.C[fut]), torch.from_numpy(n))

    def __getitem__(self, i):
        return tuple(t[0] for t in self.__getitems__([i]))


def loader(ds, batch_size=512, shuffle=False, num_samples=None):
    \"\"\"DataLoader que entrega el batch ya armado por __getitems__ (collate identidad).\"\"\"
    sampler = torch.utils.data.RandomSampler(ds, num_samples=num_samples) if shuffle else None
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, collate_fn=lambda b: b)


counts = {s: len(WindowDataset(Z, C, target_starts(s))) for s in ("train", "val", "test")}
print(pd.Series(counts, name=f"ventanas (L={L}, H={H}, stride 1 h, {N_CLIENTS} series)").map("{:,}".format))
print({s: f"{c // N_CLIENTS:,} por serie" for s, c in counts.items()})
x, y, cf, n = next(iter(loader(WindowDataset(Z, C, target_starts("train")), shuffle=True)))
print("X", tuple(x.shape), "| y", tuple(y.shape), "| cal_f", tuple(cf.shape), "| series", tuple(n.shape))
""")

md("""
### 2.6 Auditoría de fugas

Regla: cada entrada de la red debe poder calcularse en el instante en que se emite el pronóstico (00:00 del día
objetivo, con datos hasta las 23:00 del día anterior). Las cuatro fugas vistas en clase y cómo las evita el pipeline:
""")
code("""
times = Y.index
tr, va, te = (target_starts(s) for s in ("train", "val", "test"))

# 1) Split aleatorio de ventanas -> split cronológico por fecha del objetivo, sin solapes de objetivos
assert times[tr.max() + H - 1] <= TRAIN_END < times[va.min()]
assert times[va.max() + H - 1] <= VAL_END < times[te.min()]
rand = np.random.default_rng(0).permutation(np.r_[tr, va])[: len(va)]       # lo que haría un split aleatorio
print("1) split cronológico OK | con split aleatorio, % de 'val' con objetivo en train:",
      f"{(times[rand] <= TRAIN_END).mean():.0%}")

# 2) Escalar con toda la serie -> μ, σ solo de train; con la serie completa cambian (el futuro se filtra)
mu_all, sd_all = Y.mean().values, Y.std().values
assert np.allclose(MU, Y.loc[:TRAIN_END].mean().values)
print(f"2) scaler solo con train OK | |Δμ| medio si se usara toda la serie: {np.mean(np.abs(mu_all / MU - 1)):.1%}")

# 3) Early stopping con prueba -> los loaders de entrenamiento solo se construyen con 'train' y 'val';
#    train_model() (§5) verifica que la última hora objetivo de su loader de validación sea <= VAL_END
assert times[va.max() + H - 1] == VAL_END
print("3) validación termina en", VAL_END, "| prueba solo se toca dentro de backtest()")

# 4) Covariables mal alineadas -> X termina en t0-1 (dato real hasta 23:00); y y cal_f empiezan en t0
ds = WindowDataset(Z, C, te[::24])
x, y, cf, n = ds.__getitems__([0])
t0 = te[0]
assert times[t0] == TEST_START and times[t0 - 1].hour == 23
assert np.allclose(x[0, -1, 0], Z[t0 - 1, n[0]]) and np.allclose(y[0, 0], Z[t0, n[0]])
assert np.allclose(cf[0], calendar(times[t0:t0 + H]).values)          # calendario recalculado a futuro = cal_f
print("4) X:", times[t0 - L], "→", times[t0 - 1], "| y:", times[t0], "→", times[t0 + H - 1])
""")

# ================================================================== 3. Investigación
batch(2)
md("""
## 3. Investigación: herramientas para forecasting en PyTorch

**`nn.LSTM` / `nn.GRU`** procesan una secuencia con un estado recurrente. Parámetros: `input_size` (features por paso),
`hidden_size` (tamaño del estado), `num_layers` (capas apiladas), `batch_first=True` (entrada `(B, L, F)` en vez de
`(L, B, F)`), `dropout` (entre capas, solo si `num_layers > 1`). Salidas de la LSTM: `output` `(B, L, D·hidden)` con el
estado oculto de la última capa en cada paso; `h_n` `(D·num_layers, B, hidden)` el estado oculto final de cada capa;
`c_n` (solo LSTM) la celda de memoria final, misma forma. La GRU devuelve solo `(output, h_n)` (no tiene celda).

**`nn.Conv1d` dilatada y causal** (`in_channels, out_channels, kernel_size, dilation, padding`): la dilatación `d` salta
`d−1` pasos entre taps del kernel. Para que la salida en `t` no vea `t+1…` se rellena `(k−1)·d` solo a la izquierda
(`F.pad`) o se usa `padding=(k−1)·d` y se recortan los últimos pasos. Con capas de dilatación 1, 2, 4, …, 2ⁿ⁻¹ el campo
receptivo es `R = 1 + (k−1)·Σdᵢ = 1 + (k−1)(2ⁿ−1)`; con 2 convoluciones por bloque residual (TCN), `1 + 2(k−1)(2ⁿ−1)`.
Ej.: k = 3 y 7 bloques → R = 509 h, más de 3 semanas.

**`nn.TransformerEncoderLayer`** (`d_model` dimensión de los tokens, `nhead` cabezas de atención, `dim_feedforward`
ancho de la FFN, `dropout`, `batch_first`, `norm_first`): self-attention + FFN con residuales y LayerNorm. La atención es
invariante a permutaciones, así que el orden se inyecta sumando una **codificación posicional** a los embeddings antes
del encoder: `x = Linear(entrada) + PE[:L]`, con PE sinusoidal fija o una `nn.Parameter` aprendida. PatchTST aplica esto a
*parches* de la serie (16 pasos por token) en vez de a cada hora.

**StatsForecast / NeuralForecast (Nixtla)** reciben un DataFrame largo con `unique_id` (serie), `ds` (fecha) y `y`
(+ columnas exógenas). `StatsForecast(models, freq, n_jobs)` ajusta modelos locales (uno por serie);
`NeuralForecast(models, freq)` entrena modelos globales. `cross_validation(df, h, n_windows, step_size, refit, …)` hace
backtesting con origen móvil: `n_windows` = número de orígenes, `step_size` = distancia entre orígenes (24 → uno diario),
`refit` = re-entrenar en cada ventana (`True`), nunca (`False`) o cada k ventanas (entero). En StatsForecast,
`input_size` limita la historia de cada ajuste; en NeuralForecast, `val_size` reserva el bloque de early stopping.
Devuelve `unique_id, ds, cutoff, y, <modelo>`.

**Chronos-Bolt** (`chronos-forecasting`): `BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small")`;
`predict_quantiles(inputs, prediction_length, quantile_levels)` recibe un tensor `(B, contexto)` o una lista de tensores 1-D
(NaN = faltante), y devuelve `quantiles` `(B, H, Q)` y `mean` `(B, H)`. Bolt predice 9 cuantiles fijos (0.1…0.9) en
bloques de 64 pasos; horizontes más largos se generan de forma autorregresiva con la mediana.

**Conceptos.**
- *Local vs global*: un modelo local (ARIMA, ETS) ajusta parámetros por serie; uno global comparte pesos entre todas las
  series (normalizadas), aprende patrones comunes con muchos más datos y sirve para series nuevas, pero puede ignorar
  particularidades de cada cliente.
- *Teacher forcing*: al entrenar un modelo autorregresivo o un decoder seq2seq se le da como entrada el valor **real**
  del paso anterior; en inferencia recibe su propia predicción. Entrena rápido y estable, pero crea *exposure bias*: el
  modelo nunca vio sus propios errores y estos se acumulan con el horizonte.
- *RevIN*: normaliza cada ventana con su propia media y desviación (`x̃ = (x − μ_w)/σ_w`, con afín aprendible), la red
  predice en esa escala y la salida se des-normaliza con los mismos `μ_w, σ_w`. Si la serie cambia de nivel entre train y
  test, la red ve entradas con la misma distribución y el nivel nuevo se restituye al final.
- *Pinball loss*: `L_q(y, ŷ) = max(q(y − ŷ), (q − 1)(y − ŷ))`. Su minimizador es el cuantil q de la distribución
  condicional; promediada sobre varios q aproxima el CRPS. Con q = 0.5 es MAE/2.
- *MASE vs WAPE*: MASE = MAE / MAE del seasonal naive dentro de muestra, por serie: no depende de la escala, < 1 significa
  que se le gana al naive, y cada cliente pesa igual. WAPE = Σ|e| / Σ|y| sobre todo el panel: es un porcentaje de volumen,
  dominado por los clientes grandes (vista de negocio), e indefinido si Σ|y| = 0.
""")

# ================================================================== 4. Protocolo
md("""
## 4. Protocolo de validación y métricas

`backtest(model)` es la única función de evaluación. Todo modelo implementa:
- `fit(train_end, val_end)`: entrena con datos ≤ `train_end`; el bloque `(train_end, val_end]` solo sirve para early
  stopping. Por defecto train ≤ 2014-06-30 y val hasta 2014-09-30: los pesos se entrenan una sola vez.
- `predict(origins, H)`: para cada origen `o` (00:00) pronostica `[o, o+H)` con datos `< o` (hasta las 23:00 previas).

Prueba: 92 orígenes diarios, 2014-10-01 … 2014-12-31. Las iteraciones de §5.2 usan los 92 orígenes de validación.
MASE usa como escala, por cliente, el MAE del seasonal naive de 24 h dentro del bloque de entrenamiento.
""")
code("""
N = N_CLIENTS
TEST_ORIGINS = pd.date_range(TEST_START, END.normalize(), freq="D")
VAL_ORIGINS = pd.date_range(VAL_START, VAL_END.normalize(), freq="D")
assert len(TEST_ORIGINS) == len(VAL_ORIGINS) == 92
SCALE = Y.loc[:TRAIN_END].diff(24).abs().mean()           # escala MASE por cliente (kWh)
QS = (0.1, 0.5, 0.9)

def origins_for(H, origins=TEST_ORIGINS):
    \"\"\"Orígenes cuyo horizonte completo cae dentro de los datos (H=168 -> 86 orígenes de prueba).\"\"\"
    return origins[origins + pd.Timedelta(hours=H - 1) <= END]

def to_long(P, origins, H):
    \"\"\"(n_orig, N, H) -> formato largo unique_id, ds, origin, h, y, y_hat [, q10, q50, q90].\"\"\"
    t_o = Y.index.get_indexer(origins)
    tt = np.broadcast_to(t_o[:, None, None] + np.arange(H), (len(t_o), N, H)).ravel()
    nn_ = np.broadcast_to(np.arange(N)[None, :, None], (len(t_o), N, H)).ravel()
    fc = pd.DataFrame({"unique_id": pd.Categorical(np.array(CLIENTS)[nn_]), "ds": Y.index[tt],
                       "origin": np.repeat(origins, N * H), "h": np.tile(np.arange(1, H + 1), len(t_o) * N),
                       "y": Y.values[tt, nn_].astype(np.float32), "y_hat": P["y_hat"].ravel().astype(np.float32)})
    for q in QS:
        if f"q{int(q * 100)}" in P:
            fc[f"q{int(q * 100)}"] = P[f"q{int(q * 100)}"].ravel().astype(np.float32)
    return fc

def evaluate(fc):
    \"\"\"Métricas en kWh agregadas sobre todos los clientes y orígenes.\"\"\"
    e = fc.y_hat - fc.y
    ae = e.abs()
    sae = ae / fc.unique_id.map(SCALE).astype(float)           # error escalado (MASE puntual)
    nz = fc.y > 0
    xm = fc.ds.between("2014-12-22", "2014-12-28 23:00")
    m = {"MAE": ae.mean(), "RMSE": np.sqrt((e ** 2).mean()), "WAPE": ae.sum() / fc.y.abs().sum(),
         "MAPE": (ae[nz] / fc.y[nz]).mean(), "MASE": sae.mean(), "MASE_xmas": sae[xm].mean() if xm.any() else np.nan}
    if "q10" in fc:
        pl = [np.maximum(q * (fc.y - fc[f"q{int(q * 100)}"]), (q - 1) * (fc.y - fc[f"q{int(q * 100)}"])).mean() for q in QS]
        m["pinball"] = np.mean(pl)
        m["cov80"] = ((fc.y >= fc.q10) & (fc.y <= fc.q90)).mean()
    return m

def per_client(fc):
    e = fc.y_hat - fc.y
    g = pd.DataFrame({"unique_id": fc.unique_id, "ae": e.abs(), "se": e ** 2, "ape": e.abs() / fc.y.where(fc.y > 0)})
    out = g.groupby("unique_id", observed=True).agg(MAE=("ae", "mean"), MSE=("se", "mean"), MAPE=("ape", "mean"))
    out["RMSE"] = np.sqrt(out.pop("MSE"))
    out["MASE"] = out.MAE / SCALE[out.index].values
    return out

def by_step(fc, col="h"):
    \"\"\"MAE (kWh) y MASE por paso del horizonte (col='h') o por hora del día (col='hour').\"\"\"
    ae = (fc.y_hat - fc.y).abs()
    key = fc.ds.dt.hour if col == "hour" else fc[col]
    return pd.DataFrame({"MAE": ae, "MASE": ae / fc.unique_id.map(SCALE).astype(float)}).groupby(key.values).mean()

def peak_mem_mb():
    return torch.mps.driver_allocated_memory() / 2 ** 20 if DEVICE.type == "mps" else np.nan

def backtest(model, origins=TEST_ORIGINS, H=H, train_end=TRAIN_END, val_end=VAL_END, fit=True):
    \"\"\"Protocolo único: entrena una vez y pronostica H horas desde cada origen con datos < origen.\"\"\"
    t = time.perf_counter()
    if fit:
        model.fit(train_end, val_end)
    train_s = time.perf_counter() - t
    t = time.perf_counter()
    P = model.predict(origins, H)
    infer_s = time.perf_counter() - t
    assert P["y_hat"].shape == (len(origins), N, H) and np.isfinite(P["y_hat"]).all()
    return {"fc": to_long(P, origins, H), "train_s": train_s, "infer_s": infer_s,
            "n_params": getattr(model, "n_params", (0, 0)), "history": getattr(model, "history", None),
            "epoch_s": getattr(model, "epoch_s", None), "peak_mem_mb": getattr(model, "peak_mem", np.nan)}

REG = {}                                                         # nombre -> fila de la tabla comparativa
FCS = {}                                                         # nombre -> pronósticos de prueba (H=24)

def register(name, res, family, L_=L, strategy="multi-salida", covariates="no", H_=H, hw="CPU"):
    m = evaluate(res["fc"])
    if H_ != H:                                                  # solo agrega MASE con H=168 a la fila existente
        REG.setdefault(name, {})["MASE_H168"] = m["MASE"]
        return m
    FCS[name] = res["fc"]
    tot, trn = res["n_params"]
    REG[name] = {**REG.get(name, {}), "familia": family, "L": L_, "estrategia": strategy, "covariables": covariates,
                 "params": tot, "entrenables": trn, "train_s": res["train_s"], "infer_s": res["infer_s"],
                 "hardware": hw, **m}
    return m

def show(names=None):
    df = pd.DataFrame(REG).T
    df = df.loc[names] if names else df
    cols = [c for c in ["MAE", "RMSE", "WAPE", "MAPE", "MASE", "MASE_xmas", "MASE_H168", "pinball", "cov80",
                        "train_s", "infer_s"] if c in df]
    return df[cols].astype(float).sort_values("MASE").round(3)
""")

md("""
### 4.1 Baselines (StatsForecast)

Naive, seasonal naive de 24 h y 168 h y MSTL (estacionalidades 24 y 168, tendencia con AutoETS). Son modelos locales: en
cada origen se re-ajustan con la historia de esa serie, por lo que `fit()` no hace nada y todo el costo está en
`predict()`. MSTL usa las últimas 6 semanas (`input_size=1008`) para que 9,200 ajustes sean viables.
""")
code("""
from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive, MSTL

def long_df(Yw):
    return Yw.reset_index(names="ds").melt("ds", var_name="unique_id", value_name="y")

class SFModel:
    \"\"\"Adaptador de un modelo de StatsForecast al protocolo fit/predict.\"\"\"

    def __init__(self, model, input_size=None):
        self.model, self.input_size = model, input_size

    def fit(self, train_end, val_end):
        pass                                                     # modelo local: se ajusta en cada origen

    def predict(self, origins, H):
        df = long_df(Y.loc[:origins[-1] + pd.Timedelta(hours=H - 1)])
        sf = StatsForecast(models=[self.model], freq="h", n_jobs=-1)
        cv = sf.cross_validation(df=df, h=H, n_windows=len(origins), step_size=24, input_size=self.input_size)
        cv = cv.sort_values(["cutoff", "unique_id", "ds"])
        assert (pd.DatetimeIndex(cv.cutoff.unique()) == origins - pd.Timedelta("1h")).all()
        assert list(cv.unique_id.iloc[: N * H: H]) == CLIENTS
        return {"y_hat": cv.iloc[:, -1].values.reshape(len(origins), N, H)}

BASELINES = {"Naive": (Naive(), None), "SNaive24": (SeasonalNaive(24), None),
             "SNaive168": (SeasonalNaive(168), None), "MSTL": (MSTL(season_length=[24, 168]), 24 * 7 * 6)}
for name, (mdl, isz) in BASELINES.items():
    res = cached(f"bt_{name}", lambda: backtest(SFModel(mdl, isz)))
    register(name, res, "Baseline", L_="—" if name != "MSTL" else 1008, strategy="directa (local)")
    res168 = cached(f"bt_{name}_H168", lambda: backtest(SFModel(mdl, isz), origins_for(168), 168))
    register(name, res168, "Baseline", H_=168)
show()
""")

md("""
### 4.2 MAPE y clientes de consumo bajo

El MAPE divide cada error entre el valor real: con consumos cercanos a cero un error pequeño en kWh se vuelve un
porcentaje enorme. El problema no es la media del cliente sino las horas en que su consumo cae casi a cero (cortes,
cierres): esas pocas horas dominan su MAPE.
""")
code("""
fc = FCS["SNaive168"]
ape = (fc.y_hat - fc.y).abs() / fc.y.where(fc.y > 0)
pc = per_client(fc).join(means.rename("media_kWh"))
pc["horas_y<1kWh_%"] = 100 * (fc.y < 1).groupby(fc.unique_id, observed=True).mean()
print(pc.sort_values("MAPE", ascending=False).head(5).round(2))
print("Spearman MAPE vs consumo medio:", pc.MAPE.corr(pc.media_kWh, method="spearman").round(2),
      "| vs % de horas < 1 kWh:", pc.MAPE.corr(pc["horas_y<1kWh_%"], method="spearman").round(2))
print(f"horas con y < 1 kWh: {(fc.y < 1).mean():.2%} | APE máximo: {ape.max():,.0f} (= {100 * ape.max():,.0f} %)"
      f" | MAPE {ape.mean():.3f} -> sin horas < 1 kWh: {ape[fc.y >= 1].mean():.3f}")
""")

md("""
### 4.3 LightGBM global con rezagos

Un solo modelo para los 100 clientes (series en z-score). Features de la fila `t`: rezagos 1–24 y 168 de `y` y
calendario de `t` (hora, día, mes, feriado). Se entrena a un paso y se pronostica de forma **recursiva**: para el paso
k del día, los rezagos 1…k−1 son predicciones propias (nunca datos reales posteriores al origen). Early stopping con la
pérdida a un paso en validación.
""")
code("""
import lightgbm as lgb

LAGS = list(range(1, 25)) + [168]
CALI = np.column_stack([Y.index.hour, Y.index.dayofweek, Y.index.month, CAL.holiday]).astype(np.float32)

class LGBMRecursive:
    def __init__(self, months=None, seed=SEED, **params):
        self.months = months
        self.params = {"objective": "l2", "learning_rate": 0.05, "num_leaves": 127, "min_data_in_leaf": 200,
                       "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "seed": seed,
                       "verbose": -1, "num_threads": 0, **params}

    def _rows(self, t_idx):
        tt, nn_ = np.repeat(t_idx, N), np.tile(np.arange(N), len(t_idx))
        X = np.column_stack([Z[tt - k, nn_] for k in LAGS] + [CALI[tt]])
        return X, Z[tt, nn_]

    def fit(self, train_end, val_end):
        te, ve = Y.index.get_loc(train_end), Y.index.get_loc(val_end)
        start = Y.index.get_loc(train_end + pd.Timedelta("1h") - pd.DateOffset(months=self.months)) if self.months else 0
        Xtr, ytr = self._rows(np.arange(start + 168, te + 1))
        Xva, yva = self._rows(np.arange(te + 1, ve + 1))
        cat = list(range(len(LAGS), len(LAGS) + 3))
        dtr = lgb.Dataset(Xtr, ytr, categorical_feature=cat)
        self.booster = lgb.train(self.params, dtr, num_boost_round=3000,
                                 valid_sets=[lgb.Dataset(Xva, yva, reference=dtr)],
                                 callbacks=[lgb.early_stopping(50, verbose=False)])
        dump = self.booster.dump_model()["tree_info"]
        nodes = sum(2 * t["num_leaves"] - 1 for t in dump)        # nodos (splits + hojas) como "parámetros"
        self.n_params = (nodes, nodes)

    def predict(self, origins, H):
        t_o = Y.index.get_indexer(origins)
        B = np.empty((len(t_o), N, 168 + H), dtype=np.float32)
        B[:, :, :168] = Z[t_o[:, None] + np.arange(-168, 0)].transpose(0, 2, 1)
        for k in range(H):
            lags = np.stack([B[:, :, 168 + k - j] for j in LAGS], -1)                 # (n_o, N, 25)
            cal = np.broadcast_to(CALI[t_o + k][:, None, :], (len(t_o), N, CALI.shape[1]))
            X = np.concatenate([lags, cal], -1).reshape(-1, len(LAGS) + CALI.shape[1])
            B[:, :, 168 + k] = self.booster.predict(X, num_threads=0).reshape(len(t_o), N)
        return {"y_hat": B[:, :, 168:] * SD[None, :, None] + MU[None, :, None]}

def lgbm_runs():
    m = LGBMRecursive()
    r24 = backtest(m)
    r168 = backtest(m, origins_for(168), 168, fit=False)
    r168["train_s"] = r24["train_s"]
    return r24, r168, m

r24, r168, lgbm = cached("bt_LightGBM", lgbm_runs)
register("LightGBM", r24, "Árboles", L_=168, strategy="recursiva", covariates="calendario")
register("LightGBM", r168, "Árboles", H_=168)
print("árboles:", lgbm.booster.best_iteration, "| nodos:", r24["n_params"][0], "| entrenamiento", round(r24["train_s"]), "s")
show()
""")

md("""
**Comprobación de fugas en `predict()`.** Se pronostica desde un origen reemplazando por NaN todo dato ≥ origen: si el
modelo leyera el futuro saldría NaN o cambiaría el pronóstico. La misma prueba se aplica a los modelos propios de §5.
""")
code("""
def future_masked(model, origin=TEST_ORIGINS[10], H=H):
    \"\"\"Pronóstico con el futuro borrado == pronóstico normal  =>  el modelo solo usa datos < origen.\"\"\"
    global Y, Z
    Y0, Z0 = Y, Z
    t = Y0.index.get_loc(origin)
    normal = model.predict(pd.DatetimeIndex([origin]), H)["y_hat"]
    Y, Z = Y0.copy(), Z0.copy()
    Y.iloc[t:], Z[t:] = np.nan, np.nan
    try:
        masked = model.predict(pd.DatetimeIndex([origin]), H)["y_hat"]
    finally:
        Y, Z = Y0, Z0
    assert np.isfinite(masked).all() and np.allclose(normal, masked, rtol=1e-4, atol=1e-3)
    return f"{type(model).__name__}: sin fuga hacia el futuro"

print(future_masked(lgbm))
""")

# ================================================================== 5. Modelos propios
batch(3)
md("""
## 5. Modelos propios en PyTorch

Todas las redes reciben `X` `(B, L, 8)` (y normalizado + 7 columnas de calendario) y `cal_f` `(B, H, 7)` y devuelven
`(B, H)` en la escala normalizada. Pérdida L1 (las métricas objetivo son MAE/MASE). Con `revin=True` la red se envuelve con
RevIN sobre el canal y.

| Modelo | Arquitectura | Estrategia | Covariables |
|---|---|---|---|
| MLP | flatten(X) ‖ flatten(cal_f) → 512 → 512 → H | multi-salida | pasado + futuro |
| DLinear | media móvil (k=25) separa tendencia/estacional; Linear(L→H) para cada una | multi-salida | no (solo y) |
| NLinear | resta el último valor, Linear(L→H), lo suma | multi-salida | no (solo y) |
| LSTM | LSTM(8→h, capas) → h_T ‖ flatten(cal_f) → Linear → H | multi-salida | pasado + futuro |
| LSTM-AR | LSTM → y_{t+1}; en inferencia realimenta ŷ con el calendario conocido | autorregresiva | pasado + futuro |
| Seq2seq | encoder LSTM(pasado) → estado → decoder LSTM(cal_f) → Linear por paso | encoder–decoder | pasado + futuro (decoder) |

La LSTM-AR se entrena con *teacher forcing*: la secuencia de entrada se extiende con los valores reales del horizonte
y en cada posición se predice el siguiente valor. El decoder del seq2seq recibe solo el calendario de las horas objetivo
(no consume y), así que no tiene exposure bias y sus 24 salidas se producen en una sola pasada.
""")
code("""
import copy
import torch.nn.functional as F

class RevIN(nn.Module):
    \"\"\"Reversible instance normalization del canal y de cada ventana (con afín aprendible).\"\"\"

    def __init__(self):
        super().__init__()
        self.w, self.b = nn.Parameter(torch.ones(1)), nn.Parameter(torch.zeros(1))

    def norm(self, x):
        y = x[..., :1]
        self.mu, self.sd = y.mean(1, keepdim=True), y.std(1, keepdim=True) + 1e-4
        return torch.cat([(y - self.mu) / self.sd * self.w + self.b, x[..., 1:]], -1)

    def norm_y(self, y):
        return (y - self.mu[:, :, 0]) / self.sd[:, :, 0] * self.w + self.b

    def denorm(self, out):
        return (out - self.b) / (self.w + 1e-8) * self.sd[:, :, 0] + self.mu[:, :, 0]


class Forecaster(nn.Module):
    \"\"\"Base: forward(x, cal_f) -> (B, H). Las subclases implementan core().\"\"\"

    def __init__(self, revin=False):
        super().__init__()
        self.rev = RevIN() if revin else None

    def forward(self, x, cf):
        out = self.core(self.rev.norm(x) if self.rev else x, cf)
        return self.rev.denorm(out) if self.rev else out

    def train_loss(self, x, y, cf):
        return F.l1_loss(self(x, cf), y)


class MLP(Forecaster):
    def __init__(self, L, H, hidden=512, dropout=0.1, revin=False):
        super().__init__(revin)
        self.net = nn.Sequential(nn.Linear(L * 8 + H * 7, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, H))

    def core(self, x, cf):
        return self.net(torch.cat([x.flatten(1), cf.flatten(1)], 1))


class DLinear(Forecaster):
    \"\"\"Zeng et al. (2023): Linear(tendencia) + Linear(estacional); tendencia = media móvil de k pasos.\"\"\"

    def __init__(self, L, H, kernel=25, revin=False):
        super().__init__(revin)
        self.k, self.trend, self.season = kernel, nn.Linear(L, H), nn.Linear(L, H)

    def core(self, x, cf):
        y = x[..., 0]
        pad = F.pad(y[:, None], ((self.k - 1) // 2, self.k // 2), mode="replicate")
        trend = F.avg_pool1d(pad, self.k, stride=1)[:, 0]
        return self.trend(trend) + self.season(y - trend)


class NLinear(Forecaster):
    \"\"\"Resta el último valor observado, Linear(L->H), y lo vuelve a sumar.\"\"\"

    def __init__(self, L, H, revin=False):
        super().__init__(revin)
        self.lin = nn.Linear(L, H)

    def core(self, x, cf):
        last = x[:, -1:, 0]
        return self.lin(x[..., 0] - last) + last


class LSTMMulti(Forecaster):
    \"\"\"Multi-salida: estado final de la LSTM + calendario futuro -> una capa densa con los H pasos.\"\"\"

    def __init__(self, L, H, hidden=128, layers=2, dropout=0.1, revin=False):
        super().__init__(revin)
        self.lstm = nn.LSTM(8, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0)
        self.head = nn.Linear(hidden + H * 7, H)

    def core(self, x, cf):
        _, (h, _) = self.lstm(x)
        return self.head(torch.cat([h[-1], cf.flatten(1)], 1))


class LSTMAR(Forecaster):
    \"\"\"Autorregresiva: predice y_{t+1}; en inferencia realimenta su propia predicción.\"\"\"

    def __init__(self, L, H, hidden=128, layers=2, dropout=0.1, revin=False):
        super().__init__(revin)
        self.H = H
        self.lstm = nn.LSTM(8, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0)
        self.head = nn.Linear(hidden, 1)

    def core(self, x, cf):
        out, state = self.lstm(x)
        yk = self.head(out[:, -1])                                  # ŷ_{t0}
        preds = [yk]
        for k in range(self.H - 1):                                 # entrada [ŷ_{t0+k}, cal_{t0+k}] -> ŷ_{t0+k+1}
            out, state = self.lstm(torch.cat([yk, cf[:, k]], -1)[:, None], state)
            yk = self.head(out[:, -1])
            preds.append(yk)
        return torch.cat(preds, 1)

    def train_loss(self, x, y, cf):
        \"\"\"Teacher forcing: la entrada se extiende con los y reales del horizonte (desplazados un paso).\"\"\"
        if self.rev:
            x, y = self.rev.norm(x), None if y is None else self.rev.norm_y(y)
        seq = torch.cat([x, torch.cat([y[:, :-1, None], cf[:, :-1]], -1)], 1)
        out, _ = self.lstm(seq)
        return F.l1_loss(self.head(out)[..., 0], torch.cat([x[:, 1:, 0], y], 1))


class Seq2Seq(Forecaster):
    \"\"\"Encoder LSTM sobre el pasado; decoder LSTM alimentado con el calendario de las H horas objetivo.\"\"\"

    def __init__(self, L, H, hidden=128, layers=2, dropout=0.1, revin=False):
        super().__init__(revin)
        d = dropout if layers > 1 else 0
        self.enc = nn.LSTM(8, hidden, layers, batch_first=True, dropout=d)
        self.dec = nn.LSTM(7, hidden, layers, batch_first=True, dropout=d)
        self.head = nn.Linear(hidden, 1)

    def core(self, x, cf):
        _, state = self.enc(x)
        out, _ = self.dec(cf, state)
        return self.head(out)[..., 0]


ARCH = {"MLP": MLP, "DLinear": DLinear, "NLinear": NLinear, "LSTM": LSTMMulti, "LSTM-AR": LSTMAR, "Seq2seq": Seq2Seq}
x, y, cf, n = next(iter(loader(WindowDataset(Z, C, target_starts("train")), batch_size=4)))
for name, cls in ARCH.items():
    assert cls(L, H)(x, cf).shape == (4, H), name
print("formas OK:", list(ARCH))
""")

md("""
### Entrenamiento

`TorchModel` adapta cualquier arquitectura al protocolo `fit/predict`. `fit()` muestrea en cada epoch 200k ventanas al azar
del bloque de entrenamiento (de 2.17M posibles con stride 1; se recorre todo el bloque cada ~11 epochs) y al final de cada
epoch pronostica los 92 orígenes de **validación** (00:00 de cada día) para la pérdida, el MAE y el MASE de validación.
Early stopping con paciencia 4 sobre la pérdida de validación y se restauran los mejores pesos. La prueba nunca se usa aquí.
""")
code("""
def norm_panel(kind):
    \"\"\"Panel normalizado con estadísticos de train. 'series' y 'revin' usan z-score por serie; 'global' uno solo.\"\"\"
    mu, sd = fit_scaler(Y.loc[:TRAIN_END], "global" if kind == "global" else "series")
    return ((Y.values - mu) / sd).astype(np.float32), mu, sd

def count_params(net):
    ps = list(net.parameters())
    return sum(p.numel() for p in ps), sum(p.numel() for p in ps if p.requires_grad)

class TorchModel:
    def __init__(self, arch, L=L, H=H, months=None, norm="series", lr=1e-3, batch=512, epochs=25,
                 samples=200_000, patience=4, seed=SEED, series=None, **net_kw):
        self.arch, self.L, self.H, self.months, self.norm, self.net_kw = arch, L, H, months, norm, net_kw
        self.lr, self.batch, self.epochs, self.samples, self.patience, self.seed = lr, batch, epochs, samples, patience, seed
        self.series = np.arange(N) if series is None else np.asarray(series)

    def build(self):
        torch.manual_seed(self.seed)
        self.net = ARCH[self.arch](self.L, self.H, revin=self.norm == "revin", **self.net_kw).to(DEVICE)
        self.n_params = count_params(self.net)
        return self

    def _inputs(self, Zn, t0s):
        \"\"\"Solo entradas (X, cal_f) para los orígenes t0s: lee y en [t0-L, t0) y nunca y >= t0.\"\"\"
        n, t0 = np.tile(self.series, len(t0s)), np.repeat(t0s, len(self.series))
        past = t0[:, None] + np.arange(-self.L, 0)
        x = np.concatenate([Zn[past, n[:, None]][..., None], C[past]], -1)
        return torch.from_numpy(x), torch.from_numpy(C[t0[:, None] + np.arange(self.H)])

    @torch.no_grad()
    def _forecast(self, Zn, t0s, bs=4096):
        self.net.eval()
        x, cf = self._inputs(Zn, t0s)
        out = torch.cat([self.net(x[i:i + bs].to(DEVICE), cf[i:i + bs].to(DEVICE)).cpu() for i in range(0, len(x), bs)])
        out = out.numpy().reshape(len(t0s), len(self.series), self.H)
        return out * self.sd[self.series][None, :, None] + self.mu[self.series][None, :, None]      # kWh

    def fit(self, train_end, val_end):
        np.random.seed(self.seed)
        Zn, self.mu, self.sd = norm_panel(self.norm)
        te, ve = Y.index.get_loc(train_end) + 1, Y.index.get_loc(val_end) + 1
        start = (Y.index.get_loc(train_end + pd.Timedelta("1h") - pd.DateOffset(months=self.months))
                 if self.months else 0)
        tr = np.arange(start + self.L, te - self.H + 1)              # objetivos completos dentro de train
        va = np.arange(te, ve - self.H + 1, 24)                      # orígenes diarios 00:00 de validación
        assert Y.index[tr[-1] + self.H - 1] <= train_end and Y.index[va[-1] + self.H - 1] <= val_end  # fuga 3
        ds = WindowDataset(Zn, C, tr, self.L, self.H, self.series)
        dl = loader(ds, self.batch, shuffle=True, num_samples=min(self.samples, len(ds)))
        y_va = Y.values[va[:, None] + np.arange(self.H)][:, :, self.series].transpose(0, 2, 1)
        sd_s, sc = self.sd[self.series][None, :, None], SCALE.values[self.series][None, :, None]
        self.build()
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.history, best, wait, self.peak_mem = [], np.inf, 0, 0.0
        for ep in range(1, self.epochs + 1):
            t, tl, nb_ = time.perf_counter(), torch.zeros((), device=DEVICE), 0
            self.net.train()
            for x, y, cf, _ in dl:
                loss = self.net.train_loss(x.to(DEVICE), y.to(DEVICE), cf.to(DEVICE))
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                tl, nb_ = tl + loss.detach(), nb_ + 1
            self.peak_mem = max(self.peak_mem, peak_mem_mb())
            err = self._forecast(Zn, va) - y_va
            vl = np.abs(err / sd_s).mean()                               # L1 en escala normalizada
            self.history.append({"epoch": ep, "train_loss": tl.item() / nb_, "val_loss": vl,
                                 "val_MAE": np.abs(err).mean(), "val_MASE": np.abs(err / sc).mean(),
                                 "epoch_s": time.perf_counter() - t})
            if vl < best - 1e-4:
                best, wait, state = vl, 0, copy.deepcopy(self.net.state_dict())
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.net.load_state_dict(state)
        self.epoch_s = np.mean([h["epoch_s"] for h in self.history])

    def predict(self, origins, H):
        assert H == self.H
        Zn, self.mu, self.sd = norm_panel(self.norm)                     # desde el Y actual
        return {"y_hat": self._forecast(Zn, Y.index.get_indexer(origins))}

ITERS = {}

def iterate(name, arch, **cfg):
    \"\"\"Una iteración de §5.2: train + early stopping/selección en validación (92 orígenes). No toca la prueba.\"\"\"
    def go():
        m = TorchModel(arch, **cfg)
        res = backtest(m, VAL_ORIGINS, m.H)
        res["val"] = evaluate(res.pop("fc"))
        res["state"] = {k: v.cpu() for k, v in m.net.state_dict().items()}
        return res
    res = cached(f"it_{name}", go)
    ITERS[name] = {"arch": arch, "cfg": cfg, **res}
    h = res["history"][-1]
    print(f"{name:16s} epochs {h['epoch']:2d} | val MAE {res['val']['MAE']:7.2f} | val MASE {res['val']['MASE']:.3f} "
          f"| {res['train_s']:5.0f} s | params {res['n_params'][0]:,}")
    return res

def load_iter(name):
    \"\"\"Reconstruye el TorchModel ya entrenado de una iteración (pesos del caché).\"\"\"
    it = ITERS[name]
    m = TorchModel(it["arch"], **it["cfg"]).build()
    m.net.load_state_dict(it["state"])
    _, m.mu, m.sd = norm_panel(m.norm)
    return m

def test_torch(name, it_name, family, strategy, covariates):
    \"\"\"Evalúa una sola vez en prueba la configuración elegida en validación (mismos pesos, sin re-entrenar).\"\"\"
    it = ITERS[it_name]
    def go():
        res = backtest(load_iter(it_name), TEST_ORIGINS, it["cfg"].get("H", H), fit=False)
        res.update({k: it[k] for k in ("train_s", "n_params", "history", "epoch_s", "peak_mem_mb")})
        return res
    res = cached(f"bt_{name}", go)
    register(name, res, family, L_=it["cfg"].get("L", L), strategy=strategy, covariates=covariates, hw=HARDWARE)
    return res

print(future_masked(TorchModel("LSTM-AR").build()))         # la AR realimenta predicciones: el caso más delicado
""")

md("""
### 5.2 Iteraciones de hiperparámetros

Base LSTM: L = 168, H = 24, 30 meses de historia, z-score por serie, hidden 128, 2 capas, dropout 0.1, lr 1e-3, batch 512.
Cada iteración cambia un factor. Para DLinear se repiten los barridos de L y de historia (la curva de historia de §6
necesita DLinear, LSTM, LightGBM y PatchTST).
""")
code("""
LSTM_BASE = dict(L=168, hidden=128, layers=2, dropout=0.1, lr=1e-3, batch=512, norm="series")
LSTM_RUNS = {"LSTM-base": {}, "LSTM-L48": {"L": 48}, "LSTM-L336": {"L": 336},
             "LSTM-3m": {"months": 3}, "LSTM-12m": {"months": 12},
             "LSTM-global": {"norm": "global"}, "LSTM-revin": {"norm": "revin"},
             "LSTM-h256": {"hidden": 256}, "LSTM-1capa": {"layers": 1}, "LSTM-drop0.3": {"dropout": 0.3},
             "LSTM-lr3e-3": {"lr": 3e-3}, "LSTM-b128": {"batch": 128}}
for name, ch in LSTM_RUNS.items():
    iterate(name, "LSTM", **{**LSTM_BASE, **ch})
""")
code("""
LIN_RUNS = {"DLinear-L48": ("DLinear", {"L": 48}), "DLinear-L168": ("DLinear", {"L": 168}),
            "DLinear-L336": ("DLinear", {"L": 336}), "DLinear-L336-3m": ("DLinear", {"L": 336, "months": 3}),
            "DLinear-L336-12m": ("DLinear", {"L": 336, "months": 12}), "DLinear-L336-revin": ("DLinear", {"L": 336, "norm": "revin"}),
            "NLinear-L168": ("NLinear", {"L": 168}), "NLinear-L336": ("NLinear", {"L": 336}),
            "MLP-L168": ("MLP", {"L": 168}), "MLP-L336": ("MLP", {"L": 336}), "MLP-L168-revin": ("MLP", {"L": 168, "norm": "revin"})}
for name, (arch, cfg) in LIN_RUNS.items():
    iterate(name, arch, lr=3e-3 if arch != "MLP" else 1e-3, **cfg)
""")

md("""
### 5.1 Estrategias multi-paso con la LSTM

Las tres estrategias usan los hiperparámetros de la mejor LSTM de validación.
""")
code("""
def best_of(prefix):
    cands = {k: v for k, v in ITERS.items() if k.startswith(prefix)}
    return min(cands, key=lambda k: cands[k]["val"]["MASE"])

best_lstm = best_of("LSTM-")
cfg_best = ITERS[best_lstm]["cfg"]
print("mejor LSTM en validación:", best_lstm, cfg_best)
iterate("LSTMAR-best", "LSTM-AR", **cfg_best)
iterate("Seq2seq-best", "Seq2seq", **cfg_best)
""")
code("""
test_torch("LSTM", best_lstm, "Recurrente", "multi-salida", "calendario")
test_torch("LSTM-AR", "LSTMAR-best", "Recurrente", "autorregresiva", "calendario")
test_torch("Seq2seq", "Seq2seq-best", "Encoder–decoder", "encoder–decoder", "calendario (decoder)")
fig, ax = plt.subplots(figsize=(7, 3.2))
for name in ["LSTM-AR", "LSTM", "Seq2seq"]:
    ax.plot(range(1, H + 1), by_step(FCS[name]).MAE, marker=".", label=f"{name} ({REG[name]['estrategia']})")
ax.set(xlabel="paso del horizonte h", ylabel="MAE (kWh)", title="LSTM: estrategias multi-paso (prueba)")
ax.legend(); plt.tight_layout(); plt.savefig(FIGS / "estrategias.png", dpi=150); plt.show()
print(pd.DataFrame({n: by_step(FCS[n]).MAE for n in ["LSTM-AR", "LSTM", "Seq2seq"]}).iloc[[0, 5, 11, 17, 23]].round(2))
show(["LSTM-AR", "LSTM", "Seq2seq"])
""")

md("### Tabla de iteraciones y curvas de pérdida")
code("""
def iter_table():
    rows = []
    for name, it in ITERS.items():
        c = {"L": L, "months": 30, "norm": "series", **it["cfg"]}
        rows.append({"iteración": name, "arq": it["arch"], "L": c["L"], "H": c.get("H", H), "meses": c["months"] or 30,
                     "norm": c["norm"], "hidden": c.get("hidden", "—"), "capas": c.get("layers", "—"),
                     "dropout": c.get("dropout", "—"), "lr": c.get("lr", 1e-3), "batch": c.get("batch", 512),
                     "epochs": it["history"][-1]["epoch"], "params": it["n_params"][0], "entrenables": it["n_params"][1],
                     "s/epoch": round(it["epoch_s"], 1), "train_s": round(it["train_s"]), "infer_s (92 orig.)": round(it["infer_s"], 2),
                     "mem_pico_MB": round(it["peak_mem_mb"]), "val_MAE": round(it["val"]["MAE"], 2), "val_MASE": round(it["val"]["MASE"], 3)})
    return pd.DataFrame(rows).set_index("iteración")

IT = iter_table()
IT.to_csv(RESULTS / "iteraciones.csv")
IT
""")
code("""
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
for name in ["LSTM-base", "LSTM-L48", "LSTM-revin", "LSTM-lr3e-3"]:
    hist = pd.DataFrame(ITERS[name]["history"])
    l, = ax[0].plot(hist.epoch, hist.train_loss, label=f"{name} train")
    ax[0].plot(hist.epoch, hist.val_loss, ls="--", color=l.get_color(), label=f"{name} val")
    ax[1].plot(hist.epoch, hist.val_MASE, color=l.get_color(), marker=".", label=name)
ax[0].set(xlabel="epoch", ylabel="L1 (escala normalizada)", title="Pérdida de entrenamiento y validación"); ax[0].legend(fontsize=6, ncol=2)
ax[1].set(xlabel="epoch", ylabel="MASE val", title="MASE de validación por epoch"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(FIGS / "curvas.png", dpi=150); plt.show()

final = pd.DataFrame({n: {"train_loss": it["history"][-1]["train_loss"], "val_MASE": it["val"]["MASE"]}
                      for n, it in ITERS.items() if it["arch"] == "LSTM"}).T
print("LSTM: correlación de Spearman entre pérdida final de train y MASE de val:",
      final.train_loss.corr(final.val_MASE, method="spearman").round(2))
print(final.sort_values("train_loss").round(3))
""")

md("""
### Modelos lineales y MLP en prueba

Para cada arquitectura se lleva a prueba su mejor configuración de validación. DLinear y NLinear también se re-entrenan con
H = 168 para el MASE semanal.
""")
code("""
for arch, fam, cov in [("DLinear", "Lineal", "no"), ("NLinear", "Lineal", "no"), ("MLP", "MLP", "calendario")]:
    b = best_of(arch + "-")
    test_torch(arch, b, fam, "multi-salida", cov)
    print(arch, "<-", b)
for arch in ["DLinear", "NLinear"]:
    cfg = {**ITERS[best_of(arch + "-")]["cfg"], "H": 168}
    iterate(f"{arch}-H168", arch, **cfg)
    res = cached(f"bt_{arch}_H168", lambda: backtest(load_iter(f"{arch}-H168"), origins_for(168), 168, fit=False))
    register(arch, res, "Lineal", H_=168)
show()
""")

md("""
### 5.3 Modelo local contra global

El mejor modelo propio en validación se re-entrena por separado para 10 clientes (semilla fija), cada uno solo con sus
propias ventanas, y se compara el MASE de prueba en esos 10 clientes contra el modelo global.
""")
code("""
own = {k: v for k, v in ITERS.items() if not k.endswith("H168")}
best_own = min(own, key=lambda k: own[k]["val"]["MASE"])
print("mejor modelo propio en validación:", best_own, ITERS[best_own]["arch"], ITERS[best_own]["cfg"])
LOCAL10 = np.sort(np.random.default_rng(SEED).choice(N, 10, replace=False))
glob_name = {"LSTM": "LSTM", "DLinear": "DLinear", "NLinear": "NLinear", "MLP": "MLP", "LSTM-AR": "LSTM-AR", "Seq2seq": "Seq2seq"}[ITERS[best_own]["arch"]]

def local_runs():
    out = {}
    for i in LOCAL10:
        m = TorchModel(ITERS[best_own]["arch"], series=[i], **ITERS[best_own]["cfg"])
        m.fit(TRAIN_END, VAL_END)
        P = m.predict(TEST_ORIGINS, H)["y_hat"][:, 0]                 # (92, H)
        y = Y.values[Y.index.get_indexer(TEST_ORIGINS)[:, None] + np.arange(H), i]
        out[CLIENTS[i]] = {"MASE_local": np.abs(P - y).mean() / SCALE.iloc[i], "epochs": m.history[-1]["epoch"]}
    return out

loc = pd.DataFrame(cached("local_vs_global", local_runs)).T
pcg = per_client(FCS[glob_name])
loc["MASE_global"] = pcg.loc[loc.index, "MASE"]
loc["media_kWh"] = means[loc.index].round(1)
loc["mejora_global_%"] = (100 * (1 - loc.MASE_global / loc.MASE_local)).round(1)
print(loc.round(3))
print("MASE medio en los 10 clientes -> local:", loc.MASE_local.mean().round(3), "| global:", loc.MASE_global.mean().round(3))
""")

nb.cells = [c for b, c in cells if b <= int(os.environ.get("UPTO", 99))]
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "notebook.ipynb")
print("notebook.ipynb:", len(nb.cells), "cells")
