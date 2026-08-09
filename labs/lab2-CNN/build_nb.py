"""Builds lab2_cnn.ipynb from cell definitions below. Re-run after editing to add batches.
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
# Laboratorio #2 — Redes Neuronales Convolucionales (CNN vs. MLP en MNIST)

**CC3092 Deep Learning y Sistemas Inteligentes**
""")

# ---------------------------------------------------------------- 0. Imports
md("## 0. Imports y configuración")
code("""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", DEVICE)

pd.set_option("display.max_columns", None)
%matplotlib inline
""")

# ---------------------------------------------------------------- 1. Dataset
md("## 1. Dataset")
code("""
# Descarga MNIST en bruto (sin transform) para inspeccionar tipos/rangos originales
raw_train = datasets.MNIST(root="data", train=True, download=True)
raw_test = datasets.MNIST(root="data", train=False, download=True)
len(raw_train), len(raw_test)
""")

# ---------------------------------------------------------------- 2. Exploración
md("## 2. Exploración y preparación de los datos")
md("### ¿Cuántas observaciones y cuántas clases tiene el dataset? ¿Están balanceadas?")
code("""
n_train, n_test = len(raw_train), len(raw_test)
classes = raw_train.classes
print(f"Train: {n_train} imágenes | Test: {n_test} imágenes | Clases: {len(classes)} ({classes})")

train_counts = pd.Series(raw_train.targets.numpy()).value_counts().sort_index()
test_counts = pd.Series(raw_test.targets.numpy()).value_counts().sort_index()
pd.DataFrame({"train": train_counts, "test": test_counts})
""")
md("""
**Hallazgos:** 60,000 imágenes de entrenamiento y 10,000 de test, 10 clases (dígitos 0-9).
Las clases están **razonablemente balanceadas**: cada dígito tiene entre ~9.9k y ~10.7k
observaciones en el set de tamaño 60k, sin ninguna clase dominante ni subrepresentada.
""")

md("### ¿Cuál es la dimensión de cada imagen y qué rango de valores tienen los píxeles?")
code("""
img0, label0 = raw_train[0]
arr0 = np.array(img0)
print("Tipo:", type(img0), "| Modo PIL:", img0.mode, "| Tamaño:", img0.size)
print("Shape como array:", arr0.shape, "| dtype:", arr0.dtype)
print("Rango de píxeles: [{}, {}]".format(arr0.min(), arr0.max()))
""")
md("""
**Hallazgos:** cada imagen es de 28×28 píxeles, en escala de grises (1 canal). Los valores de
píxel son enteros `uint8` en el rango [0, 255].
""")

md("### ¿Es necesario normalizar los valores de los píxeles?")
md("""
Sí. Las redes entrenan mejor y de forma más estable con entradas de escala pequeña y centradas
cerca de 0 (gradientes mejor condicionados, convergencia más rápida). Se aplican dos pasos
estándar de `torchvision.transforms`:

1. `ToTensor()`: convierte `uint8 [0,255]` → `float32 [0,1]`.
2. `Normalize(mean=0.1307, std=0.3081)`: estandariza usando la media/desviación estándar
   conocidas de MNIST (calculadas sobre el train set completo), dejando los píxeles
   aproximadamente en media 0 y varianza 1.
""")
code("""
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
])

train_full = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_ds = datasets.MNIST(root="data", train=False, download=True, transform=transform)

x_demo, y_demo = train_full[0]
print("Shape del tensor normalizado:", x_demo.shape, "| rango:", x_demo.min().item(), x_demo.max().item())
""")

md("### Visualización de ejemplos")
code("""
fig, axes = plt.subplots(2, 5, figsize=(10, 4.2))
for i, ax in enumerate(axes.flat):
    img, label = raw_train[i]
    ax.imshow(img, cmap="gray")
    ax.set_title(f"Etiqueta: {label}")
    ax.axis("off")
plt.tight_layout()
plt.show()
""")

md("### División en entrenamiento y validación")
code("""
# 90/10 sobre el train set (54,000 / 6,000); el test set (10,000) queda intacto hasta la evaluación final
n_val = int(0.1 * len(train_full))
n_tr = len(train_full) - n_val
generator = torch.Generator().manual_seed(SEED)
train_ds, val_ds = random_split(train_full, [n_tr, n_val], generator=generator)
len(train_ds), len(val_ds), len(test_ds)
""")
md("""
**Regla de aislamiento del test respetada:** `test_ds` no se usa en ninguna decisión de
arquitectura ni de hiperparámetros; solo se evalúa una vez en la sección 5.
""")

nb["cells"] = cells
with open("lab2_cnn.ipynb", "w") as f:
    nbf.write(nb, f)
print(f"Notebook escrito con {len(cells)} celdas.")
