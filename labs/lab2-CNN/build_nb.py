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

# ---------------------------------------------------------------- 3. Investigación de capas
md("## 3. Investigación: capas de PyTorch para la CNN")
md("""
### `nn.Conv2d`
Aplica convolución 2D: desliza `out_channels` kernels de tamaño `kernel_size` sobre la
entrada, cada uno compartiendo sus pesos en toda la imagen (weight sharing). Parámetros
clave: `in_channels`, `out_channels`, `kernel_size`, `stride` (paso del kernel),
`padding` (relleno de borde para controlar el tamaño de salida). A diferencia de
`nn.Linear`, no conecta cada píxel con cada neurona: solo mira una ventana local
(campo receptivo), lo que preserva la estructura espacial 2D de la imagen.
""")
code("""
conv = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
x = torch.randn(4, 1, 28, 28)  # batch de 4 imágenes MNIST
out = conv(x)
print("Entrada:", x.shape, "-> Salida:", out.shape)
print("Parámetros entrenables:", sum(p.numel() for p in conv.parameters()))
""")

md("""
### `nn.MaxPool2d`
Reduce la resolución espacial tomando el valor máximo dentro de cada ventana
(`kernel_size`, con `stride` por defecto igual al kernel). No tiene parámetros
entrenables; resume la activación más fuerte de cada región, aportando invarianza a
pequeñas traslaciones y reduciendo el costo computacional de las capas siguientes.
""")
code("""
pool = nn.MaxPool2d(kernel_size=2)
out_pool = pool(out)
print("Entrada:", out.shape, "-> Salida:", out_pool.shape)
""")

md("""
### `nn.AvgPool2d`
Igual que `MaxPool2d` pero promedia en vez de tomar el máximo dentro de cada ventana.
Suaviza la activación en vez de quedarse con el pico; útil cuando interesa la
intensidad promedio de una región y no solo su activación más fuerte (p. ej. como
pooling final antes de una capa totalmente conectada, en vez de aplanar todo el mapa).
""")
code("""
avgpool = nn.AvgPool2d(kernel_size=2)
out_avg = avgpool(out)
print("Entrada:", out.shape, "-> Salida:", out_avg.shape)
""")

md("""
### `nn.BatchNorm2d`
Normaliza las activaciones de cada canal usando la media/varianza del mini-batch
actual, luego aplica un reescalado y desplazamiento aprendibles (`γ`, `β`). Parámetro
principal: `num_features` (= número de canales de entrada, debe igualar
`out_channels` de la capa conv anterior). Estabiliza y acelera el entrenamiento al
evitar que la distribución de activaciones se desplace demasiado entre capas
(internal covariate shift).
""")
code("""
bn = nn.BatchNorm2d(num_features=8)
out_bn = bn(out)
print("Shape:", out_bn.shape, "| media por canal ~0:", out_bn.mean(dim=(0,2,3)).detach().numpy().round(3))
""")

md("""
### `nn.Flatten`
Convierte un tensor multidimensional (p. ej. `[batch, C, H, W]`) en un vector 2D
`[batch, C*H*W]`, necesario para conectar la salida de las capas convolucionales con
capas `nn.Linear`. Parámetro `start_dim` (por defecto 1) controla desde qué eje se
aplana, preservando la dimensión de batch.
""")
code("""
flatten = nn.Flatten()
out_flat = flatten(out_pool)
print("Entrada:", out_pool.shape, "-> Salida:", out_flat.shape)
""")

md("""
### `nn.CrossEntropyLoss`
Pérdida estándar para clasificación multiclase. Combina internamente `log_softmax` +
`NLLLoss`: espera **logits crudos** (sin softmax aplicado) de forma `[batch, n_clases]`
y las etiquetas como enteros de clase `[batch]` (no one-hot). Parámetro relevante:
`weight` (para ponderar clases desbalanceadas, no necesario aquí dado el balance visto
en Batch 1).
""")
code("""
criterion = nn.CrossEntropyLoss()
logits_demo = torch.randn(4, 10)          # 4 ejemplos, 10 clases (logits crudos)
labels_demo = torch.tensor([0, 3, 7, 1])  # etiquetas como enteros, no one-hot
loss_demo = criterion(logits_demo, labels_demo)
print("Loss:", loss_demo.item())
""")

md("""
### Tensor en PyTorch
Un tensor es un arreglo multidimensional (escalar, vector, matriz o de mayor orden)
que además de guardar datos numéricos rastrea las operaciones aplicadas sobre él
(autograd) para poder calcular gradientes automáticamente, y puede ejecutarse en
CPU, GPU (CUDA) o MPS de forma transparente.

### Campo receptivo (receptive field)
Es la región de la imagen de entrada que influye en el valor de una unidad de
activación dada, en cierta capa. En la primera capa conv es del tamaño del kernel
(p. ej. 3×3); al apilar capas conv/pool, el campo receptivo de las capas profundas
crece y termina cubriendo regiones mucho más grandes de la imagen original —
así la red combina información local en patrones cada vez más globales.

### ¿Por qué la CNN requiere menos parámetros que un MLP equivalente?
Una capa `nn.Linear` conecta **cada** píxel de entrada con **cada** neurona de salida:
para una imagen 28×28 y 128 neuronas ocultas, eso son `28*28*128 ≈ 100k` pesos solo en
la primera capa. Una capa `nn.Conv2d` reutiliza el **mismo** kernel pequeño (p. ej.
3×3) en toda la imagen (weight sharing) y cada salida solo depende de una ventana
local (conectividad local) — un `Conv2d(1, 8, kernel_size=3)` tiene apenas
`8*(1*3*3+1) = 80` parámetros sin importar el tamaño de la imagen. Esa reutilización
también actúa como regularización estructural: la misma detección de bordes/texturas
es válida en cualquier posición de la imagen.
""")

# ---------------------------------------------------------------- 4. Modelos y entrenamiento
md("## 4. Construcción y entrenamiento de las arquitecturas")
md("### Definición de los modelos")
code("""
class MLP(nn.Module):
    def __init__(self, hidden_sizes=(128, 64), activation=nn.ReLU, dropout=0.0):
        super().__init__()
        layers = [nn.Flatten()]
        in_dim = 28 * 28
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), activation()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self, conv_channels=(16, 32), fc_hidden=64, activation=nn.ReLU,
                 dropout=0.0, use_batchnorm=True, pool="max"):
        super().__init__()
        Pool = nn.MaxPool2d if pool == "max" else nn.AvgPool2d
        conv_layers = []
        in_ch = 1
        for out_ch in conv_channels:
            conv_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            if use_batchnorm:
                conv_layers.append(nn.BatchNorm2d(out_ch))
            conv_layers += [activation(), Pool(kernel_size=2)]
            in_ch = out_ch
        self.conv = nn.Sequential(*conv_layers)

        spatial = 28 // (2 ** len(conv_channels))
        flat_dim = in_ch * spatial * spatial
        fc_layers = [nn.Flatten(), nn.Linear(flat_dim, fc_hidden), activation()]
        if dropout > 0:
            fc_layers.append(nn.Dropout(dropout))
        fc_layers.append(nn.Linear(fc_hidden, 10))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        return self.fc(self.conv(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
""")

md("### Datasets y DataLoaders")
code("""
def make_loaders(batch_size):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
""")

md("### Ciclo de entrenamiento genérico")
code("""
def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.set_grad_enabled(is_train):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = criterion(logits, yb)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * xb.size(0)
            all_preds.append(logits.argmax(dim=1).cpu())
            all_labels.append(yb.cpu())
    avg_loss = total_loss / len(loader.dataset)
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    return avg_loss, preds, labels


def train_model(model, train_loader, val_loader, optimizer, epochs):
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": []}
    for epoch in range(epochs):
        train_loss, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, _, _ = run_epoch(model, val_loader, criterion, optimizer=None)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
    return history
""")

md("### Métricas de evaluación")
code("""
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate_metrics(model, loader):
    criterion = nn.CrossEntropyLoss()
    loss, preds, labels = run_epoch(model, loader, criterion, optimizer=None)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {"loss": loss, "accuracy": acc, "precision": precision, "recall": recall, "f1": f1}
""")

md("### Prueba de humo (smoke test)")
code("""
torch.manual_seed(SEED)
smoke_train, smoke_val, _ = make_loaders(batch_size=64)

mlp_smoke = MLP()
cnn_smoke = CNN()
print("MLP params:", count_params(mlp_smoke), "| CNN params:", count_params(cnn_smoke))

for name, m in [("MLP", mlp_smoke), ("CNN", cnn_smoke)]:
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    hist = train_model(m, smoke_train, smoke_val, opt, epochs=1)
    assert hist["train_loss"][0] < np.log(10) * 1.5, f"{name}: loss inicial fuera de rango esperado"
    print(f"{name} smoke test OK -> train_loss={hist['train_loss'][0]:.3f} val_loss={hist['val_loss'][0]:.3f}")
""")

nb["cells"] = cells
with open("lab2_cnn.ipynb", "w") as f:
    nbf.write(nb, f)
print(f"Notebook escrito con {len(cells)} celdas.")
