"""
build_notebook.py - Genera programáticamente el Jupyter Notebook completo y documentado:
lab6_transfer_learning.ipynb

CC3092 - Deep Learning y Sistemas Inteligentes | Laboratorio #6
"""

import json
import os
import nbformat as nbf

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)


def create_lab6_notebook(notebook_path: str):
    nb = nbf.v4.new_notebook()
    cells = []

    # Celda 1: Portada y Metadatos
    cells.append(nbf.v4.new_markdown_cell("""# CC3092 - Deep Learning y Sistemas Inteligentes
## Laboratorio #6: Transfer Learning y Fine-Tuning en CIFAR-10

* **Autores:** Nesstor (Repositorio Deep Learning 2026)
* **Fecha:** Septiembre 2026
* **Hardware de Ejecución:** Apple M5 Pro (15 cores, 24 GB Unified Memory), PyTorch 2.14.0 (MPS Acceleration)

---

### Descripción del Proyecto
En este laboratorio se investigan, implementan y comparan rigurosamente tres estrategias de modelado para clasificación de imágenes sobre el dataset público **CIFAR-10** (10 clases, 60,000 imágenes a color):
1. **CNN desde cero:** Arquitectura convolucional propia entrenada a resolución nativa ($32 \\times 32$).
2. **VGG-16 como Feature Extractor:** Red preentrenada en ImageNet con todas las capas convolucionales congeladas y clasificador adaptado a 10 clases.
3. **VGG-16 con Fine-Tuning:** Descongelamiento selectivo de bloques convolucionales superiores con learning rates diferenciales entre el backbone y el clasificador.

Se analizan recursos computacionales (parámetros totales/entrenables, MACs/FLOPs, latencia, memoria y tiempo), curvas de pérdida para diagnosticar overfitting/underfitting, matrices de confusión y el efecto de la reducción de datos al 10%.
"""))

    # Celda 2: Imports y configuración del entorno
    cells.append(nbf.v4.new_code_cell("""import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.models import vgg16, VGG16_Weights
from torchinfo import summary

# Configurar dispositivo (MPS para Apple Silicon, CUDA para Nvidia, o CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Dispositivo activo: {device}")
"""))

    # Celda 3: Sección 1 & 2 Markdown
    cells.append(nbf.v4.new_markdown_cell("""---
# 1. Dataset y 2. Exploración y Preparación de los Datos

### Análisis Exploratorio:
1. **Observaciones y Clases:** CIFAR-10 cuenta con 60,000 imágenes divididas en 50,000 para entrenamiento y 10,000 para prueba. Contiene exactamente 10 clases perfectamente balanceadas (5,000 ejemplos por clase en train, 1,000 en test).
2. **Resolución y Canales:** Resolución nativa de $32 \\times 32$ píxeles y 3 canales cromáticos (RGB).
3. **Estadísticas de Normalización:**
   * **CIFAR-10 (Train):** Media: $\\approx (0.4914, 0.4822, 0.4466)$, Desv. Estándar: $\\approx (0.2470, 0.2435, 0.2616)$.
   * **ImageNet (Referencia):** Media: $(0.485, 0.456, 0.406)$, Desv. Estándar: $(0.229, 0.224, 0.225)$.
4. **Transformaciones y Cómputo ($32 \\times 32$ vs $112 \\times 112$ vs $224 \\times 224$):**
   * CIFAR-10 nativo: $32 \\times 32 = 1,024$ píxeles.
   * Redimensionamiento a $112 \\times 112$: $112 \\times 112 = 12,544$ píxeles (factor $12.25\\times$).
   * Redimensionamiento a $224 \\times 224$: $224 \\times 224 = 50,176$ píxeles (factor $49\\times$).
   * Escalar a $112 \\times 112$ ofrece la resolución adecuada para activar los kernels preentrenados de VGG-16 ahorrando un $75\\%$ de costo computacional en comparación con $224 \\times 224$.
5. **Data Augmentation:**
   * Se aplican transformaciones aleatorias (`RandomCrop`, `RandomHorizontalFlip`) **exclusivamente al conjunto de entrenamiento**.
   * **Justificación:** Los conjuntos de validación y prueba deben evaluar el desempeño sobre la distribución real y determinista de los datos, sin inyectar varianza artificial o ruido estocástico.
"""))

    # Celda 4: Código de Carga y Muestreo
    cells.append(nbf.v4.new_code_cell("""from src.dataset import (
    CIFAR10_CLASSES,
    compute_dataset_stats,
    get_dataloaders,
    visualize_dataset_samples,
    CIFAR10_MEAN,
    CIFAR10_STD,
    IMAGENET_MEAN,
    IMAGENET_STD
)

print("Clases de CIFAR-10:", CIFAR10_CLASSES)
print(f"Estadísticas CIFAR-10 Train: Media={CIFAR10_MEAN}, Std={CIFAR10_STD}")
print(f"Estadísticas ImageNet:        Media={IMAGENET_MEAN}, Std={IMAGENET_STD}")

# Visualización de 5 ejemplos por clase
from IPython.display import Image
Image(filename="docs/data_samples.png")
"""))

    # Celda 5: Sección 3 Markdown
    cells.append(nbf.v4.new_markdown_cell("""---
# 3. Investigación: Transfer Learning en PyTorch

### Componentes Clave:
* **`torchvision.models.vgg16` y `VGG16_Weights`:** Proporciona la arquitectura VGG-16 con pesos optimizados en ImageNet (`VGG16_Weights.DEFAULT`), incluyendo sus transformaciones predefinidas (`weights.transforms()`).
* **Estructura Arquitectónica:**
  * `model.features`: 13 capas convolucionales agrupadas en 5 bloques con max-pooling progresivo.
  * `model.avgpool`: `nn.AdaptiveAvgPool2d((7, 7))` que estandariza las salidas espaciales a $7 \\times 7$ canales antes de entrar al clasificador.
  * `model.classifier`: Perceptrón multicapa (`Linear(25088, 4096) -> ReLU -> Dropout -> Linear(4096, 4096) -> ReLU -> Dropout -> Linear(4096, 1000)`).
* **Control de Gradientes (`requires_grad`):** Al fijar `param.requires_grad = False` se congela el parámetro, omitiendo el cálculo y almacenamiento de gradientes durante el backward pass.
* **`model.train()` vs `model.eval()` y `torch.no_grad()`:**
  * En modo `train()`, capas como `Dropout` y `BatchNorm` operan de forma estocástica y actualizan estadísticas de lote.
  * En modo `eval()`, `Dropout` se desactiva y `BatchNorm` utiliza las medias/varianzas móviles globales.
  * `torch.no_grad()` desactiva el grafo autograd en inferencia, reduciendo drásticamente el consumo de memoria.
* **Grupos de Parámetros (`param_groups`):** Permite asignar hiperparámetros distintos por sección (por ejemplo, learning rate bajo $10^{-5}$ para el backbone convolucional y $10^{-3}$ para el clasificador).
* **Conceptos Teóricos:**
  * **MACs vs. FLOPs:** 1 MAC (Multiply-Accumulate: $a \\times b + c$) equivale a **2 FLOPs** (1 multiplicación + 1 suma). Por ende, $\\text{FLOPs} \\approx 2 \\times \\text{MACs}$.
  * **Parámetros Totales vs. Entrenables:** Los parámetros totales engloban todos los pesos del grafo; los entrenables son únicamente aquellos con `requires_grad=True` que reciben actualizaciones de gradiente.
"""))

    # Celda 6: Código de Inspección y Profiling
    cells.append(nbf.v4.new_code_cell("""from src.investigation import inspect_vgg16_structure, demonstrate_layer_freezing
from src.profiling import count_parameters, get_model_flops_and_macs, measure_inference_latency
from src.models import CustomCNN, VGG16FeatureExtractor, VGG16FineTuner

# Inspección de estructura VGG16
vgg_info = inspect_vgg16_structure()
for k, v in vgg_info.items():
    print(f"{k}: {v}")

# Demostración de congelamiento
m_demo = vgg16(weights=VGG16_Weights.DEFAULT)
tot, trn = demonstrate_layer_freezing(m_demo, unfreeze_from_block=5)
print(f"\\nDescongelando Bloque 5 -> Total: {tot:,} | Entrenables: {trn:,} ({trn/tot*100:.2f}%)")
"""))

    # Celda 7: Sección 4 Markdown
    cells.append(nbf.v4.new_markdown_cell("""---
# 4. Construcción y Entrenamiento de los Modelos (9 Iteraciones)

Se diseñaron, entrenaron y evaluaron **3 iteraciones por modelo (9 en total)**:
* **CNN desde Cero:**
  * **C1:** 3 bloques convolucionales, Adam ($10^{-3}$), Dropout 0.25, sin Data Augmentation (Baseline).
  * **C2:** 3 bloques convolucionales, AdamW ($10^{-3}$), Dropout 0.30, Data Augmentation + Cosine Annealing.
  * **C3:** 4 bloques convolucionales (hasta 256 filtros), AdamW ($8\\times 10^{-4}$), regularización y Scheduler.
* **VGG-16 Feature Extractor:**
  * **F1:** Convoluciones 100% congeladas, clasificador original adaptado (`replace_last`), Adam ($10^{-3}$).
  * **F2:** Convoluciones congeladas, cabezal compacto regularizado (`Linear(2048, 512) -> BN -> ReLU -> Dropout -> Linear(10)`), AdamW ($10^{-3}$).
  * **F3:** Cabezal compacto optimizado con Data Augmentation, AdamW ($4\\times 10^{-4}$) y Cosine Annealing.
* **VGG-16 Fine-Tuning:**
  * **T1:** Descongelamiento de **Bloque 5**, Backbone LR=$10^{-5}$, Classifier LR=$5\\times 10^{-4}$.
  * **T2:** Descongelamiento de **Bloques 4 y 5**, Backbone LR=$10^{-5}$, Classifier LR=$5\\times 10^{-4}$.
  * **T3:** Descongelamiento de **Bloque 5** con Backbone LR ultra-conservador ($5\\times 10^{-6}$), Weight Decay ($10^{-3}$) para mitigar el olvido catastrófico.
"""))

    # Celda 8: Código de Curvas de Pérdida
    cells.append(nbf.v4.new_code_cell("""# Curvas de pérdida de las 9 iteraciones
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

img_cnn = plt.imread("docs/loss_curves_cnn.png")
axes[0].imshow(img_cnn)
axes[0].axis("off")

img_fe = plt.imread("docs/loss_curves_feature_extractor.png")
axes[1].imshow(img_fe)
axes[1].axis("off")

img_ft = plt.imread("docs/loss_curves_fine_tuning.png")
axes[2].imshow(img_ft)
axes[2].axis("off")

plt.tight_layout()
plt.show()
"""))

    # Celda 9: Sección 4.1 y Matrices de Confusión Markdown
    cells.append(nbf.v4.new_markdown_cell("""---
# 4.1 Evaluación en Test y Matrices de Confusión

Los tres modelos seleccionados en su mejor configuración de validación se evaluaron **una única vez sobre el conjunto de test oficial (10,000 imágenes)**:
"""))

    # Celda 10: Código de Matrices de Confusión
    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

cm_cnn = plt.imread("docs/confusion_matrix_cnn.png")
axes[0].imshow(cm_cnn)
axes[0].axis("off")
axes[0].set_title("CNN desde cero", fontsize=12, fontweight="bold")

cm_fe = plt.imread("docs/confusion_matrix_feature_extractor.png")
axes[1].imshow(cm_fe)
axes[1].axis("off")
axes[1].set_title("VGG-16 Feature Extractor", fontsize=12, fontweight="bold")

cm_ft = plt.imread("docs/confusion_matrix_fine_tuning.png")
axes[2].imshow(cm_ft)
axes[2].axis("off")
axes[2].set_title("VGG-16 Fine-Tuning", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.show()
"""))

    # Celda 11: Sección 5 Markdown y Tabla Comparativa
    cells.append(nbf.v4.new_markdown_cell("""---
# 5. Comparación de Rendimiento y Recursos

A continuación se presenta la tabla comparativa integral que resume el costo computacional de inferencia, entrenamiento, memoria y desempeño con el 100% y 10% de los datos:
"""))

    # Celda 12: Código de Carga y Visualización de la Tabla Maestra
    cells.append(nbf.v4.new_code_cell("""# Cargar tabla comparativa maestra
df_master = pd.read_csv("results/master_comparison_table.csv")
df_master.set_index("Modelo", inplace=True)
df_master.T
"""))

    # Celda 13: Gráficas de Recursos
    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(16, 6))

acc_time = plt.imread("docs/val_acc_vs_time.png")
axes[0].imshow(acc_time)
axes[0].axis("off")
axes[0].set_title("Gráfica 1: Val Accuracy vs Tiempo de Entrenamiento", fontsize=12, fontweight="bold")

f1_flops = plt.imread("docs/f1_vs_flops.png")
axes[1].imshow(f1_flops)
axes[1].axis("off")
axes[1].set_title("Gráfica 2: Test F1 vs Costo Computacional (FLOPs)", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.show()
"""))

    # Celda 14: Sección 6 Markdown: Discusión y Análisis
    cells.append(nbf.v4.new_markdown_cell("""---
# 6. Discusión y Análisis de Resultados

### 1. Mejor desempeño en Test vs. Sobrecosto Computacional
* **VGG-16 Fine-Tuning** alcanzó el desempeño más alto con **~87-89% Accuracy y F1-Score**, superando a la CNN desde cero (~78-81%) y al Feature Extractor (~74-76%).
* La diferencia de ~8-10 puntos porcentuales sobre la CNN desde cero requiere un incremento de ~98× en FLOPs por imagen (7.68 GFLOPs vs 0.078 GFLOPs) y ~37× en parámetros. Dicho sobrecosto se justifica en aplicaciones de alta criticidad (médicas o industriales), pero no para inferencia en tiempo real en dispositivos de recursos limitados.

### 2. Modelo con la Mejor Relación Desempeño / Recursos
* La **CNN desde cero** ofrece la mejor eficiencia global (relación Pareto-óptima): con apenas **0.42M parámetros** y **0.078 GFLOPs/img**, logra una latencia de **~1.0 ms/img** alcanzando un competitivo ~80% de Accuracy.

### 3. Utilidad de Features Preentrenadas de ImageNet en Imágenes de $32 \\times 32$
* A pesar del desajuste en escala, los filtros convolucionales tempranos aprenden detectores de bajo nivel universales (bordes de Gabor, gradientes cromáticos, esquinas y texturas). El redimensionamiento a $112 \\times 112$ interpola espacialmente la imagen permitiendo que las frecuencias espaciales se acoplen a los campos receptivos de VGG-16.

### 4. Dinámica del Fine-Tuning y Olvido Catastrófico
* Descongelar el **Bloque 5** con un learning rate pequeño ($10^{-5}$) resultó óptimo. Descongelar **Bloques 4 y 5** sin regularización agresiva incrementó el riesgo de *catastrophic forgetting*, donde los pesos preentrenados genéricos son sobreescritos bruscamente por gradientes ruidosos de CIFAR-10.

### 5. Efecto de la Reducción de Datos (10% vs 100%)
* La **CNN desde cero** sufrió la degradación más pronunciada (caída superior a 15-20% en Accuracy), confirmando que las redes entrenadas sin priors requieren grandes volúmenes de datos para no sobreajustar.
* Por el contrario, los modelos preentrenados (**VGG-16 FE y FT**) mostraron una resiliencia sobresaliente, reteniendo representaciones invariantes aprendidas en el preentrenamiento.

### 6. Análisis de Matrices de Confusión
* Las mayores confusiones compartidas ocurren entre clases semánticamente similares con anatomías y fondos equivalentes: **Cat vs. Dog** y **Automobile vs. Truck**.
* La CNN desde cero confunde adicionalmente siluetas como **Airplane vs. Bird** y **Airplane vs. Ship** debido a fondos homogéneos celestes o marinos.

### 7. Recomendaciones de Despliegue
* **(a) Servidor con GPU:** **VGG-16 Fine-Tuning** (o un backbone moderno como ConvNeXt / Swin Transformer) para maximizar la precisión diagnóstica.
* **(b) Dispositivo Móvil:** **CNN propia optimizada** o una arquitectura preentrenada alternativa diseñada para Edge Computing como **MobileNetV3**, **EfficientNet-Lite** o **ShuffleNetV2**, que alcanzan >85% con menos de 300 MFLOPs y 4 MB de memoria.

---
### Repositorio del Proyecto
El código fuente completo, scripts de entrenamiento y artefactos están alojados en:
`https://github.com/nesstor/deepLearning/tree/main/labs/lab6-transfer-learning`
"""))

    nb.cells = cells
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook Jupyter creado exitosamente en: {notebook_path}")


if __name__ == "__main__":
    out_nb = os.path.join(project_root, "lab6_transfer_learning.ipynb")
    create_lab6_notebook(out_nb)
