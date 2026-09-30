"""
evaluate.py - Evaluación sobre test set, generación de matrices de confusión,
curvas de pérdida y gráficas comparativas finales.

CC3092 - Deep Learning y Sistemas Inteligentes | Laboratorio #6
"""

import os
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from src.dataset import CIFAR10_CLASSES
from src.trainer import evaluate


def evaluate_on_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evalúa el modelo final una única vez sobre el conjunto de test oficial."""
    return evaluate(model, test_loader, criterion, device)


def plot_confusion_matrix(
    targets: np.ndarray,
    predictions: np.ndarray,
    model_name: str,
    save_path: str,
    title_suffix: str = "",
) -> None:
    """Genera y guarda la matriz de confusión normalizada con seaborn heatmap."""
    cm = confusion_matrix(targets, predictions, normalize="true")

    plt.figure(figsize=(9, 7.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=CIFAR10_CLASSES,
        yticklabels=CIFAR10_CLASSES,
        cbar=True,
    )
    plt.title(f"Matriz de Confusión Normalizada - {model_name} {title_suffix}", fontsize=13, pad=12, fontweight="bold")
    plt.xlabel("Clase Predicha", fontsize=11, fontweight="bold")
    plt.ylabel("Clase Verdadera", fontsize=11, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Matriz de confusión guardada en: {save_path}")


def plot_loss_curves_for_model(
    iterations_history: List[Dict],
    model_category: str,
    save_path: str,
) -> None:
    """Grafica las curvas de pérdida (train vs val) de 3 iteraciones de un modelo

    en subplots lado a lado para diagnosticar overfitting o underfitting.
    """
    n_iters = len(iterations_history)
    fig, axes = plt.subplots(1, n_iters, figsize=(n_iters * 4.8, 4.0), sharey=False)
    if n_iters == 1:
        axes = [axes]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for idx, (hist, ax) in enumerate(zip(iterations_history, axes)):
        epochs = hist["epochs"]
        ax.plot(epochs, hist["train_loss"], label="Train Loss", color="#1f77b4", lw=2, marker="o", ms=4)
        ax.plot(epochs, hist["val_loss"], label="Val Loss", color="#d62728", lw=2, linestyle="--", marker="s", ms=4)

        ax.set_title(hist["iteration_name"], fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Cross-Entropy Loss", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", frameon=True)

        best_ep = hist["best_epoch"]
        ax.axvline(best_ep, color="gray", linestyle=":", alpha=0.7, label=f"Best Ep ({best_ep})")

    fig.suptitle(f"Curvas de Pérdida Train/Val: {model_category}", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Curvas de pérdida guardadas en: {save_path}")


def plot_val_accuracy_vs_cumulative_time(
    histories: Dict[str, Dict],
    save_path: str = "docs/val_acc_vs_time.png",
) -> None:
    """Gráfica 1 requerida: Accuracy de validación contra tiempo acumulado de entrenamiento

    para las mejores configuraciones de los 3 modelos en la misma figura.
    """
    plt.figure(figsize=(8, 5.5))
    styles = {
        "CNN desde cero": {"color": "#1f77b4", "marker": "o", "ls": "-"},
        "VGG-16 Feature Extractor": {"color": "#ff7f0e", "marker": "s", "ls": "--"},
        "VGG-16 Fine-Tuning": {"color": "#2ca02c", "marker": "^", "ls": "-."},
    }

    for name, hist in histories.items():
        style = styles.get(name, {"color": "black", "marker": "x", "ls": "-"})
        times = hist["cumulative_time_s"]
        val_accs = [acc * 100 for acc in hist["val_acc"]]
        plt.plot(times, val_accs, label=name, lw=2.2, **style)

    plt.title("Accuracy de Validación vs. Tiempo Acumulado de Entrenamiento", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Tiempo de Entrenamiento Acumulado (segundos)", fontsize=11)
    plt.ylabel("Accuracy de Validación (%)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", frameon=True, fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Gráfica Acc vs Tiempo guardada en: {save_path}")


def plot_test_f1_vs_computational_cost(
    model_data: List[Dict],
    save_path: str = "docs/f1_vs_flops.png",
) -> None:
    """Gráfica 2 requerida: F1-Score de Test contra Costo Computacional (GigaFLOPs de inferencia)."""
    plt.figure(figsize=(8, 5.5))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for idx, item in enumerate(model_data):
        name = item["name"]
        flops_g = item["flops_g"]
        f1 = item["test_f1"]
        params_m = item["total_params"] / 1e6

        plt.scatter(
            flops_g,
            f1,
            s=params_m * 12 + 80,
            color=colors[idx % len(colors)],
            alpha=0.85,
            edgecolors="black",
            linewidth=1.5,
            label=f"{name} ({params_m:.1f}M params)",
        )
        plt.annotate(
            f"{name}\n(F1: {f1:.4f}, {flops_g:.2f} GFLOPs)",
            (flops_g, f1),
            textcoords="offset points",
            xytext=(10, -5),
            fontsize=9.5,
            fontweight="bold",
        )

    plt.title("F1-Score en Test vs. Costo Computacional (GigaFLOPs de Inferencia)", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("GigaFLOPs por Imagen (Forward Pass)", fontsize=11)
    plt.ylabel("Test F1-Score (Macro)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", frameon=True, fontsize=9.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Gráfica F1 vs FLOPs guardada en: {save_path}")
