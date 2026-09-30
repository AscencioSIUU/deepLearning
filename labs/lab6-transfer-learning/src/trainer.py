"""
trainer.py - Loop de entrenamiento, cálculo de métricas multiclase (Accuracy, Precision, Recall, F1 macro),
temporización por epoch y monitoreo de memoria.

CC3092 - Deep Learning y Sistemas Inteligentes | Laboratorio #6
"""

import time
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from src.profiling import measure_peak_memory


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evalúa el modelo sobre un DataLoader (validación o test) calculando loss

    y métricas macro de clasificación: Accuracy, Precision, Recall, F1-Score.
    """
    model.eval()
    running_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

    total_samples = len(all_targets)
    avg_loss = running_loss / total_samples

    all_preds_arr = np.array(all_preds)
    all_targets_arr = np.array(all_targets)

    acc = accuracy_score(all_targets_arr, all_preds_arr)
    prec = precision_score(all_targets_arr, all_preds_arr, average="macro", zero_division=0)
    rec = recall_score(all_targets_arr, all_preds_arr, average="macro", zero_division=0)
    f1 = f1_score(all_targets_arr, all_preds_arr, average="macro", zero_division=0)

    return {
        "loss": round(float(avg_loss), 4),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "predictions": all_preds_arr,
        "targets": all_targets_arr,
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: Optional[torch.device] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    iteration_name: str = "iteration",
    verbose: bool = True,
) -> Tuple[Dict, nn.Module]:
    """Entrena el modelo durante un número determinado de epochs y registra

    todas las métricas solicitadas por iteración.
    """
    if device is None:
        device = torch.device(
            "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        )

    model = model.to(device)

    history = {
        "iteration_name": iteration_name,
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "epoch_time_s": [],
        "cumulative_time_s": [],
        "peak_mem_mb": 0.0,
        "best_epoch": 0,
        "best_val_f1": 0.0,
        "best_val_acc": 0.0,
    }

    best_state_dict = None
    best_f1 = -1.0
    cumulative_time = 0.0

    if verbose:
        print(f"\n--- Iniciando entrenamiento: {iteration_name} ({epochs} epochs en {device}) ---")

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_running_loss = 0.0
        train_samples = 0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item() * images.size(0)
            train_samples += images.size(0)

        if scheduler is not None:
            scheduler.step()

        epoch_end = time.perf_counter()
        epoch_duration = epoch_end - epoch_start
        cumulative_time += epoch_duration

        train_loss = train_running_loss / train_samples
        val_metrics = evaluate(model, val_loader, criterion, device)

        history["epochs"].append(epoch)
        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        history["val_f1"].append(val_metrics["f1_score"])
        history["epoch_time_s"].append(round(epoch_duration, 2))
        history["cumulative_time_s"].append(round(cumulative_time, 2))

        if val_metrics["f1_score"] > best_f1:
            best_f1 = val_metrics["f1_score"]
            history["best_epoch"] = epoch
            history["best_val_f1"] = val_metrics["f1_score"]
            history["best_val_acc"] = val_metrics["accuracy"]
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose:
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy'] * 100:.2f}% | "
                f"Val F1: {val_metrics['f1_score']:.4f} | "
                f"Time: {epoch_duration:.2f}s"
            )

    history["peak_mem_mb"] = measure_peak_memory(device)
    history["total_time_s"] = round(cumulative_time, 2)
    history["avg_epoch_time_s"] = round(cumulative_time / epochs, 2)

    # Cargar mejor modelo
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return history, model
