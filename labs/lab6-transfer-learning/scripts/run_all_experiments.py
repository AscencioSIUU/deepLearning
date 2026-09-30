"""
run_all_experiments.py - Orquestador para ejecutar las 9 iteraciones requeridas (3 por modelo),
generar curvas de pérdida, evaluar en Test, ejecutar el experimento de 10% de datos
y exportar la tabla comparativa final.

CC3092 - Deep Learning y Sistemas Inteligentes | Laboratorio #6
"""

import json
import os
import sys
import time
from typing import Dict, List, Tuple

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

# Añadir directorio base al path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.dataset import get_dataloaders
from src.evaluate import (
    evaluate_on_test_set,
    plot_confusion_matrix,
    plot_loss_curves_for_model,
    plot_test_f1_vs_computational_cost,
    plot_val_accuracy_vs_cumulative_time,
)
from src.models import CustomCNN, VGG16FeatureExtractor, VGG16FineTuner
from src.profiling import (
    count_parameters,
    estimate_total_training_flops,
    get_model_flops_and_macs,
    measure_inference_latency,
)
from src.trainer import train_model


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_cnn_iterations(device: torch.device, epochs: int = 12) -> Tuple[List[Dict], nn.Module, Dict]:
    """Ejecuta las 3 iteraciones de la CNN desde cero (C1, C2, C3)."""
    print("\n==================================================")
    print("=== MODELO 1: CNN DESDE CERO (3 ITERACIONES) ===")
    print("==================================================")

    criterion = nn.CrossEntropyLoss()
    histories = []
    best_overall_f1 = -1.0
    best_model = None
    best_config = None

    # Iteración C1: Baseline (3 bloques, sin augmentation, Adam lr=1e-3, dropout 0.25)
    loaders_c1 = get_dataloaders("cnn", batch_size=128, augment=False)
    c1_kwargs = dict(num_blocks=3, base_channels=32, dropout_rate=0.25)
    m1 = CustomCNN(**c1_kwargs)
    opt1 = torch.optim.Adam(m1.parameters(), lr=1e-3, weight_decay=1e-4)
    h1, m1 = train_model(
        m1, loaders_c1["train"], loaders_c1["val"], criterion, opt1,
        epochs=epochs, device=device, iteration_name="C1: Baseline (No Aug, Adam 1e-3)"
    )
    p_tot, p_trn = count_parameters(m1)
    h1["total_params"] = p_tot
    h1["trainable_params"] = p_trn
    h1["frozen_layers"] = "Ninguna (100% entrenable)"
    h1["model_type"] = "CNN desde cero"
    h1["model_kwargs"] = c1_kwargs
    histories.append(h1)

    if h1["best_val_f1"] > best_overall_f1:
        best_overall_f1 = h1["best_val_f1"]
        best_model = m1
        best_config = h1

    # Iteración C2: Con Data Augmentation + Cosine Annealing (Dropout 0.30)
    loaders_c2 = get_dataloaders("cnn", batch_size=128, augment=True)
    c2_kwargs = dict(num_blocks=3, base_channels=32, dropout_rate=0.30)
    m2 = CustomCNN(**c2_kwargs)
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3, weight_decay=1e-4)
    sch2 = CosineAnnealingLR(opt2, T_max=epochs, eta_min=1e-5)
    h2, m2 = train_model(
        m2, loaders_c2["train"], loaders_c2["val"], criterion, opt2,
        epochs=epochs, device=device, scheduler=sch2, iteration_name="C2: Data Aug + Cosine LR"
    )
    p_tot, p_trn = count_parameters(m2)
    h2["total_params"] = p_tot
    h2["trainable_params"] = p_trn
    h2["frozen_layers"] = "Ninguna (100% entrenable)"
    h2["model_type"] = "CNN desde cero"
    h2["model_kwargs"] = c2_kwargs
    histories.append(h2)

    if h2["best_val_f1"] > best_overall_f1:
        best_overall_f1 = h2["best_val_f1"]
        best_model = m2
        best_config = h2

    # Iteración C3: 4 Bloques convolucionales (mayor capacidad) + regularización
    loaders_c3 = get_dataloaders("cnn", batch_size=128, augment=True)
    c3_kwargs = dict(num_blocks=4, base_channels=32, dropout_rate=0.30, classifier_dropout=0.5)
    m3 = CustomCNN(**c3_kwargs)
    opt3 = torch.optim.AdamW(m3.parameters(), lr=8e-4, weight_decay=3e-4)
    sch3 = CosineAnnealingLR(opt3, T_max=epochs, eta_min=1e-5)
    h3, m3 = train_model(
        m3, loaders_c3["train"], loaders_c3["val"], criterion, opt3,
        epochs=epochs, device=device, scheduler=sch3, iteration_name="C3: 4 Bloques + AdamW Regul."
    )
    p_tot, p_trn = count_parameters(m3)
    h3["total_params"] = p_tot
    h3["trainable_params"] = p_trn
    h3["frozen_layers"] = "Ninguna (100% entrenable)"
    h3["model_type"] = "CNN desde cero"
    h3["model_kwargs"] = c3_kwargs
    histories.append(h3)

    if h3["best_val_f1"] > best_overall_f1:
        best_overall_f1 = h3["best_val_f1"]
        best_model = m3
        best_config = h3

    return histories, best_model, best_config


def run_feature_extractor_iterations(device: torch.device, epochs: int = 8) -> Tuple[List[Dict], nn.Module, Dict]:
    """Ejecuta las 3 iteraciones de VGG-16 Feature Extractor (F1, F2, F3)."""
    print("\n========================================================")
    print("=== MODELO 2: VGG-16 FEATURE EXTRACTOR (3 ITERACIONES) ===")
    print("========================================================")

    criterion = nn.CrossEntropyLoss()
    histories = []
    best_overall_f1 = -1.0
    best_model = None
    best_config = None

    # Iteración F1: Clásico replace_last (congelado 100% features, lr=1e-3, sin augmentation)
    loaders_f1 = get_dataloaders("vgg", target_size=112, batch_size=128, augment=False)
    f1_kwargs = dict(classifier_type="replace_last")
    m1 = VGG16FeatureExtractor(**f1_kwargs)
    opt1 = torch.optim.Adam([p for p in m1.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
    h1, m1 = train_model(
        m1, loaders_f1["train"], loaders_f1["val"], criterion, opt1,
        epochs=epochs, device=device, iteration_name="F1: Replace-Last (No Aug, Adam 1e-3)"
    )
    p_tot, p_trn = count_parameters(m1)
    h1["total_params"] = p_tot
    h1["trainable_params"] = p_trn
    h1["frozen_layers"] = "Bloques 1 al 5 completos (model.features)"
    h1["model_type"] = "VGG-16 Feature Extractor"
    h1["model_kwargs"] = f1_kwargs
    histories.append(h1)

    if h1["best_val_f1"] > best_overall_f1:
        best_overall_f1 = h1["best_val_f1"]
        best_model = m1
        best_config = h1

    # Iteración F2: Cabezal compacto moderno con BatchNorm + Data Augmentation
    loaders_f2 = get_dataloaders("vgg", target_size=112, batch_size=128, augment=True)
    f2_kwargs = dict(classifier_type="compact", dropout_rate=0.5)
    m2 = VGG16FeatureExtractor(**f2_kwargs)
    opt2 = torch.optim.AdamW([p for p in m2.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
    h2, m2 = train_model(
        m2, loaders_f2["train"], loaders_f2["val"], criterion, opt2,
        epochs=epochs, device=device, iteration_name="F2: Compact Head + Data Aug"
    )
    p_tot, p_trn = count_parameters(m2)
    h2["total_params"] = p_tot
    h2["trainable_params"] = p_trn
    h2["frozen_layers"] = "Bloques 1 al 5 completos (model.features)"
    h2["model_type"] = "VGG-16 Feature Extractor"
    h2["model_kwargs"] = f2_kwargs
    histories.append(h2)

    if h2["best_val_f1"] > best_overall_f1:
        best_overall_f1 = h2["best_val_f1"]
        best_model = m2
        best_config = h2

    # Iteración F3: Cabezal compacto + Cosine LR Scheduler (lr=3e-4)
    loaders_f3 = get_dataloaders("vgg", target_size=112, batch_size=128, augment=True)
    f3_kwargs = dict(classifier_type="compact", dropout_rate=0.5)
    m3 = VGG16FeatureExtractor(**f3_kwargs)
    opt3 = torch.optim.AdamW([p for p in m3.parameters() if p.requires_grad], lr=4e-4, weight_decay=1e-4)
    sch3 = CosineAnnealingLR(opt3, T_max=epochs, eta_min=1e-5)
    h3, m3 = train_model(
        m3, loaders_f3["train"], loaders_f3["val"], criterion, opt3,
        epochs=epochs, device=device, scheduler=sch3, iteration_name="F3: Compact + Cosine LR (4e-4)"
    )
    p_tot, p_trn = count_parameters(m3)
    h3["total_params"] = p_tot
    h3["trainable_params"] = p_trn
    h3["frozen_layers"] = "Bloques 1 al 5 completos (model.features)"
    h3["model_type"] = "VGG-16 Feature Extractor"
    h3["model_kwargs"] = f3_kwargs
    histories.append(h3)

    if h3["best_val_f1"] > best_overall_f1:
        best_overall_f1 = h3["best_val_f1"]
        best_model = m3
        best_config = h3

    return histories, best_model, best_config


def run_fine_tuning_iterations(device: torch.device, epochs: int = 8) -> Tuple[List[Dict], nn.Module, Dict]:
    """Ejecuta las 3 iteraciones de VGG-16 Fine-Tuning (T1, T2, T3)

    variando los bloques descongelados y learning rates diferenciales.
    """
    print("\n==================================================")
    print("=== MODELO 3: VGG-16 FINE-TUNING (3 ITERACIONES) ===")
    print("==================================================")

    criterion = nn.CrossEntropyLoss()
    histories = []
    best_overall_f1 = -1.0
    best_model = None
    best_config = None

    # Iteración T1: Descongelar Bloque 5 (capas 24..30), Backbone LR=1e-5, Classifier LR=5e-4
    loaders_t1 = get_dataloaders("vgg", target_size=112, batch_size=128, augment=True)
    t1_kwargs = dict(unfreeze_blocks="block5", classifier_type="compact")
    m1 = VGG16FineTuner(**t1_kwargs)
    param_groups1 = m1.get_param_groups(backbone_lr=1e-5, classifier_lr=5e-4, weight_decay=1e-4)
    opt1 = torch.optim.AdamW(param_groups1)
    sch1 = CosineAnnealingLR(opt1, T_max=epochs, eta_min=1e-6)
    h1, m1 = train_model(
        m1, loaders_t1["train"], loaders_t1["val"], criterion, opt1,
        epochs=epochs, device=device, scheduler=sch1, iteration_name="T1: Unfreeze Block 5 (BB 1e-5 / Clf 5e-4)"
    )
    p_tot, p_trn = count_parameters(m1)
    h1["total_params"] = p_tot
    h1["trainable_params"] = p_trn
    h1["frozen_layers"] = "Bloques 1 al 4 (0..23). Descongelado: Bloque 5"
    h1["model_type"] = "VGG-16 Fine-Tuning"
    h1["model_kwargs"] = t1_kwargs
    h1["opt_kwargs"] = dict(backbone_lr=1e-5, classifier_lr=5e-4, weight_decay=1e-4)
    histories.append(h1)

    if h1["best_val_f1"] > best_overall_f1:
        best_overall_f1 = h1["best_val_f1"]
        best_model = m1
        best_config = h1

    # Iteración T2: Descongelar Bloques 4 y 5 (capas 17..30), Backbone LR=1e-5, Classifier LR=5e-4
    loaders_t2 = get_dataloaders("vgg", target_size=112, batch_size=128, augment=True)
    t2_kwargs = dict(unfreeze_blocks="block4_5", classifier_type="compact")
    m2 = VGG16FineTuner(**t2_kwargs)
    param_groups2 = m2.get_param_groups(backbone_lr=1e-5, classifier_lr=5e-4, weight_decay=1e-4)
    opt2 = torch.optim.AdamW(param_groups2)
    sch2 = CosineAnnealingLR(opt2, T_max=epochs, eta_min=1e-6)
    h2, m2 = train_model(
        m2, loaders_t2["train"], loaders_t2["val"], criterion, opt2,
        epochs=epochs, device=device, scheduler=sch2, iteration_name="T2: Unfreeze Blocks 4+5 (BB 1e-5 / Clf 5e-4)"
    )
    p_tot, p_trn = count_parameters(m2)
    h2["total_params"] = p_tot
    h2["trainable_params"] = p_trn
    h2["frozen_layers"] = "Bloques 1 al 3 (0..16). Descongelados: Bloques 4 y 5"
    h2["model_type"] = "VGG-16 Fine-Tuning"
    h2["model_kwargs"] = t2_kwargs
    h2["opt_kwargs"] = dict(backbone_lr=1e-5, classifier_lr=5e-4, weight_decay=1e-4)
    histories.append(h2)

    if h2["best_val_f1"] > best_overall_f1:
        best_overall_f1 = h2["best_val_f1"]
        best_model = m2
        best_config = h2

    # Iteración T3: Bloque 5 con LR ultraconservador (5e-6) y Weight Decay para evitar catastrophic forgetting
    loaders_t3 = get_dataloaders("vgg", target_size=112, batch_size=128, augment=True)
    t3_kwargs = dict(unfreeze_blocks="block5", classifier_type="compact")
    m3 = VGG16FineTuner(**t3_kwargs)
    param_groups3 = m3.get_param_groups(backbone_lr=5e-6, classifier_lr=2.5e-4, weight_decay=1e-3)
    opt3 = torch.optim.AdamW(param_groups3)
    sch3 = CosineAnnealingLR(opt3, T_max=epochs, eta_min=5e-7)
    h3, m3 = train_model(
        m3, loaders_t3["train"], loaders_t3["val"], criterion, opt3,
        epochs=epochs, device=device, scheduler=sch3, iteration_name="T3: Unfreeze Block 5 (BB 5e-6 / Clf 2.5e-4)"
    )
    p_tot, p_trn = count_parameters(m3)
    h3["total_params"] = p_tot
    h3["trainable_params"] = p_trn
    h3["frozen_layers"] = "Bloques 1 al 4 (0..23). Descongelado: Bloque 5"
    h3["model_type"] = "VGG-16 Fine-Tuning"
    h3["model_kwargs"] = t3_kwargs
    h3["opt_kwargs"] = dict(backbone_lr=5e-6, classifier_lr=2.5e-4, weight_decay=1e-3)
    histories.append(h3)

    if h3["best_val_f1"] > best_overall_f1:
        best_overall_f1 = h3["best_val_f1"]
        best_model = m3
        best_config = h3

    return histories, best_model, best_config


def run_reduced_data_experiments(
    best_cnn_fn,
    best_fe_fn,
    best_ft_fn,
    best_ft_opt_kwargs: Dict,
    device: torch.device,
) -> Dict[str, Dict]:
    """Experimento 4.1: Efecto de la cantidad de datos (10% de imágenes de entrenamiento)."""
    print("\n========================================================")
    print("=== SECCIÓN 4.1: EXPERIMENTO CON 10% DE LOS DATOS ===")
    print("========================================================")
    criterion = nn.CrossEntropyLoss()
    results_reduced = {}

    # 1. CNN con 10% de datos
    loaders_cnn_red = get_dataloaders("cnn", batch_size=128, use_reduced_train=True, augment=True)
    m_cnn = best_cnn_fn()
    opt_cnn = torch.optim.AdamW(m_cnn.parameters(), lr=1e-3, weight_decay=1e-4)
    sch_cnn = CosineAnnealingLR(opt_cnn, T_max=12, eta_min=1e-5)
    h_cnn, m_cnn = train_model(
        m_cnn, loaders_cnn_red["train"], loaders_cnn_red["val"], criterion, opt_cnn,
        epochs=12, device=device, scheduler=sch_cnn, iteration_name="CNN (10% datos)"
    )
    test_cnn_red = evaluate_on_test_set(m_cnn, loaders_cnn_red["test"], criterion, device)
    results_reduced["CNN desde cero"] = {
        "val_acc": h_cnn["best_val_acc"],
        "val_f1": h_cnn["best_val_f1"],
        "test_acc": test_cnn_red["accuracy"],
        "test_f1": test_cnn_red["f1_score"],
        "test_prec": test_cnn_red["precision"],
        "test_rec": test_cnn_red["recall"],
    }

    # 2. VGG-16 Feature Extractor con 10% de datos
    loaders_vgg_red = get_dataloaders("vgg", target_size=112, batch_size=128, use_reduced_train=True, augment=True)
    m_fe = best_fe_fn()
    opt_fe = torch.optim.AdamW([p for p in m_fe.parameters() if p.requires_grad], lr=5e-4, weight_decay=1e-4)
    sch_fe = CosineAnnealingLR(opt_fe, T_max=8, eta_min=1e-5)
    h_fe, m_fe = train_model(
        m_fe, loaders_vgg_red["train"], loaders_vgg_red["val"], criterion, opt_fe,
        epochs=8, device=device, scheduler=sch_fe, iteration_name="VGG-16 FE (10% datos)"
    )
    test_fe_red = evaluate_on_test_set(m_fe, loaders_vgg_red["test"], criterion, device)
    results_reduced["VGG-16 Feature Extractor"] = {
        "val_acc": h_fe["best_val_acc"],
        "val_f1": h_fe["best_val_f1"],
        "test_acc": test_fe_red["accuracy"],
        "test_f1": test_fe_red["f1_score"],
        "test_prec": test_fe_red["precision"],
        "test_rec": test_fe_red["recall"],
    }

    # 3. VGG-16 Fine-Tuning con 10% de datos
    m_ft = best_ft_fn()
    param_groups = m_ft.get_param_groups(**best_ft_opt_kwargs)
    opt_ft = torch.optim.AdamW(param_groups)
    sch_ft = CosineAnnealingLR(opt_ft, T_max=8, eta_min=5e-7)
    h_ft, m_ft = train_model(
        m_ft, loaders_vgg_red["train"], loaders_vgg_red["val"], criterion, opt_ft,
        epochs=8, device=device, scheduler=sch_ft, iteration_name="VGG-16 FT (10% datos)"
    )
    test_ft_red = evaluate_on_test_set(m_ft, loaders_vgg_red["test"], criterion, device)
    results_reduced["VGG-16 Fine-Tuning"] = {
        "val_acc": h_ft["best_val_acc"],
        "val_f1": h_ft["best_val_f1"],
        "test_acc": test_ft_red["accuracy"],
        "test_f1": test_ft_red["f1_score"],
        "test_prec": test_ft_red["precision"],
        "test_rec": test_ft_red["recall"],
    }

    return results_reduced


def main():
    device = get_device()
    print(f"Dispositivo de ejecución seleccionado: {device}")

    docs_dir = os.path.join(project_root, "docs")
    results_dir = os.path.join(project_root, "results")
    os.makedirs(docs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 1. Ejecutar las 9 iteraciones
    cnn_histories, best_cnn_model, best_cnn_cfg = run_cnn_iterations(device, epochs=12)
    fe_histories, best_fe_model, best_fe_cfg = run_feature_extractor_iterations(device, epochs=8)
    ft_histories, best_ft_model, best_ft_cfg = run_fine_tuning_iterations(device, epochs=8)

    # 2. Graficar las curvas de pérdida de las 9 iteraciones (3 figuras de 3 paneles)
    plot_loss_curves_for_model(cnn_histories, "CNN desde cero", os.path.join(docs_dir, "loss_curves_cnn.png"))
    plot_loss_curves_for_model(fe_histories, "VGG-16 Feature Extractor", os.path.join(docs_dir, "loss_curves_feature_extractor.png"))
    plot_loss_curves_for_model(ft_histories, "VGG-16 Fine-Tuning", os.path.join(docs_dir, "loss_curves_fine_tuning.png"))

    # 3. Evaluar los 3 mejores modelos sobre el conjunto de test (10,000 imágenes)
    print("\n========================================================")
    print("=== EVALUACIÓN FINAL DE LOS 3 MODELOS SOBRE TEST SET ===")
    print("========================================================")
    criterion = nn.CrossEntropyLoss()

    test_loader_cnn = get_dataloaders("cnn", batch_size=128)["test"]
    test_loader_vgg = get_dataloaders("vgg", target_size=112, batch_size=128)["test"]

    test_metrics_cnn = evaluate_on_test_set(best_cnn_model, test_loader_cnn, criterion, device)
    test_metrics_fe = evaluate_on_test_set(best_fe_model, test_loader_vgg, criterion, device)
    test_metrics_ft = evaluate_on_test_set(best_ft_model, test_loader_vgg, criterion, device)

    # Matrices de confusión
    plot_confusion_matrix(
        test_metrics_cnn["targets"], test_metrics_cnn["predictions"],
        "CNN desde cero", os.path.join(docs_dir, "confusion_matrix_cnn.png"),
        title_suffix=f"(Test Acc: {test_metrics_cnn['accuracy']*100:.1f}%)"
    )
    plot_confusion_matrix(
        test_metrics_fe["targets"], test_metrics_fe["predictions"],
        "VGG-16 Feature Extractor", os.path.join(docs_dir, "confusion_matrix_feature_extractor.png"),
        title_suffix=f"(Test Acc: {test_metrics_fe['accuracy']*100:.1f}%)"
    )
    plot_confusion_matrix(
        test_metrics_ft["targets"], test_metrics_ft["predictions"],
        "VGG-16 Fine-Tuning", os.path.join(docs_dir, "confusion_matrix_fine_tuning.png"),
        title_suffix=f"(Test Acc: {test_metrics_ft['accuracy']*100:.1f}%)"
    )

    # 4. Experimento con 10% de los datos
    reduced_results = run_reduced_data_experiments(
        best_cnn_fn=lambda: CustomCNN(**best_cnn_cfg["model_kwargs"]),
        best_fe_fn=lambda: VGG16FeatureExtractor(**best_fe_cfg["model_kwargs"]),
        best_ft_fn=lambda: VGG16FineTuner(**best_ft_cfg["model_kwargs"]),
        best_ft_opt_kwargs=best_ft_cfg["opt_kwargs"],
        device=device,
    )

    # 5. Profiling detallado de inferencia y cómputo de entrenamiento
    print("\n========================================================")
    print("=== PERFILADO DE RECURSOS Y CÓMPUTO (SECCIÓN 5) ===")
    print("========================================================")
    cnn_flops = get_model_flops_and_macs(best_cnn_model, input_size=(1, 3, 32, 32))
    vgg_fe_flops = get_model_flops_and_macs(best_fe_model, input_size=(1, 3, 112, 112))
    vgg_ft_flops = get_model_flops_and_macs(best_ft_model, input_size=(1, 3, 112, 112))

    lat_cnn = measure_inference_latency(best_cnn_model, input_size=(1, 3, 32, 32), device=device)
    lat_fe = measure_inference_latency(best_fe_model, input_size=(1, 3, 112, 112), device=device)
    lat_ft = measure_inference_latency(best_ft_model, input_size=(1, 3, 112, 112), device=device)

    train_flops_cnn = estimate_total_training_flops(cnn_flops["flops"], 1.0, "scratch", 45000, len(best_cnn_cfg["epochs"]))
    train_flops_fe = estimate_total_training_flops(vgg_fe_flops["flops"], 0.067, "feature_extractor", 45000, len(best_fe_cfg["epochs"]))
    train_flops_ft = estimate_total_training_flops(vgg_ft_flops["flops"], 0.516, "fine_tuning", 45000, len(best_ft_cfg["epochs"]))

    # Gráfica 1: Accuracy de validación vs tiempo acumulado
    best_histories = {
        "CNN desde cero": best_cnn_cfg,
        "VGG-16 Feature Extractor": best_fe_cfg,
        "VGG-16 Fine-Tuning": best_ft_cfg,
    }
    plot_val_accuracy_vs_cumulative_time(best_histories, os.path.join(docs_dir, "val_acc_vs_time.png"))

    # Gráfica 2: F1 de test vs FLOPs de inferencia
    models_comparison_data = [
        {
            "name": "CNN desde cero",
            "flops_g": cnn_flops["flops_g"],
            "test_f1": test_metrics_cnn["f1_score"],
            "total_params": best_cnn_cfg["total_params"],
        },
        {
            "name": "VGG-16 Feature Extractor",
            "flops_g": vgg_fe_flops["flops_g"],
            "test_f1": test_metrics_fe["f1_score"],
            "total_params": best_fe_cfg["total_params"],
        },
        {
            "name": "VGG-16 Fine-Tuning",
            "flops_g": vgg_ft_flops["flops_g"],
            "test_f1": test_metrics_ft["f1_score"],
            "total_params": best_ft_cfg["total_params"],
        },
    ]
    plot_test_f1_vs_computational_cost(models_comparison_data, os.path.join(docs_dir, "f1_vs_flops.png"))

    # Construir tabla comparativa consolidada
    master_table = [
        {
            "Modelo": "CNN desde cero",
            "Configuración": best_cnn_cfg["iteration_name"],
            "Parámetros Totales": f"{best_cnn_cfg['total_params']:,}",
            "Parámetros Entrenables": f"{best_cnn_cfg['trainable_params']:,}",
            "MACs / Imagen": f"{cnn_flops['macs_m']:.2f} M",
            "FLOPs / Imagen (Fwd)": f"{cnn_flops['flops_g']:.3f} G",
            "Latencia (ms/img)": lat_cnn["avg_latency_ms"],
            "Tiempo / Epoch (s)": best_cnn_cfg["avg_epoch_time_s"],
            "Epochs a Mejor Val": f"{best_cnn_cfg['best_epoch']}/{len(best_cnn_cfg['epochs'])}",
            "Tiempo Total (s)": best_cnn_cfg["total_time_s"],
            "Memoria Pico (MB)": best_cnn_cfg["peak_mem_mb"],
            "FLOPs Estimados Entrenamiento": f"{train_flops_cnn['total_tflops']:.2f} TFLOPs",
            "Test Accuracy": f"{test_metrics_cnn['accuracy']*100:.2f}%",
            "Test Precision": f"{test_metrics_cnn['precision']*100:.2f}%",
            "Test Recall": f"{test_metrics_cnn['recall']*100:.2f}%",
            "Test F1-Score": f"{test_metrics_cnn['f1_score']*100:.2f}%",
            "Test Acc (10% Datos)": f"{reduced_results['CNN desde cero']['test_acc']*100:.2f}%",
            "Test F1 (10% Datos)": f"{reduced_results['CNN desde cero']['test_f1']*100:.2f}%",
            "Caída Acc (10% vs 100%)": f"{(reduced_results['CNN desde cero']['test_acc'] - test_metrics_cnn['accuracy'])*100:.2f}%",
        },
        {
            "Modelo": "VGG-16 Feature Extractor",
            "Configuración": best_fe_cfg["iteration_name"],
            "Parámetros Totales": f"{best_fe_cfg['total_params']:,}",
            "Parámetros Entrenables": f"{best_fe_cfg['trainable_params']:,}",
            "MACs / Imagen": f"{vgg_fe_flops['macs_m']:.2f} M",
            "FLOPs / Imagen (Fwd)": f"{vgg_fe_flops['flops_g']:.3f} G",
            "Latencia (ms/img)": lat_fe["avg_latency_ms"],
            "Tiempo / Epoch (s)": best_fe_cfg["avg_epoch_time_s"],
            "Epochs a Mejor Val": f"{best_fe_cfg['best_epoch']}/{len(best_fe_cfg['epochs'])}",
            "Tiempo Total (s)": best_fe_cfg["total_time_s"],
            "Memoria Pico (MB)": best_fe_cfg["peak_mem_mb"],
            "FLOPs Estimados Entrenamiento": f"{train_flops_fe['total_tflops']:.2f} TFLOPs",
            "Test Accuracy": f"{test_metrics_fe['accuracy']*100:.2f}%",
            "Test Precision": f"{test_metrics_fe['precision']*100:.2f}%",
            "Test Recall": f"{test_metrics_fe['recall']*100:.2f}%",
            "Test F1-Score": f"{test_metrics_fe['f1_score']*100:.2f}%",
            "Test Acc (10% Datos)": f"{reduced_results['VGG-16 Feature Extractor']['test_acc']*100:.2f}%",
            "Test F1 (10% Datos)": f"{reduced_results['VGG-16 Feature Extractor']['test_f1']*100:.2f}%",
            "Caída Acc (10% vs 100%)": f"{(reduced_results['VGG-16 Feature Extractor']['test_acc'] - test_metrics_fe['accuracy'])*100:.2f}%",
        },
        {
            "Modelo": "VGG-16 Fine-Tuning",
            "Configuración": best_ft_cfg["iteration_name"],
            "Parámetros Totales": f"{best_ft_cfg['total_params']:,}",
            "Parámetros Entrenables": f"{best_ft_cfg['trainable_params']:,}",
            "MACs / Imagen": f"{vgg_ft_flops['macs_m']:.2f} M",
            "FLOPs / Imagen (Fwd)": f"{vgg_ft_flops['flops_g']:.3f} G",
            "Latencia (ms/img)": lat_ft["avg_latency_ms"],
            "Tiempo / Epoch (s)": best_ft_cfg["avg_epoch_time_s"],
            "Epochs a Mejor Val": f"{best_ft_cfg['best_epoch']}/{len(best_ft_cfg['epochs'])}",
            "Tiempo Total (s)": best_ft_cfg["total_time_s"],
            "Memoria Pico (MB)": best_ft_cfg["peak_mem_mb"],
            "FLOPs Estimados Entrenamiento": f"{train_flops_ft['total_tflops']:.2f} TFLOPs",
            "Test Accuracy": f"{test_metrics_ft['accuracy']*100:.2f}%",
            "Test Precision": f"{test_metrics_ft['precision']*100:.2f}%",
            "Test Recall": f"{test_metrics_ft['recall']*100:.2f}%",
            "Test F1-Score": f"{test_metrics_ft['f1_score']*100:.2f}%",
            "Test Acc (10% Datos)": f"{reduced_results['VGG-16 Fine-Tuning']['test_acc']*100:.2f}%",
            "Test F1 (10% Datos)": f"{reduced_results['VGG-16 Fine-Tuning']['test_f1']*100:.2f}%",
            "Caída Acc (10% vs 100%)": f"{(reduced_results['VGG-16 Fine-Tuning']['test_acc'] - test_metrics_ft['accuracy'])*100:.2f}%",
        },
    ]

    df_master = pd.DataFrame(master_table)
    csv_path = os.path.join(results_dir, "master_comparison_table.csv")
    df_master.to_csv(csv_path, index=False)
    print(f"\nTabla comparativa maestra guardada en: {csv_path}")

    # Guardar resumen completo en JSON
    all_experiments_data = {
        "hardware": {
            "device": str(device),
            "chip": "Apple M5 Pro",
            "cores": 15,
            "memory_gb": 24,
            "framework": "PyTorch 2.14.0",
        },
        "cnn_iterations": cnn_histories,
        "feature_extractor_iterations": fe_histories,
        "fine_tuning_iterations": ft_histories,
        "test_results": {
            "CNN desde cero": {k: v for k, v in test_metrics_cnn.items() if not isinstance(v, (np.ndarray, list))},
            "VGG-16 Feature Extractor": {k: v for k, v in test_metrics_fe.items() if not isinstance(v, (np.ndarray, list))},
            "VGG-16 Fine-Tuning": {k: v for k, v in test_metrics_ft.items() if not isinstance(v, (np.ndarray, list))},
        },
        "reduced_10pct_results": reduced_results,
        "master_table": master_table,
    }

    json_path = os.path.join(results_dir, "all_experiments_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_experiments_data, f, indent=2)
    print(f"Resumen completo en JSON guardado en: {json_path}")

    print("\n========================================================")
    print("=== ¡TODAS LAS ITERACIONES Y EXPERIMENTOS COMPLETADOS! ===")
    print("========================================================")


if __name__ == "__main__":
    main()
