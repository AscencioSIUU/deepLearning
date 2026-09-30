"""
investigation.py - Demostración y utilidades de la investigación teórica de Transfer Learning en PyTorch.
CC3092 - Deep Learning y Sistemas Inteligentes | Laboratorio #6
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights
from torchinfo import summary


def inspect_vgg16_structure() -> Dict[str, str]:
    """Inspecciona y desglosa los componentes de VGG-16:

    - features (bloques convolucionales 1 a 5)
    - avgpool (AdaptiveAvgPool2d a 7x7)
    - classifier (perceptrón multicapa fully connected)
    """
    weights = VGG16_Weights.DEFAULT
    model = vgg16(weights=weights)

    structure_info = {
        "features_layers": len(model.features),
        "avgpool_target": str(model.avgpool),
        "classifier_layers": len(model.classifier),
        "default_transforms": str(weights.transforms()),
    }
    return structure_info


def demonstrate_layer_freezing(model: nn.Module, unfreeze_from_block: int = 5) -> Tuple[int, int]:
    """Demuestra cómo congelar capas convolucionales y descongelar únicamente

    los bloques superiores usando `requires_grad`.

    VGG-16 `features` contiene 31 capas (incluyendo ReLU y MaxPool2d):
    - Bloque 1: capas 0..4   (Conv 64, Conv 64, MaxPool)
    - Bloque 2: capas 5..9   (Conv 128, Conv 128, MaxPool)
    - Bloque 3: capas 10..16 (Conv 256, Conv 256, Conv 256, MaxPool)
    - Bloque 4: capas 17..23 (Conv 512, Conv 512, Conv 512, MaxPool)
    - Bloque 5: capas 24..30 (Conv 512, Conv 512, Conv 512, MaxPool)
    """
    # 1. Congelar todo el modelo base inicialmente
    for param in model.parameters():
        param.requires_grad = False

    # 2. Descongelar clasificador
    for param in model.classifier.parameters():
        param.requires_grad = True

    # 3. Descongelar bloques superiores si se especifica
    if unfreeze_from_block == 5:
        # Descongelar bloque 5 (capas 24 en adelante)
        for layer in model.features[24:]:
            for param in layer.parameters():
                param.requires_grad = True
    elif unfreeze_from_block == 4:
        # Descongelar bloques 4 y 5 (capas 17 en adelante)
        for layer in model.features[17:]:
            for param in layer.parameters():
                param.requires_grad = True
    elif unfreeze_from_block is None or unfreeze_from_block > 5:
        # Feature extraction puro: features congeladas al 100%
        pass

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params


def get_differential_param_groups(
    model: nn.Module,
    backbone_lr: float = 1e-5,
    classifier_lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> List[Dict]:
    """Configura grupos de parámetros (param_groups) para optimizadores en PyTorch,

    permitiendo aplicar learning rates diferenciados entre el backbone convolucional
    y la cabeza clasificadora (fine-tuning).
    """
    backbone_params = []
    classifier_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in name:
            classifier_params.append(param)
        else:
            backbone_params.append(param)

    groups = []
    if backbone_params:
        groups.append({
            "params": backbone_params,
            "lr": backbone_lr,
            "weight_decay": weight_decay,
            "name": "backbone",
        })
    if classifier_params:
        groups.append({
            "params": classifier_params,
            "lr": classifier_lr,
            "weight_decay": weight_decay,
            "name": "classifier",
        })
    return groups


def get_model_summary(model: nn.Module, input_size: Tuple[int, int, int, int] = (1, 3, 112, 112)) -> str:
    """Utiliza `torchinfo` para generar un reporte detallado de parámetros,

    tamaño de activaciones en memoria y costo computacional (MACs/FLOPs).
    """
    model_stats = summary(
        model,
        input_size=input_size,
        col_names=["input_size", "output_size", "num_params", "trainable", "mult_adds"],
        verbose=0,
    )
    return str(model_stats)


def macs_to_flops(macs: int) -> int:
    """Convierte Multiply-Accumulate operations (MACs) a FLOPs.

    Por convención en deep learning, 1 MAC = 2 FLOPs (1 multiplicación + 1 suma).
    """
    return macs * 2


def track_device_memory(device: torch.device) -> Dict[str, float]:
    """Mide la memoria asignada actual y pico en GPU (CUDA o Apple Silicon MPS) en Megabytes (MB)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        return {"current_mb": allocated, "peak_mb": max_allocated}
    elif device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        torch.mps.synchronize()
        allocated = torch.mps.current_allocated_memory() / (1024 ** 2)
        driver_allocated = (
            torch.mps.driver_allocated_memory() / (1024 ** 2)
            if hasattr(torch.mps, "driver_allocated_memory")
            else allocated
        )
        return {"current_mb": allocated, "peak_mb": driver_allocated}
    else:
        return {"current_mb": 0.0, "peak_mb": 0.0}
