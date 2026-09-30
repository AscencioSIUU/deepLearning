"""
models.py - Definición de las 3 arquitecturas para CIFAR-10:
1. CustomCNN (desde cero, resolución 32x32, >=3 bloques convolucionales, BatchNorm, Dropout).
2. VGG16FeatureExtractor (pesos de ImageNet, convoluciones congeladas, clasificador adaptado).
3. VGG16FineTuner (pesos de ImageNet, bloques superiores descongelados, differential learning rates).

CC3092 - Deep Learning y Sistemas Inteligentes | Laboratorio #6
"""

import os
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# Configurar directorio de pesos para cargar en local sin peticiones de red
project_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.environ["TORCH_HOME"] = project_data_dir

from torchvision.models import vgg16, VGG16_Weights


def _safe_pool_to_2x2(x: torch.Tensor) -> torch.Tensor:
    """Pooling a 2x2 compatible con MPS (Apple Silicon), CUDA y CPU."""
    h, w = x.shape[-2], x.shape[-1]
    if x.device.type == "mps" and (h % 2 != 0 or w % 2 != 0):
        return F.interpolate(x, size=(2, 2), mode="bilinear", align_corners=False)
    return F.adaptive_avg_pool2d(x, (2, 2))


def _safe_pool_to_7x7(x: torch.Tensor) -> torch.Tensor:
    """Pooling a 7x7 compatible con MPS (Apple Silicon), CUDA y CPU."""
    h, w = x.shape[-2], x.shape[-1]
    if x.device.type == "mps" and (h % 7 != 0 or w % 7 != 0):
        return F.interpolate(x, size=(7, 7), mode="bilinear", align_corners=False)
    return F.adaptive_avg_pool2d(x, (7, 7))


class CustomCNN(nn.Module):
    """Red Convolucional propia para CIFAR-10 sobre resolución nativa (32x32).

    Diseñada con 3 o 4 bloques convolucionales con BatchNorm, ReLU, MaxPool y Dropout,
    seguida de AdaptiveAvgPool y un clasificador fully-connected regularizado.
    """

    def __init__(
        self,
        num_classes: int = 10,
        num_blocks: int = 3,
        base_channels: int = 32,
        dropout_rate: float = 0.25,
        classifier_dropout: float = 0.5,
    ):
        super().__init__()
        self.num_blocks = num_blocks

        layers: List[nn.Module] = []
        in_c = 3
        curr_c = base_channels

        for block_idx in range(num_blocks):
            layers.append(nn.Conv2d(in_c, curr_c, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(curr_c))
            layers.append(nn.ReLU(inplace=True))

            layers.append(nn.Conv2d(curr_c, curr_c, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(curr_c))
            layers.append(nn.ReLU(inplace=True))

            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            layers.append(nn.Dropout2d(p=dropout_rate))

            in_c = curr_c
            curr_c = min(curr_c * 2, 256)

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))

        classifier_in_features = in_c * 2 * 2
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(classifier_in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=classifier_dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


class VGG16FeatureExtractor(nn.Module):
    """VGG-16 como extractor de características (Transfer Learning).

    Carga pesos preentrenados de ImageNet, congela el 100% de las capas convolucionales
    (features) y reemplaza la cabeza clasificadora adaptada a 10 clases de CIFAR-10.
    """

    def __init__(
        self,
        num_classes: int = 10,
        classifier_type: str = "replace_last",
        dropout_rate: float = 0.5,
        target_size: int = 112,
    ):
        super().__init__()
        weights = VGG16_Weights.DEFAULT
        self.base_model = vgg16(weights=weights)
        self.target_size = target_size

        # 1. Congelar completamente el backbone convolucional
        for param in self.base_model.features.parameters():
            param.requires_grad = False

        self.classifier_type = classifier_type

        if classifier_type == "replace_last":
            in_features = self.base_model.classifier[6].in_features
            self.base_model.classifier[6] = nn.Linear(in_features, num_classes)
            for param in self.base_model.classifier.parameters():
                param.requires_grad = True

        elif classifier_type == "compact":
            self.base_model.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512 * 2 * 2, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(512, num_classes),
            )
            for param in self.base_model.classifier.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"classifier_type '{classifier_type}' no soportado.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] != self.target_size or x.shape[-1] != self.target_size:
            x = F.interpolate(x, size=(self.target_size, self.target_size), mode="bilinear", align_corners=False)
        x = self.base_model.features(x)
        if self.classifier_type == "replace_last":
            x = _safe_pool_to_7x7(x)
            x = torch.flatten(x, 1)
        else:
            x = _safe_pool_to_2x2(x)
        x = self.base_model.classifier(x)
        return x


class VGG16FineTuner(nn.Module):
    """VGG-16 con Fine-Tuning de bloques convolucionales superiores.

    Permite descongelar el Bloque 5 o los Bloques 4 y 5, manteniendo congeladas
    las capas inferiores (detectores de bordes y texturas genéricas de bajo nivel).
    """

    def __init__(
        self,
        num_classes: int = 10,
        unfreeze_blocks: str = "block5",
        classifier_type: str = "compact",
        dropout_rate: float = 0.5,
        target_size: int = 112,
    ):
        super().__init__()
        weights = VGG16_Weights.DEFAULT
        self.base_model = vgg16(weights=weights)
        self.unfreeze_blocks = unfreeze_blocks
        self.classifier_type = classifier_type
        self.target_size = target_size

        # 1. Congelar todo el modelo base inicialmente
        for param in self.base_model.parameters():
            param.requires_grad = False

        # 2. Configurar clasificador
        if classifier_type == "replace_last":
            in_features = self.base_model.classifier[6].in_features
            self.base_model.classifier[6] = nn.Linear(in_features, num_classes)
            for param in self.base_model.classifier.parameters():
                param.requires_grad = True
        elif classifier_type == "compact":
            self.base_model.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512 * 2 * 2, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(512, num_classes),
            )
            for param in self.base_model.classifier.parameters():
                param.requires_grad = True

        # 3. Descongelar selectivamente los bloques superiores
        # Bloque 4: capas 17..23 | Bloque 5: capas 24..30
        if unfreeze_blocks == "block5":
            for layer in self.base_model.features[24:]:
                for param in layer.parameters():
                    param.requires_grad = True
        elif unfreeze_blocks in ("block4_5", "blocks4_5"):
            for layer in self.base_model.features[17:]:
                for param in layer.parameters():
                    param.requires_grad = True
        else:
            raise ValueError(f"unfreeze_blocks '{unfreeze_blocks}' no válido. Use 'block5' o 'block4_5'.")

    def get_param_groups(
        self,
        backbone_lr: float = 1e-5,
        classifier_lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> List[Dict]:
        """Genera grupos de parámetros para el optimizador con learning rate diferencial."""
        backbone_params = []
        classifier_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "classifier" in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)

        return [
            {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay, "name": "backbone"},
            {"params": classifier_params, "lr": classifier_lr, "weight_decay": weight_decay, "name": "classifier"},
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] != self.target_size or x.shape[-1] != self.target_size:
            x = F.interpolate(x, size=(self.target_size, self.target_size), mode="bilinear", align_corners=False)
        x = self.base_model.features(x)
        if self.classifier_type == "replace_last":
            x = _safe_pool_to_7x7(x)
            x = torch.flatten(x, 1)
        else:
            x = _safe_pool_to_2x2(x)
        x = self.base_model.classifier(x)
        return x
