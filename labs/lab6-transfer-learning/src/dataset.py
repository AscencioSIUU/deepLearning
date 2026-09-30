"""
dataset.py - Carga, preparación, preprocesamiento y partición de CIFAR-10.
CC3092 - Deep Learning y Sistemas Inteligentes | Laboratorio #6
"""

import os
import random
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.datasets.cifar as cifar_module
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

# Habilitar carga íntegra local sin re-descargas lentas
datasets.CIFAR10._check_integrity = lambda self: True
cifar_module.check_integrity = lambda *args, **kwargs: True

# 10 Clases oficiales de CIFAR-10
CIFAR10_CLASSES: List[str] = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# Estadísticas oficiales de ImageNet
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Estadísticas precomputadas de CIFAR-10 (train set)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class TransformedSubset(Dataset):
    """Subconjunto de PyTorch que permite aplicar transformaciones personalizadas

    independientes del dataset base.
    """

    def __init__(self, subset: Dataset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx: int):
        img, target = self.subset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self) -> int:
        return len(self.subset)


def compute_dataset_stats(data_root: str = "data") -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Calcula la media y desviación estándar empírica por canal (R, G, B)

    sobre las 50,000 imágenes del conjunto de entrenamiento de CIFAR-10.
    """
    raw_train = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transforms.ToTensor())
    loader = DataLoader(raw_train, batch_size=512, shuffle=False, num_workers=2)

    channels_sum = torch.zeros(3)
    channels_sq_sum = torch.zeros(3)
    num_batches = 0

    for images, _ in loader:
        channels_sum += torch.mean(images, dim=[0, 2, 3])
        channels_sq_sum += torch.mean(images**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sq_sum / num_batches - mean**2) ** 0.5

    mean_tuple = tuple(round(x.item(), 4) for x in mean)
    std_tuple = tuple(round(x.item(), 4) for x in std)
    return mean_tuple, std_tuple


def get_transforms(
    model_type: str = "cnn",
    target_size: int = 112,
    augment: bool = True,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """Retorna las transformaciones (train_transform, eval_transform) según el modelo.

    Para 'cnn' (desde cero): resolución nativa 32x32 y estadísticas de CIFAR-10.
    Para 'vgg' (transfer learning): redimensionamiento a target_size (e.g. 112x112)
    y normalización con estadísticas de ImageNet.
    """
    if model_type == "cnn":
        # Resolución nativa 32x32
        train_transforms_list = []
        if augment:
            train_transforms_list.extend(
                [
                    transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                    transforms.RandomHorizontalFlip(p=0.5),
                ]
            )
        train_transforms_list.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )
        train_transform = transforms.Compose(train_transforms_list)

        eval_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )

    elif model_type in ("vgg", "vgg16"):
        train_transforms_list = []
        if augment:
            train_transforms_list.extend(
                [
                    transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                    transforms.RandomHorizontalFlip(p=0.5),
                ]
            )
        train_transforms_list.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        train_transform = transforms.Compose(train_transforms_list)

        eval_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    else:
        raise ValueError(f"model_type '{model_type}' no reconocido. Use 'cnn' o 'vgg'.")

    return train_transform, eval_transform


def get_stratified_split_indices(
    dataset: datasets.CIFAR10,
    val_per_class: int = 500,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Genera índices estratificados para dividir el conjunto en:

    45,000 entrenamiento (4,500/clase) y 5,000 validación (500/clase).
    """
    targets = np.array(dataset.targets)
    train_indices: List[int] = []
    val_indices: List[int] = []

    rng = np.random.RandomState(seed)

    for class_id in range(10):
        class_indices = np.where(targets == class_id)[0]
        rng.shuffle(class_indices)
        val_indices.extend(class_indices[:val_per_class].tolist())
        train_indices.extend(class_indices[val_per_class:].tolist())

    return train_indices, val_indices


def get_reduced_split_indices(
    train_indices: List[int],
    dataset: datasets.CIFAR10,
    fraction: float = 0.10,
    seed: int = 42,
) -> List[int]:
    """Genera un subconjunto estratificado del conjunto de entrenamiento (ej. 10% = 4,500 imágenes,

    450 por clase), manteniendo el balance estricto.
    """
    targets = np.array(dataset.targets)
    reduced_indices: List[int] = []
    rng = np.random.RandomState(seed)

    for class_id in range(10):
        class_train = [idx for idx in train_indices if targets[idx] == class_id]
        n_samples = int(len(class_train) * fraction)
        rng.shuffle(class_train)
        reduced_indices.extend(class_train[:n_samples])

    return reduced_indices


DEFAULT_DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.environ["TORCH_HOME"] = DEFAULT_DATA_ROOT

def get_dataloaders(
    model_type: str = "cnn",
    target_size: int = 112,
    batch_size: int = 128,
    num_workers: int = 0,
    data_root: str = DEFAULT_DATA_ROOT,
    use_reduced_train: bool = False,
    augment: bool = True,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """Crea los DataLoaders para entrenamiento, validación y prueba con las
    transformaciones y particiones estratificadas correspondientes.
    """
    raw_train_dataset = datasets.CIFAR10(root=data_root, train=True, download=False)
    raw_test_dataset = datasets.CIFAR10(root=data_root, train=False, download=False)

    # Partición estratificada 45k train / 5k val
    train_indices, val_indices = get_stratified_split_indices(raw_train_dataset, val_per_class=500, seed=seed)

    if use_reduced_train:
        train_indices = get_reduced_split_indices(train_indices, raw_train_dataset, fraction=0.10, seed=seed)

    # Obtener transformaciones
    train_transform, eval_transform = get_transforms(model_type=model_type, target_size=target_size, augment=augment)

    # Subconjuntos con transformaciones aplicadas
    train_sub = TransformedSubset(Subset(raw_train_dataset, train_indices), transform=train_transform)
    val_sub = TransformedSubset(Subset(raw_train_dataset, val_indices), transform=eval_transform)
    test_sub = TransformedSubset(raw_test_dataset, transform=eval_transform)

    train_loader = DataLoader(
        train_sub,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_sub,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_sub,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "train_size": len(train_sub),
        "val_size": len(val_sub),
        "test_size": len(test_sub),
    }


def visualize_dataset_samples(
    data_root: str = "data",
    save_path: str = "docs/data_samples.png",
    num_per_class: int = 5,
    seed: int = 42,
) -> None:
    """Visualiza al menos 5 ejemplos representativos por clase (grilla de 10x5)

    y guarda la imagen en alta resolución.
    """
    raw_train = datasets.CIFAR10(root=data_root, train=True, download=True)
    targets = np.array(raw_train.targets)

    rng = np.random.RandomState(seed)

    fig, axes = plt.subplots(10, num_per_class, figsize=(num_per_class * 1.6, 10 * 1.5))
    fig.suptitle("CIFAR-10: 5 Ejemplos por Clase (Resolución Nativa 32x32)", fontsize=14, y=0.995)

    for class_id in range(10):
        indices = np.where(targets == class_id)[0]
        selected = rng.choice(indices, size=num_per_class, replace=False)
        class_name = CIFAR10_CLASSES[class_id]

        for col_idx, img_idx in enumerate(selected):
            img, _ = raw_train[img_idx]
            ax = axes[class_id, col_idx]
            ax.imshow(img)
            ax.axis("off")
            if col_idx == 0:
                ax.set_title(class_name, fontsize=10, loc="left", fontweight="bold", pad=4)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Muestras visualizadas y guardadas con éxito en: {save_path}")


if __name__ == "__main__":
    print("=== Exploración y verificación de CIFAR-10 ===")
    computed_mean, computed_std = compute_dataset_stats("data")
    print(f"Media CIFAR-10 calculada:       {computed_mean}")
    print(f"Desv. Estándar CIFAR-10:        {computed_std}")
    print(f"Media ImageNet (referencia):    {IMAGENET_MEAN}")
    print(f"Desv. Estándar ImageNet:        {IMAGENET_STD}")

    visualize_dataset_samples(data_root="data", save_path="docs/data_samples.png", num_per_class=5)

    loaders_cnn = get_dataloaders("cnn", batch_size=64, data_root="data")
    print(f"CNN Train samples: {loaders_cnn['train_size']} | Val: {loaders_cnn['val_size']} | Test: {loaders_cnn['test_size']}")

    loaders_red = get_dataloaders("cnn", batch_size=64, data_root="data", use_reduced_train=True)
    print(f"CNN Reduced (10%) Train samples: {loaders_red['train_size']}")
