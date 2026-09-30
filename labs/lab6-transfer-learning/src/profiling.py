"""
profiling.py - Medición de recursos computacionales, parámetros, MACs/FLOPs,
latencia de inferencia y memoria pico en GPU (CUDA / Apple Silicon MPS / CPU).

CC3092 - Deep Learning y Sistemas Inteligentes | Laboratorio #6
"""

import time
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from torchinfo import summary


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Retorna (total_parameters, trainable_parameters) del modelo."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model_flops_and_macs(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 112, 112),
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Calcula los MACs y FLOPs por imagen (forward pass) usando torchinfo.

    Por convención en deep learning, 1 MAC = 2 FLOPs (1 multiplicación + 1 suma).
    """
    model_cpu = model.cpu() if device is None else model.to(device)
    stats = summary(model_cpu, input_size=input_size, verbose=0)

    # total_mult_adds representa los MACs en torchinfo
    macs = stats.total_mult_adds
    flops = macs * 2  # 1 MAC = 2 FLOPs

    return {
        "macs": float(macs),
        "flops": float(flops),
        "macs_m": round(macs / 1e6, 2),  # MegaMACs
        "flops_g": round(flops / 1e9, 3),  # GigaFLOPs
    }


def measure_inference_latency(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 112, 112),
    device: Optional[torch.device] = None,
    num_warmup: int = 10,
    num_runs: int = 50,
) -> Dict[str, float]:
    """Mide la latencia promedio de inferencia por imagen (en milisegundos)

    asegurando la sincronización de hardware (CUDA / MPS).
    """
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(input_size, device=device)

    # Warm-up para estabilizar pipelines del GPU y compilador
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)

    # Sincronización previa
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()

    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
    end_time = time.perf_counter()

    total_time_ms = (end_time - start_time) * 1000.0
    batch_size = input_size[0]
    avg_latency_per_image_ms = total_time_ms / (num_runs * batch_size)

    return {
        "avg_latency_ms": round(avg_latency_per_image_ms, 3),
        "throughput_img_per_sec": round(1000.0 / avg_latency_per_image_ms, 1) if avg_latency_per_image_ms > 0 else 0.0,
    }


def measure_peak_memory(device: torch.device) -> float:
    """Retorna la memoria pico asignada en MB en el dispositivo de aceleración."""
    if device.type == "cuda":
        return round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2)
    elif device.type == "mps" and hasattr(torch.mps, "driver_allocated_memory"):
        return round(torch.mps.driver_allocated_memory() / (1024 ** 2), 2)
    elif device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        return round(torch.mps.current_allocated_memory() / (1024 ** 2), 2)
    return 0.0


def estimate_total_training_flops(
    forward_flops_per_img: float,
    trainable_ratio: float,
    strategy: str,
    num_samples: int,
    epochs: int,
) -> Dict[str, float]:
    """Estima el cómputo total de entrenamiento (en FLOPs) considerando el backward pass:

    - CNN desde cero (100% entrenable):
      Forward: 1x FLOPs
      Backward: 2x FLOPs (gradientes con respecto a entradas + gradientes con respecto a pesos)
      Factor por muestra = ~3x Forward FLOPs.

    - VGG-16 Feature Extractor:
      Backbone congelado: forward pass completo (1x FLOPs del backbone + clasificador).
      Backward pass: se detiene en la entrada del clasificador (sin cómputo de gradientes
      hacia el backbone). El clasificador representa < 1% de los FLOPs del backbone.
      Factor por muestra = Forward_total + 2 * Forward_classifier ≈ ~1.05x Forward FLOPs.

    - VGG-16 Fine-Tuning:
      Backbone congelado inferior: 1x forward.
      Bloques superiores descongelados + clasificador: forward (1x) + backward gradientes (2x).
      Factor por muestra = 1x (congelado) + 3x (descongelado) ≈ ~2.2x Forward FLOPs.
    """
    if strategy == "scratch":
        backward_multiplier = 3.0  # 1x forward + 2x backward
    elif strategy == "feature_extractor":
        backward_multiplier = 1.05  # forward completo + backward solo en clasificador liviano
    elif strategy == "fine_tuning":
        backward_multiplier = 2.20  # forward completo + backward en capas superiores
    else:
        backward_multiplier = 3.0

    total_training_flops = forward_flops_per_img * backward_multiplier * num_samples * epochs
    total_teraflops = total_training_flops / 1e12
    total_petaflops = total_training_flops / 1e15

    return {
        "multiplier": backward_multiplier,
        "total_flops": total_training_flops,
        "total_tflops": round(total_teraflops, 3),
        "total_pflops": round(total_petaflops, 4),
    }
