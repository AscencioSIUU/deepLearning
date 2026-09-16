"""El ladder de iteraciones, en codigo.

Una iteracion = una entrada de CONFIGS. La regla del proyecto es cambiar UNA sola cosa
respecto a la iteracion anterior; el campo `cambio` documenta cual, y acaba en la tabla de
la seccion 2.3 del trabajo escrito.

Hiperparametros de I2 en adelante: receta de rl-baselines3-zoo para Atari.
"""

from dataclasses import dataclass, field


def resolve_device(device="auto"):
    """SB3 con "auto" solo detecta CUDA y cae a CPU, pero en Apple Silicon MPS es ~2.4x
    mas rapido (368 vs 153 FPS medidos en el M5 Pro). Hay que pedirlo explicitamente."""
    if device not in (None, "auto"):
        return device
    import torch
    return "mps" if torch.backends.mps.is_available() else "auto"


@dataclass
class Config:
    nombre: str
    cambio: str                  # que cambia respecto a la iteracion anterior
    algo: str = "DQN"            # DQN | QRDQN | PPO
    total_timesteps: int = 100_000
    n_envs: int = 1
    seed: int = 0
    env_id: str = "ALE/SpaceInvaders-v5"
    device: str = "auto"
    video_length: int = 6000     # pasos maximos grabados (un episodio cabe de sobra)
    hp: dict = field(default_factory=dict)   # se pasa tal cual al constructor de SB3


# Receta de Atari del rl-zoo. buffer_size=100k y no 1M: con frames apilados, 1M de
# transiciones no cabe en 24 GB de RAM.
DQN_ATARI = dict(
    learning_rate=1e-4,
    buffer_size=100_000,
    learning_starts=100_000,
    batch_size=32,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=1_000,
    exploration_fraction=0.1,
    exploration_final_eps=0.01,
    optimize_memory_usage=False,
)

PPO_ATARI = dict(
    learning_rate=2.5e-4,
    n_steps=128,
    batch_size=256,
    n_epochs=4,
    clip_range=0.1,
    vf_coef=0.5,
    ent_coef=0.01,
)


CONFIGS = {
    # I1 - valida el circuito completo, no el score.
    "i1_smoke": Config(
        nombre="i1_smoke",
        cambio="DQN de SB3 por defecto, corrida corta para validar el pipeline",
        algo="DQN",
        total_timesteps=100_000,
        hp=dict(learning_starts=10_000, buffer_size=50_000, exploration_fraction=0.3),
    ),

    # I2 - primera corrida seria.
    "i2_dqn_1m": Config(
        nombre="i2_dqn_1m",
        cambio="hiperparametros de Atari del rl-zoo en lugar de los de SB3 por defecto",
        algo="DQN",
        total_timesteps=1_000_000,
        hp=dict(DQN_ATARI),
    ),

    # I3 - mismo agente, mas presupuesto. Aisla el efecto de los pasos.
    "i3_dqn_5m": Config(
        nombre="i3_dqn_5m",
        cambio="mismos hiperparametros que i2, 5x mas pasos de entrenamiento",
        algo="DQN",
        total_timesteps=5_000_000,
        hp=dict(DQN_ATARI),
    ),

    # I4 - cambia el algoritmo, mismo presupuesto que I3.
    "i4_qrdqn_5m": Config(
        nombre="i4_qrdqn_5m",
        cambio="QRDQN (distribucional) en lugar de DQN, mismo presupuesto que i3",
        algo="QRDQN",
        total_timesteps=5_000_000,
        hp=dict(DQN_ATARI),
    ),

    # I5 - familia distinta: on-policy con entornos paralelos.
    "i5_ppo_10m": Config(
        nombre="i5_ppo_10m",
        cambio="PPO con 12 entornos paralelos, aprovecha los 15 nucleos del M5 Pro",
        algo="PPO",
        total_timesteps=10_000_000,
        n_envs=12,
        hp=dict(PPO_ATARI),
    ),
}
