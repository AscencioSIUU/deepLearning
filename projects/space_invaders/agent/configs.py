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
    # pasos maximos grabados. No es un objetivo, es una salvaguarda: el episodio real corta
    # la grabacion por `done` (evaluate.py), no por este numero. 6000 alcanzaba para los
    # agentes tempranos, pero i12 (80M) tuvo episodios de hasta 27_000 pasos y con 30_000 el
    # margen ya era estrecho. Se sube bien alto para que ningun episodio futuro se corte.
    video_length: int = 500_000
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


def lineal(valor_inicial):
    """Schedule del rl-zoo: decae linealmente hasta 0 al acabar el entrenamiento."""
    return lambda progreso_restante: progreso_restante * valor_inicial


# i5 uso learning_rate y clip_range constantes; el rl-zoo los define como lin_2.5e-4 y
# lin_0.1 (decaen a 0). Sin el annealing, el update sigue siendo agresivo al final del
# entrenamiento (clip_fraction 0.263 al 83% de i5).
PPO_ATARI_TUNED = dict(PPO_ATARI, learning_rate=lineal(2.5e-4), clip_range=lineal(0.1))


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

    # I6 - misma familia y presupuesto que i5, aisla el efecto del annealing de lr/clip.
    "i6_ppo_tuned": Config(
        nombre="i6_ppo_tuned",
        cambio="learning_rate y clip_range con schedule lineal del rl-zoo en lugar de constantes",
        algo="PPO",
        total_timesteps=10_000_000,
        n_envs=12,
        hp=dict(PPO_ATARI_TUNED),
    ),

    # I7 - PPO afinado, doble presupuesto.
    "i7_ppo_40m": Config(
        nombre="i7_ppo_40m",
        cambio="mismos hiperparametros que i6, 4x mas pasos de entrenamiento",
        algo="PPO",
        total_timesteps=40_000_000,
        n_envs=12,
        hp=dict(PPO_ATARI_TUNED),
    ),

    # I8 - DQN al presupuesto para el que el rl-zoo diseño la receta. Aisla si el plateau
    # de i3 (ep_rew_mean 703-708 desde 4.26M) es real o falta de presupuesto.
    "i8_dqn_10m": Config(
        nombre="i8_dqn_10m",
        cambio="mismos hiperparametros que i3, 2x mas pasos de entrenamiento",
        algo="DQN",
        total_timesteps=10_000_000,
        hp=dict(DQN_ATARI),
    ),

    # I9 - QRDQN a velocidad viable. i4 (200 cuantiles) tiene el mejor maximo (1775) pero
    # a 124 FPS; bajar los cuantiles ataca directamente el cuello de la perdida cuantilica.
    "i9_qrdqn_q50": Config(
        nombre="i9_qrdqn_q50",
        cambio="n_quantiles 200 -> 50 en policy_kwargs, resto igual que i4",
        algo="QRDQN",
        total_timesteps=5_000_000,
        hp=dict(DQN_ATARI, policy_kwargs={"n_quantiles": 50}),
    ),

    # I10 - ataca la causa de la oscilacion vista en los checkpoints de i8 (1023 a 513 en
    # 500k pasos), no el sintoma. target_update_interval=1000 es mas frecuente que los
    # 10_000 del Nature DQN; el objetivo se mueve rapido y realimenta la sobreestimacion.
    "i10_dqn_target10k": Config(
        nombre="i10_dqn_target10k",
        cambio="target_update_interval 1000 -> 10000, resto igual que i3",
        algo="DQN",
        total_timesteps=5_000_000,
        hp=dict(DQN_ATARI, target_update_interval=10_000),
    ),

    # I11 - repite el intento de i8 (2x pasos) pero sobre la base sin oscilacion de i10.
    # La curva de i10 seguia subiendo al cerrar (893 en 5M, pico 946 en 4.75M).
    "i11_dqn_target10k_10m": Config(
        nombre="i11_dqn_target10k_10m",
        cambio="mismos hiperparametros que i10, 2x mas pasos de entrenamiento",
        algo="DQN",
        total_timesteps=10_000_000,
        hp=dict(DQN_ATARI, target_update_interval=10_000),
    ),

    # I12 - la curva de i7 seguia subiendo sin plateau al cerrar 40M (pico en el 99% del
    # entrenamiento). Dobla el presupuesto para ver si el techo real esta mas alla de 40M.
    "i12_ppo_80m": Config(
        nombre="i12_ppo_80m",
        cambio="mismos hiperparametros que i7, 2x mas pasos de entrenamiento",
        algo="PPO",
        total_timesteps=80_000_000,
        n_envs=12,
        hp=dict(PPO_ATARI_TUNED),
    ),

    # I13 - la curva de i12 no mostraba plateau al cerrar 80M (pico en el 99%, saltos
    # grandes entre los 40-80M). Dobla otra vez mientras el patron de i7->i12 siga valiendo.
    "i13_ppo_160m": Config(
        nombre="i13_ppo_160m",
        cambio="mismos hiperparametros que i12, 2x mas pasos de entrenamiento",
        algo="PPO",
        total_timesteps=160_000_000,
        n_envs=12,
        hp=dict(PPO_ATARI_TUNED),
    ),
}
