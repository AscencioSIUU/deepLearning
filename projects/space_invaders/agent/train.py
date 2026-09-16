"""Entrena UNA iteracion.

    python -m agent.train --config i2_dqn_1m
    python -m agent.train --config i2_dqn_1m --device cpu --resume

Escribe en runs/<config>/:
    tb/              logs de tensorboard (curvas de entrenamiento, seccion 2.3 del reporte)
    checkpoints/     snapshots periodicos, por si hay que reanudar
    model.zip        pesos finales
    config.json      la configuracion exacta usada, para poder reproducir
"""

import argparse
import json
import time
from pathlib import Path

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import QRDQN

from agent.configs import CONFIGS, resolve_device
from agent.env import make_train_env

RUNS = Path(__file__).parent / "runs"
ALGOS = {"DQN": DQN, "QRDQN": QRDQN, "PPO": PPO}


def train(cfg, device=None, resume=False):
    run_dir = RUNS / cfg.nombre
    run_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device or cfg.device)

    venv = make_train_env(cfg)
    Algo = ALGOS[cfg.algo]

    modelo_previo = run_dir / "model.zip"
    if resume and modelo_previo.exists():
        print(f"reanudando desde {modelo_previo}")
        model = Algo.load(modelo_previo, env=venv, device=device)
    else:
        model = Algo(
            "CnnPolicy", venv,
            verbose=1,
            seed=cfg.seed,
            device=device,
            tensorboard_log=str(run_dir / "tb"),
            **cfg.hp,
        )

    print(f"\n[{cfg.nombre}] {cfg.algo} | {cfg.total_timesteps:,} pasos | "
          f"{cfg.n_envs} entornos | device={model.device}")
    print(f"cambio: {cfg.cambio}\n")

    checkpoint = CheckpointCallback(
        save_freq=max(50_000 // cfg.n_envs, 1),
        save_path=str(run_dir / "checkpoints"),
        name_prefix=cfg.nombre,
    )

    t0 = time.time()
    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=checkpoint,
        reset_num_timesteps=not resume,
        progress_bar=True,
    )
    dur = time.time() - t0

    model.save(run_dir / "model")
    venv.close()

    (run_dir / "config.json").write_text(json.dumps({
        "nombre": cfg.nombre, "cambio": cfg.cambio, "algo": cfg.algo,
        "total_timesteps": cfg.total_timesteps, "n_envs": cfg.n_envs,
        "seed": cfg.seed, "device": str(model.device), "hp": cfg.hp,
        "segundos": round(dur), "fps": round(cfg.total_timesteps / dur),
    }, indent=2))

    print(f"\n[{cfg.nombre}] listo en {dur/60:.1f} min "
          f"({cfg.total_timesteps/dur:.0f} FPS) -> {run_dir/'model.zip'}")
    return run_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--device", default=None, help="auto | cpu | mps")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    train(CONFIGS[a.config], device=a.device, resume=a.resume)
