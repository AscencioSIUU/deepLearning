"""Evalua una iteracion como lo hara el dia de la competencia.

    python -m agent.evaluate --config i2_dqn_1m --video

Corre 5 episodios completos con politica greedy (sin exploracion), uno por semilla 0..4,
y anade una fila a runs/results.csv. Ese CSV es la tabla de la seccion 2.3 del reporte.

Sobre la metrica: el enunciado se contradice. La tabla de "Informacion clave" dice
"puntaje promedio", pero la seccion 3 dice que el ranking usa "el mayor de esos 5
episodios". Se registran las dos, mas la desviacion.
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN, PPO
from sb3_contrib import QRDQN

from agent.configs import CONFIGS, resolve_device
from agent.env import make_eval_env

RUNS = Path(__file__).parent / "runs"
RESULTS = RUNS / "results.csv"
ALGOS = {"DQN": DQN, "QRDQN": QRDQN, "PPO": PPO}
N_EPISODIOS = 5


def _un_episodio(model, cfg, seed, video_folder=None):
    """Una partida completa con politica greedy. Retorna (score, pasos)."""
    env = make_eval_env(cfg, seed=seed, video_folder=video_folder,
                        name_prefix=f"{cfg.nombre}_seed{seed}")
    obs = env.reset()
    score, pasos = 0.0, 0
    while True:
        accion, _ = model.predict(obs, deterministic=True)   # greedy, sin exploracion
        obs, recompensa, done, _ = env.step(accion)
        score += float(recompensa[0])
        pasos += 1
        if done[0]:
            break
    env.close()
    return score, pasos


def evaluate(cfg, grabar_video=False, device=None):
    run_dir = RUNS / cfg.nombre
    pesos = run_dir / "model.zip"
    if not pesos.exists():
        raise SystemExit(f"no hay pesos en {pesos}; corre primero:  "
                         f"python -m agent.train --config {cfg.nombre}")

    model = ALGOS[cfg.algo].load(pesos, device=resolve_device(device or cfg.device))
    print(f"[{cfg.nombre}] {cfg.algo} cargado de {pesos} (device={model.device})")

    scores, todos_pasos = [], []
    for seed in range(N_EPISODIOS):
        # Solo el primer episodio se graba: el entregable pide un video, no cinco.
        carpeta = (run_dir / "videos") if (grabar_video and seed == 0) else None
        score, pasos = _un_episodio(model, cfg, seed, carpeta)
        scores.append(score)
        todos_pasos.append(pasos)
        print(f"  seed {seed}: score {score:7.1f} | {pasos:5d} pasos")

    scores = np.array(scores)
    fila = {
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "iteracion": cfg.nombre,
        "algo": cfg.algo,
        "timesteps": cfg.total_timesteps,
        "cambio": cfg.cambio,
        "score_medio": round(float(scores.mean()), 1),
        "score_max": round(float(scores.max()), 1),   # esta es la del ranking
        "score_min": round(float(scores.min()), 1),
        "score_std": round(float(scores.std()), 1),
        "pasos_medios": round(float(np.mean(todos_pasos)), 1),
    }

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    nuevo = not RESULTS.exists()
    with RESULTS.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fila))
        if nuevo:
            w.writeheader()
        w.writerow(fila)

    print(f"\n  media {fila['score_medio']}  |  MAXIMO {fila['score_max']} (ranking)  "
          f"|  std {fila['score_std']}")
    print(f"  -> fila anadida a {RESULTS}")
    if grabar_video:
        videos = sorted((run_dir / "videos").glob("*.mp4"))
        print(f"  -> video: {videos[-1] if videos else 'NO SE GENERO'}")
    return fila


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    evaluate(CONFIGS[a.config], grabar_video=a.video, device=a.device)
