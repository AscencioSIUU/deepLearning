"""Evalua una iteracion como lo hara el dia de la competencia.

    python -m agent.evaluate --config i2_dqn_1m --video
    python -m agent.evaluate --config i2_dqn_1m --solo-video       # solo graba, no toca el CSV
    python -m agent.evaluate --config i2_dqn_1m --episodios 20     # elegir entre candidatos

Corre 5 episodios completos con politica greedy (sin exploracion), uno por semilla 0..4,
y anade una fila a runs/results.csv. Ese CSV es la tabla de la seccion 2.3 del reporte.

Sobre la metrica: el enunciado se contradice. La tabla de "Informacion clave" dice
"puntaje promedio", pero la seccion 3 dice que el ranking usa "el mayor de esos 5
episodios". Se registran las dos, mas la desviacion y max_esperado = media + 1.16*std
(el maximo esperado de 5 muestras no es un objetivo aparte, ver docs/plan-entrenamiento.md).
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
FINAL = Path(__file__).parent.parent / "final"   # entrega: video + resultados por iteracion
ALGOS = {"DQN": DQN, "QRDQN": QRDQN, "PPO": PPO}
N_EPISODIOS = 5
# aprox. del maximo esperado de N muestras normales: media + c*std (c=1.16 para N=5)
C_MAXIMO_N5 = 1.16


def _carpeta_final(cfg):
    return FINAL / cfg.nombre


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


def evaluate(cfg, grabar_video=False, device=None, n_episodios=N_EPISODIOS, checkpoint=None):
    run_dir = RUNS / cfg.nombre
    # ponytail: DQN puede tener su pico antes del final y degradarse despues (visto en
    # i8: 6M > 10M en greedy). --checkpoint deja evaluar un punto intermedio sin re-entrenar.
    pesos = Path(checkpoint) if checkpoint else (run_dir / "model.zip")
    if not pesos.exists():
        raise SystemExit(f"no hay pesos en {pesos}; corre primero:  "
                         f"python -m agent.train --config {cfg.nombre}")

    model = ALGOS[cfg.algo].load(pesos, device=resolve_device(device or cfg.device))
    print(f"[{cfg.nombre}] {cfg.algo} cargado de {pesos} (device={model.device})")

    scores, todos_pasos = [], []
    for seed in range(n_episodios):
        # Solo el primer episodio se graba: el entregable pide un video, no cinco.
        carpeta = _carpeta_final(cfg) if (grabar_video and seed == 0) else None
        score, pasos = _un_episodio(model, cfg, seed, carpeta)
        scores.append(score)
        todos_pasos.append(pasos)
        print(f"  seed {seed}: score {score:7.1f} | {pasos:5d} pasos")

    scores = np.array(scores)
    media, std = float(scores.mean()), float(scores.std())
    fila = {
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "iteracion": cfg.nombre,
        "algo": cfg.algo,
        "timesteps": cfg.total_timesteps,
        "cambio": cfg.cambio,
        "score_medio": round(media, 1),
        "score_max": round(float(scores.max()), 1),   # esta es la del ranking
        "score_min": round(float(scores.min()), 1),
        "score_std": round(std, 1),
        # media + c*std: el maximo esperado de N episodios no es un objetivo aparte,
        # es la media mas una prima por varianza (ver docs/plan-entrenamiento.md).
        "max_esperado": round(media + C_MAXIMO_N5 * std, 1),
        "pasos_medios": round(float(np.mean(todos_pasos)), 1),
    }

    print(f"\n  media {fila['score_medio']}  |  MAXIMO {fila['score_max']} (ranking)  "
          f"|  std {fila['score_std']}  |  max_esperado {fila['max_esperado']}")

    if n_episodios == N_EPISODIOS:
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        nuevo = not RESULTS.exists()
        with RESULTS.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(fila))
            if nuevo:
                w.writeheader()
            w.writerow(fila)
        print(f"  -> fila anadida a {RESULTS}")

        carpeta_final = _carpeta_final(cfg)
        carpeta_final.mkdir(parents=True, exist_ok=True)
        (carpeta_final / "resultados.md").write_text(
            f"# Resultados — {cfg.nombre}\n\n"
            f"- Algoritmo: {fila['algo']}\n"
            f"- Timesteps entrenados: {fila['timesteps']:,}\n"
            f"- Cambio respecto a la iteracion anterior: {fila['cambio']}\n\n"
            "| score medio | score maximo (ranking) | score minimo | std | pasos medios |\n"
            "|---|---|---|---|---|\n"
            f"| {fila['score_medio']} | {fila['score_max']} | {fila['score_min']} | "
            f"{fila['score_std']} | {fila['pasos_medios']} |\n\n"
            f"Evaluado: {fila['fecha']}\n"
        )
        print(f"  -> resultados anadidos a {carpeta_final/'resultados.md'}")
    else:
        # corrida de seleccion con mas episodios: no es la cifra oficial del reporte,
        # no se mezcla con las filas de N_EPISODIOS en el CSV ni con final/.
        print(f"  -> {n_episodios} episodios, NO se registra en {RESULTS} "
              f"(la cifra oficial es con {N_EPISODIOS})")

    if grabar_video:
        videos = sorted(_carpeta_final(cfg).glob("*.mp4"))
        print(f"  -> video: {videos[-1] if videos else 'NO SE GENERO'}")
    return fila


def grabar_video(cfg, device=None):
    """Graba un episodio greedy sin tocar results.csv (re-grabar iteraciones ya cerradas)."""
    pesos = RUNS / cfg.nombre / "model.zip"
    model = ALGOS[cfg.algo].load(pesos, device=resolve_device(device or cfg.device))
    carpeta = _carpeta_final(cfg)
    score, pasos = _un_episodio(model, cfg, seed=0, video_folder=carpeta)
    print(f"[{cfg.nombre}] video -> {carpeta} (score {score}, {pasos} pasos)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--solo-video", action="store_true",
                     help="graba un episodio sin tocar results.csv (re-grabar iteraciones ya cerradas)")
    ap.add_argument("--episodios", type=int, default=N_EPISODIOS,
                     help="para elegir entre candidatos al final con mas semillas; no se registra en el CSV")
    ap.add_argument("--checkpoint", default=None,
                     help="evalua un checkpoint intermedio en vez de model.zip (runs/<cfg>/checkpoints/*.zip)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    if a.solo_video:
        grabar_video(CONFIGS[a.config], device=a.device)
    else:
        evaluate(CONFIGS[a.config], grabar_video=a.video, device=a.device,
                 n_episodios=a.episodios, checkpoint=a.checkpoint)
