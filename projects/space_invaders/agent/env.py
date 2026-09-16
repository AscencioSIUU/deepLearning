"""Construccion del entorno. Unica fuente de verdad del preprocesamiento.

train.py y evaluate.py llaman a estas funciones y nunca redefinen wrappers, porque el
enunciado exige que el preprocesamiento de evaluacion sea identico al de entrenamiento.

La diferencia deliberada entre ambos entornos es el manejo de la RECOMPENSA y de las VIDAS:

  entrenamiento   clip_reward=True, terminal_on_life_loss=True
                  El clipping a {-1,0,+1} permite usar los mismos hiperparametros en juegos
                  con escalas de score distintas. Tratar cada vida como fin de episodio da
                  una senal de fracaso mas inmediata.

  evaluacion      clip_reward=False, terminal_on_life_loss=False
                  La metrica del ranking es el score real de una partida completa. Evaluar
                  con clipping mediria "numero de aciertos", no puntaje.
"""

import argparse

import ale_py
import gymnasium as gym
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import (
    VecFrameStack,
    VecTransposeImage,
    VecVideoRecorder,
)

# Obligatorio desde ale-py 0.10: sin esto gym.make("ALE/...") lanza NameNotFound.
gym.register_envs(ale_py)

ENV_ID = "ALE/SpaceInvaders-v5"
N_STACK = 4  # frames apilados: un frame solo no muestra velocidad ni direccion

# frameskip=1 en el entorno base NO desactiva el frame skipping: lo delega al AtariWrapper,
# que ademas hace max-pool de los 2 ultimos frames para quitar el parpadeo de sprites.
# ALE/SpaceInvaders-v5 trae frameskip=4 de fabrica y el wrapper aplica otro 4 encima, lo que
# da un skip efectivo de 16: el agente actuaria 4 veces menos de lo debido y no podria
# reaccionar. Medido antes de corregirlo: 22 frames de emulador por paso.
# repeat_action_probability (sticky actions, 0.25) se conserva: es del entorno, no del skip.
BASE_ENV_KWARGS = {"frameskip": 1}


def make_train_env(cfg):
    """Entorno vectorizado para entrenar."""
    venv = make_atari_env(
        cfg.env_id,
        n_envs=cfg.n_envs,
        seed=cfg.seed,
        wrapper_kwargs={
            "noop_max": 30,             # arranque aleatorio: los episodios no empiezan igual
            "frame_skip": 4,
            "screen_size": 84,
            "terminal_on_life_loss": True,
            "clip_reward": True,
        },
        env_kwargs=BASE_ENV_KWARGS,
    )
    venv = VecFrameStack(venv, n_stack=N_STACK)
    return VecTransposeImage(venv)      # HWC -> CHW, lo que espera la CNN de SB3


def make_eval_env(cfg, seed=0, video_folder=None, name_prefix="agente"):
    """Entorno de evaluacion: un episodio es una partida completa y el reward es el score.

    Con video_folder, VecVideoRecorder graba el render del entorno base (RGB 210x160x3),
    no la observacion preprocesada de 84x84 en grises.
    """
    venv = make_atari_env(
        cfg.env_id,
        n_envs=1,
        seed=seed,
        wrapper_kwargs={
            "noop_max": 30,
            "frame_skip": 4,
            "screen_size": 84,
            "terminal_on_life_loss": False,   # el episodio dura las 3 vidas
            "clip_reward": False,             # recompensa = score real del juego
        },
        env_kwargs={**BASE_ENV_KWARGS, "render_mode": "rgb_array"},
    )
    venv = VecFrameStack(venv, n_stack=N_STACK)
    if video_folder is not None:
        venv = VecVideoRecorder(
            venv,
            video_folder=str(video_folder),
            record_video_trigger=lambda step: step == 0,
            video_length=cfg.video_length,
            name_prefix=name_prefix,
        )
    return VecTransposeImage(venv)


def _check():
    """Verifica que los dos entornos cumplan el contrato que asume el resto del codigo."""
    from agent.configs import CONFIGS

    cfg = CONFIGS["i1_smoke"]

    train = make_train_env(cfg)
    assert train.observation_space.shape == (N_STACK, 84, 84), train.observation_space
    assert train.num_envs == cfg.n_envs
    train.close()
    print(f"train: obs {train.observation_space.shape} uint8, {cfg.n_envs} entornos  OK")

    # El entorno de evaluacion no debe recortar la recompensa: se comprueba que aparezca
    # al menos un reward fuera de {-1, 0, 1} en una partida entera.
    ev = make_eval_env(cfg)
    ev.reset()
    vistos = set()
    for _ in range(3000):
        _, r, done, _ = ev.step([ev.action_space.sample()])
        vistos.add(float(r[0]))
        if done[0]:
            break
    ev.close()
    fuera = {r for r in vistos if r not in (-1.0, 0.0, 1.0)}
    assert fuera, f"la recompensa de eval parece recortada: solo se vio {vistos}"
    print(f"eval : recompensas sin recortar, se vieron {sorted(fuera)}  OK")

    # El frameskip efectivo debe ser 4, no 16. Este bug es silencioso: el agente entrena
    # igual, solo que actuando 4 veces menos de lo debido, y se nota semanas despues.
    ev = make_eval_env(cfg)
    ev.reset()
    e = ev
    while hasattr(e, "venv"):
        e = e.venv
    ale = e.envs[0].unwrapped.ale
    n0 = ale.getEpisodeFrameNumber()
    for _ in range(20):
        ev.step([0])
    skip = (ale.getEpisodeFrameNumber() - n0) / 20
    ev.close()
    assert skip == 4.0, f"frameskip efectivo {skip}, se esperaba 4"
    print(f"skip : {skip} frames de emulador por paso  OK")

    # El video debe salir en RGB original, no en la observacion de 84x84.
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        ev = make_eval_env(cfg, video_folder=tmp, name_prefix="check")
        ev.reset()                       # render() exige reset() antes
        frame = ev.render(mode="rgb_array")
        for _ in range(60):
            ev.step([ev.action_space.sample()])
        ev.close()
        videos = list(Path(tmp).glob("*.mp4"))
        assert videos and videos[0].stat().st_size > 0, "no se escribio el mp4"
        print(f"video: {videos[0].name} escrito, {videos[0].stat().st_size} bytes  OK")
        if frame is not None:
            assert frame.shape == (210, 160, 3), frame.shape
            print(f"render: {frame.shape} RGB original  OK")

    print("\nenv.py: todas las comprobaciones pasaron")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    if ap.parse_args().check:
        _check()
