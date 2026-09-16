"""Modulo de funciones reutilizables para interactuar con entornos de Atari via ALE.

Laboratorio #5 - CC3092. No entrena agentes: provee la infraestructura (crear entorno,
correr episodios, grabar video) que sera la base para entrenar agentes en el futuro.

Las funciones son genericas: funcionan igual con "ALE/SpaceInvaders-v5" que con
"CartPole-v1" u otro entorno de Gymnasium.
"""

from pathlib import Path

import ale_py
import gymnasium as gym

# Obligatorio desde ale-py 0.10: sin esto gym.make("ALE/...") lanza NameNotFound.
gym.register_envs(ale_py)


def crear_entorno(nombre_entorno, video_folder=None, episode_trigger=None,
                  name_prefix="rl-video", render_mode="rgb_array", **kwargs):
    """Crea un entorno de Gymnasium, opcionalmente envuelto para grabar video.

    video_folder    carpeta destino de los .mp4; si es None no se graba.
    episode_trigger funcion ep_idx -> bool; por defecto graba todos los episodios.
    render_mode     "rgb_array" es indispensable para que RecordVideo tenga frames.
    kwargs          se pasan tal cual a gym.make (obs_type, frameskip,
                    repeat_action_probability, full_action_space, ...).
    """
    env = gym.make(nombre_entorno, render_mode=render_mode, **kwargs)
    if video_folder is not None:
        Path(video_folder).mkdir(parents=True, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_folder),
            episode_trigger=episode_trigger or (lambda ep: True),
            name_prefix=name_prefix,
        )
    return env


def agente_aleatorio(observation, env):
    """Baseline: accion uniforme del espacio de acciones. Ignora la observacion."""
    return env.action_space.sample()


def agente_regla_simple(observation, env):
    """Politica fija, sin aprendizaje: barre de lado a lado disparando.

    Alterna RIGHTFIRE/LEFTFIRE cada PASOS_POR_TRAMO pasos para cubrir el ancho de la
    pantalla mientras dispara de forma continua, en vez de moverse al azar. No mira la
    observacion: es una regla sobre el contador de pasos, no una politica aprendida.
    Si el entorno no tiene las 6 acciones de Space Invaders, cae al agente aleatorio.
    """
    acciones = getattr(env.unwrapped, "get_action_meanings", lambda: [])()
    if "RIGHTFIRE" not in acciones or "LEFTFIRE" not in acciones:
        return agente_aleatorio(observation, env)

    PASOS_POR_TRAMO = 20
    agente_regla_simple.t += 1
    hacia_la_derecha = (agente_regla_simple.t // PASOS_POR_TRAMO) % 2 == 0
    return acciones.index("RIGHTFIRE" if hacia_la_derecha else "LEFTFIRE")


def _reset_regla_simple():
    """Reinicia el contador de pasos. ejecutar_episodio lo llama en cada reset."""
    agente_regla_simple.t = -1


agente_regla_simple.t = -1
agente_regla_simple.reset = _reset_regla_simple


def ejecutar_episodio(env, funcion_agente, max_steps=10000, seed=None):
    """Corre un episodio completo hasta terminated/truncated o max_steps.

    Retorna (pasos, recompensa_total).
    """
    # Si el agente lleva estado por episodio (p. ej. un contador de pasos), lo reinicia.
    # Sin esto los resultados dependen de cuantas veces se llamo antes al agente.
    reset_agente = getattr(funcion_agente, "reset", None)
    if reset_agente is not None:
        reset_agente()
    if seed is not None:
        # reset(seed=) NO siembra el RNG del action_space; sin esto agente_aleatorio
        # da resultados distintos en cada corrida.
        env.action_space.seed(seed)
    observation, _ = env.reset(seed=seed)
    pasos, recompensa_total = 0, 0.0
    while pasos < max_steps:
        accion = funcion_agente(observation, env)
        observation, recompensa, terminated, truncated, _ = env.step(accion)
        pasos += 1
        recompensa_total += float(recompensa)
        if terminated or truncated:
            break
    return pasos, recompensa_total


def generar_video_agente(nombre_entorno, funcion_agente, video_folder,
                         name_prefix, n_episodios=1, max_steps=10000, seed=None,
                         **kwargs):
    """Crea el entorno con grabacion, corre n_episodios y cierra el entorno.

    env.close() es indispensable: RecordVideo escribe el ultimo .mp4 al cerrar.
    Retorna (rutas_mp4, metricas) donde metricas es una lista de (pasos, recompensa).
    """
    video_folder = Path(video_folder)
    previos = set(video_folder.glob("*.mp4")) if video_folder.exists() else set()

    env = crear_entorno(nombre_entorno, video_folder=video_folder,
                        name_prefix=name_prefix, **kwargs)
    metricas = []
    try:
        for ep in range(n_episodios):
            semilla = None if seed is None else seed + ep
            metricas.append(ejecutar_episodio(env, funcion_agente, max_steps, semilla))
    finally:
        env.close()

    rutas = sorted(str(p) for p in set(video_folder.glob("*.mp4")) - previos)
    return rutas, metricas


if __name__ == "__main__":
    import tempfile

    env = crear_entorno("ALE/SpaceInvaders-v5")
    assert env.observation_space.shape == (210, 160, 3), env.observation_space
    assert env.action_space.n == 6, env.action_space
    obs, _ = env.reset(seed=0)
    for agente in (agente_aleatorio, agente_regla_simple):
        a = agente(obs, env)
        assert env.action_space.contains(a), (agente.__name__, a)
    env.close()

    # generico: la misma funcion sirve para un entorno no-Atari
    cartpole = crear_entorno("CartPole-v1")
    pasos, retorno = ejecutar_episodio(cartpole, agente_aleatorio, seed=0)
    cartpole.close()
    assert pasos > 0 and retorno > 0, (pasos, retorno)

    # el agente con estado da el mismo resultado sin importar cuantas veces se uso antes
    env = crear_entorno("ALE/SpaceInvaders-v5")
    a = ejecutar_episodio(env, agente_regla_simple, max_steps=300, seed=0)
    ejecutar_episodio(env, agente_regla_simple, max_steps=57, seed=1)   # desfasa el contador
    b = ejecutar_episodio(env, agente_regla_simple, max_steps=300, seed=0)
    env.close()
    assert a == b, (a, b)

    # max_steps corta el episodio aunque no haya terminado
    cartpole = crear_entorno("CartPole-v1")
    pasos, _ = ejecutar_episodio(cartpole, agente_aleatorio, max_steps=3, seed=0)
    cartpole.close()
    assert pasos <= 3, pasos

    with tempfile.TemporaryDirectory() as tmp:
        rutas, metricas = generar_video_agente(
            "ALE/SpaceInvaders-v5", agente_aleatorio, tmp, "selfcheck",
            n_episodios=1, max_steps=200, seed=0)
        assert len(rutas) == 1, rutas
        assert Path(rutas[0]).stat().st_size > 0, "el mp4 quedo vacio"
        assert len(metricas) == 1 and metricas[0][0] > 0, metricas

    print("ale_utils: self-check OK")
