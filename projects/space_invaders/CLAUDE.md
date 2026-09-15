# Laboratorio #5 — ALE y Space Invaders: reglas de tecnología

Reglas vinculantes para este directorio. El objetivo del lab es la **infraestructura**
para que un agente interactúe con ALE y se grabe en video, **no** entrenar un agente.
El enunciado lo dice explícito: *"no es necesario entrenar un agente inteligente"*.

## Permitido

| Tecnología | Para qué |
|---|---|
| Python 3.10+ | lenguaje exigido por el enunciado |
| `gymnasium` (1.3.0) | API de entornos, `RecordVideo`, spaces |
| `ale-py` (≥0.12) | Arcade Learning Environment + ROMs incluidas |
| `numpy`, `matplotlib` | métricas y gráficas |
| `moviepy` / `imageio-ffmpeg` | sólo como backend de `RecordVideo`, nunca invocados a mano |
| `nbformat` / `nbconvert` | sólo para construir y ejecutar el notebook |
| stdlib (`pathlib`, `json`, `tempfile`, …) | todo lo demás |

## Prohibido

| Prohibido | Por qué |
|---|---|
| **Entrenar agentes**: Q-learning, DQN, policy gradient, replay buffer, ε-greedy con decaimiento, redes neuronales, checkpoints | fuera del alcance del lab; sólo agente aleatorio y regla fija no aprendida |
| `torch`, `tensorflow`, `keras`, `jax` | no hay modelo que entrenar |
| `stable-baselines3`, `ray[rllib]`, `tianshou`, `cleanrl` | sustituyen la infraestructura que es el objeto del lab |
| `gym` legacy (OpenAI Gym) | reemplazado por `gymnasium`; mezclar APIs rompe el contrato de 5 valores de `step()` |
| `atari-py`, `AutoROM` | obsoletos: `ale-py` ≥0.10 ya trae las ROMs |
| `opencv`, `Pillow` para preprocesar frames a mano | si hiciera falta, se usan los wrappers de Gymnasium (`AtariPreprocessing`, `FrameStackObservation`); en este lab sólo se documentan, no se aplican |
| Escribir mp4 a mano (`imageio.mimwrite`, `ffmpeg` por `subprocess`, captura propia de frames) | el enunciado exige `gymnasium.wrappers.RecordVideo` |
| Clases, factories o capas de configuración sobre las 5 funciones | deben quedar como funciones sueltas con las firmas exactas del enunciado |
| `render_mode="human"` o cualquier dependencia de display | grabar requiere `rgb_array`; el lab debe correr headless |

## Invariantes del módulo

- `ale_utils.py` expone exactamente: `crear_entorno`, `agente_aleatorio`,
  `agente_regla_simple`, `ejecutar_episodio`, `generar_video_agente`. Las firmas son las
  del enunciado y no se cambian.
- `gym.register_envs(ale_py)` se ejecuta al importar el módulo; sin eso
  `gym.make("ALE/SpaceInvaders-v5")` lanza `NameNotFound`.
- `env.close()` siempre, o `RecordVideo` deja el último `.mp4` sin escribir.
- Las funciones son genéricas: lo que funciona con `ALE/SpaceInvaders-v5` debe funcionar
  con `CartPole-v1`.
- El notebook **importa** `ale_utils`; no redefine las funciones.

## Verificación

```sh
python ale_utils.py    # self-check con asserts; debe imprimir "self-check OK"
grep -rniE "torch|tensorflow|stable_baselines|rllib|autorom|atari_py|cv2" --include=*.py .
```
