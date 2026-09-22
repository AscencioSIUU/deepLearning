# Space Invaders — CC3092 Deep Learning y Sistemas Inteligentes

Repositorio: <https://github.com/AscencioSIUU/deepLearning/tree/main/projects/space_invaders>

Esta carpeta tiene **dos trabajos** sobre el mismo juego:

| | Qué es | Dónde | Estado |
|---|---|---|---|
| **Lab #5** | Infraestructura para que un agente interactúe con ALE y se grabe en video. No entrena nada. | `lab5/` | entregado, congelado |
| **Proyecto 2** | Entrenar un agente de Reinforcement Learning que juegue bien. | `agent/` | en curso |

---

## Proyecto 2 — el agente

El agente aprende **solo**, jugando: no hay dataset etiquetado ni demostraciones humanas.
Recibe como recompensa el incremento de score que ALE lee de la memoria de la consola, y
ajusta su política para maximizarlo.

### Cómo correrlo

```sh
pip install -r requirements.txt

python -m agent.env --check                          # valida el contrato de los entornos
python -m agent.train    --config i1_smoke           # entrena una iteración (~4 min)
python -m agent.evaluate --config i1_smoke --video   # 5 episodios greedy + video
```

`--config` acepta cualquier nombre de `agent/configs.py`. Cada uno es una iteración del
plan de entrenamiento. `python -m agent.evaluate --config <cfg>` sin más flags corre los 5
episodios oficiales (semillas 0-4, política greedy) y añade la fila a `runs/results.csv`;
es el mismo script pensado para el día de la competencia — carga pesos, evalúa y genera
video sin pasos manuales.

Flags adicionales de `evaluate.py`:
- `--video` graba el episodio de la semilla 0 (`videos/<numero>/`).
- `--solo-video` re-graba el video sin tocar `results.csv` (para iteraciones ya cerradas).
- `--episodios N` corre N episodios en vez de 5, para comparar candidatos con más precisión
  estadística; no se registra en el CSV (la cifra oficial es siempre la de 5).
- `--checkpoint <ruta>` evalúa un checkpoint intermedio en vez de `model.zip`.

### Cargar el modelo final

Los pesos de cualquier iteración quedan en `agent/runs/<config>/model.zip`. Para cargar el
mejor modelo a la fecha (ver tabla de abajo) fuera de `evaluate.py`:

```python
from stable_baselines3 import PPO   # DQN o QRDQN segun la iteracion, ver agent/configs.py
model = PPO.load("agent/runs/<config>/model.zip", device="cpu")  # o "mps"
```

El preprocesamiento (frame stack, resize, grayscale) hay que reconstruirlo con
`agent.env.make_eval_env(cfg)` — los pesos por sí solos no incluyen los wrappers.

### Los cuatro archivos

| Archivo | Responsabilidad |
|---|---|
| `agent/env.py` | Construye los entornos. **Única fuente de verdad del preprocesamiento.** |
| `agent/configs.py` | Una `Config` por iteración. El plan de entrenamiento, en código. |
| `agent/train.py` | Entrena una iteración. Guarda pesos, checkpoints y logs de tensorboard. |
| `agent/evaluate.py` | Carga pesos, evalúa con política greedy, registra en `runs/results.csv` y graba video. |

### Dónde quedan los resultados

```
agent/runs/
├── results.csv                  # una fila por iteracion: media, maximo, min, std, max_esperado
└── <iteracion>/
    ├── config.json               # la configuracion exacta usada (hp, fps, duracion)
    ├── model.zip                 # pesos finales (no versionado)
    ├── checkpoints/               # snapshots cada 500k pasos (no versionado)
    └── tb/                        # curvas para tensorboard (no versionado)

videos/
└── <numero>/                    # ej. videos/i12/ -- mp4 del episodio completo de esa iteracion
```

Ver las curvas: `tensorboard --logdir agent/runs`

### Estado actual

El líder actual es **`i12_ppo_80m`** (PPO, 80M pasos): media 36377.0, máximo 52575.0 en la
evaluación oficial de 5 episodios (media 26105.2, máximo 52605.0 confirmado con 20). El
ladder completo, el análisis de cada iteración y las decisiones tomadas con su evidencia
están en [`docs/plan-entrenamiento.md`](docs/plan-entrenamiento.md) — ese documento y
`agent/runs/results.csv` son la fuente de verdad, esta tabla puede quedar desactualizada
mientras el ladder sigue en curso.

### Tres cosas que cuestan tiempo si no se saben

- **El frameskip venía duplicado.** `ALE/SpaceInvaders-v5` trae `frameskip=4` y el
  `AtariWrapper` de SB3 aplica otro 4: skip efectivo de 16, con el agente actuando 4 veces
  menos de lo debido. Se corrige con `frameskip=1` en el entorno base. `--check` lo verifica.
- **`device="auto"` de SB3 no detecta MPS**, sólo CUDA, y cae a CPU. En este M5 Pro eso
  cuesta 2.4x de velocidad (153 vs 368 FPS). Lo resuelve `resolve_device()`.
- **Evaluación y entrenamiento difieren a propósito** en dos cosas: al evaluar la recompensa
  no se recorta y el episodio dura las 3 vidas, porque la métrica es el score real de una
  partida. Todo lo demás es idéntico.

---

## Lab #5 — la infraestructura (entregado)

Módulo de cinco funciones reutilizables sobre ALE, sin entrenar nada:
`crear_entorno`, `agente_aleatorio`, `agente_regla_simple`, `ejecutar_episodio` y
`generar_video_agente`. Verificado con `python lab5/ale_utils.py`.

**Agente aleatorio** — 298 pasos, recompensa 35.0:

![Agente aleatorio](lab5/figs/spaceinvaders_random.gif)

**Agente de regla simple** — 405 pasos, recompensa 130.0:

![Agente de regla simple](lab5/figs/spaceinvaders_regla.gif)

Sobre 10 episodios (semillas 0-9): el aleatorio saca 123.5 de media y la regla simple 141.0.
La regla gana un 14% pero **sobrevive menos** (406.7 pasos contra 461.0): no juega mejor a la
defensiva, sólo dispara más por unidad de tiempo. Eso es lo que motiva el Proyecto 2 — una
política útil tiene que ser función de la observación, y ninguna regla escrita a mano lo es.

Videos en calidad original: [`lab5/videos/`](lab5/videos/). Notebook con la investigación
completa: [`lab5/lab5_ale_space_invaders.ipynb`](lab5/lab5_ale_space_invaders.ipynb).

---

## Estructura

```
projects/space_invaders/
├── README.md                # este archivo
├── CLAUDE.md                # reglas de método (local, no versionado)
├── requirements.txt
├── docs/
│   ├── plan-entrenamiento.md    # el ladder de iteraciones y las decisiones
│   └── *.pdf                    # enunciados (no versionados)
├── lab5/                    # Lab #5, congelado
└── agent/                   # Proyecto 2
```

Dependencias: `gymnasium` 1.3.0, `ale-py` 0.12, `torch`, `stable-baselines3[extra]`,
`sb3-contrib`, `moviepy`. Las ROMs de Atari vienen incluidas en `ale-py`; el binario de
`ffmpeg` lo aporta `imageio-ffmpeg`. Probado en Python 3.14 sobre Apple Silicon.
