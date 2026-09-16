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

python -m agent.env --check                        # valida el contrato de los entornos
python -m agent.train    --config i1_smoke         # entrena una iteración (~4 min)
python -m agent.evaluate --config i1_smoke --video # 5 episodios greedy + video
```

`--config` acepta cualquier nombre de `agent/configs.py`. Cada uno es una iteración del
plan de entrenamiento.

### Los cuatro archivos

| Archivo | Responsabilidad |
|---|---|
| `agent/env.py` | Construye los entornos. **Única fuente de verdad del preprocesamiento.** |
| `agent/configs.py` | Una `Config` por iteración. El plan de entrenamiento, en código. |
| `agent/train.py` | Entrena una iteración. Guarda pesos, checkpoints y logs de tensorboard. |
| `agent/evaluate.py` | Carga pesos, corre 5 episodios greedy, añade una fila a `runs/results.csv` y graba el video. |

### Dónde quedan los resultados

```
agent/runs/
├── results.csv              # una fila por iteración: media, máximo, desviación
└── <iteracion>/
    ├── config.json          # la configuración exacta usada, y los FPS que dio
    ├── model.zip            # pesos (no versionado: 26 MB)
    ├── tb/                  # curvas para tensorboard (no versionado)
    └── videos/              # mp4 de un episodio completo
```

Ver las curvas: `tensorboard --logdir agent/runs`

### Estado actual

| Iteración | Algoritmo | Steps | Score medio | Score máx |
|---|---|---|---|---|
| agente aleatorio (Lab 5) | — | — | 123.5 | 235 |
| `i1_smoke` | DQN | 100 k | 223.0 | 510 |

El ladder completo de iteraciones, las decisiones tomadas y su evidencia están en
[`docs/plan-entrenamiento.md`](docs/plan-entrenamiento.md).

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
├── CLAUDE.md                # reglas de método del proyecto
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
