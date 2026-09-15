# CC3092 - Deep Learning y Sistemas Inteligentes

Repositorio de curso: laboratorios, hojas de trabajo (HDT) y proyectos.

**Repo:** https://github.com/AscencioSIUU/deepLearning

## Estructura

```
deepLearning/
├── labs/            # laboratorios
├── hdt/             # hojas de trabajo
└── projects/        # proyectos
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# dataset del Lab #1 (no versionado, ver labs/lab1-mlp/data/)
curl -sL -o labs/lab1-mlp/data/housing.csv \
  https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv
```

## Laboratorios

- [Lab #1 — MLP de regresión](labs/lab1-mlp/lab1_mlp.ipynb): entrenamiento de un MLP en
  PyTorch sobre el dataset California Housing Prices. Explicadores de cada etapa en
  [`labs/lab1-mlp/docs/`](labs/lab1-mlp/docs/). Reporte final (PDF):
  [`Reporte - Laboratorio 1 MLP.pdf`](Reporte%20-%20Laboratorio%201%20MLP.pdf).
- [Lab #3 — RNN y LSTM](labs/lab3-rnn-lstm/lab3_rnn_lstm.ipynb): clasificación de sentimiento
  sobre IMDB Reviews comparando MLP (bag-of-embeddings), RNN simple y LSTM (many-to-one),
  16 iteraciones y experimento de longitud de secuencia. Reporte final:
  [`reporte.pdf`](labs/lab3-rnn-lstm/reporte.pdf).

- [Lab #5 — ALE y Space Invaders](projects/space_invaders/lab5_ale_space_invaders.ipynb):
  infraestructura para que un agente interactúe con entornos de Atari a través del Arcade
  Learning Environment y se grabe en video. No se entrena ningún agente: se implementa el
  módulo reutilizable [`ale_utils.py`](projects/space_invaders/ale_utils.py) (`crear_entorno`,
  `agente_aleatorio`, `agente_regla_simple`, `ejecutar_episodio`, `generar_video_agente`) y se
  comparan un agente aleatorio y una regla fija en `ALE/SpaceInvaders-v5`. Videos de los
  agentes, resultados y reglas de tecnología del lab en
  [`projects/space_invaders/README.md`](projects/space_invaders/README.md).

## Proyectos

- [Proyecto #1 — Competencia de Modelación](projects/proy-1-competencia/): MLP en
  PyTorch sobre el dataset Ames House Prices, competencia por RMSE. EDA,
  preprocesamiento, sweep de iteraciones y modelo final en
  [`proy1_mlp.ipynb`](projects/proy-1-competencia/proy1_mlp.ipynb). Explicadores de
  cada etapa en [`projects/proy-1-competencia/docs/`](projects/proy-1-competencia/docs/).
  Reporte final: [`reporte.pdf`](projects/proy-1-competencia/reporte.pdf).
  Reproducción: [`projects/proy-1-competencia/README.md`](projects/proy-1-competencia/README.md).
