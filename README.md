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
