# CC3092 - Deep Learning y Sistemas Inteligentes

Repositorio de curso: laboratorios, hojas de trabajo (HDT) y proyectos.

## Estructura

```
deepLearning/
├── labs/           # laboratorios
│   └── lab1-mlp/    # Lab #1: MLP de regresión (California Housing)
├── hdt/             # hojas de trabajo
└── projects/        # proyectos
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Laboratorios

- [Lab #1 — MLP de regresión](labs/lab1-mlp/lab1_mlp.ipynb): entrenamiento de un MLP en
  PyTorch sobre el dataset California Housing Prices. Explicadores de cada etapa en
  [`labs/lab1-mlp/docs/`](labs/lab1-mlp/docs/).
