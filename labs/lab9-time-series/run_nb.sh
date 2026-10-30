#!/bin/sh
# Regenera y ejecuta el notebook con el kernel del venv del lab
set -e
cd "$(dirname "$0")"
.venv/bin/python build_nb.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=lab9 notebook.ipynb
