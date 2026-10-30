#!/bin/sh
# Regenera y ejecuta el notebook con el kernel del venv del lab (UPTO=n ejecuta solo los batches <= n)
set -e
cd "$(dirname "$0")"
# LightGBM y PyTorch traen cada uno su libomp: cargar dos runtimes de OpenMP en el mismo proceso hace segfault en macOS.
# Con esto LightGBM usa la libomp de torch.
export DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/torch/lib"
.venv/bin/python build_nb.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=lab9 notebook.ipynb
