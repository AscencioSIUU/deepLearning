"""Builds notebook.ipynb from the cell definitions below. Re-run after editing to add batches.
Not part of the deliverable; delete once the notebook is final."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src.strip("\n")))


def code(src):
    cells.append(nbf.v4.new_code_cell(src.strip("\n")))


# ================================================================== Header
md("""
# Laboratorio #7 — NLP end-to-end y Embeddings

**CC3092 Deep Learning y Sistemas Inteligentes**

Pipeline completo: texto crudo → tokens → IDs → embeddings (SGNS propio en PyTorch, Word2Vec de gensim,
GloVe preentrenado) → aritmética vectorial → clasificación de AG News.

- **Corpus de embeddings:** WikiText-103 (raw)
- **Clasificación:** AG News (4 clases)
- **Benchmarks:** `questions-words.txt`, `wordsim353.tsv`, `simlex999.txt` (en `data/`)
""")

# ================================================================== 0. Imports
md("## 0. Imports y configuración")
code("""
import re
import time
import json
import pickle
import random
import collections
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from gensim.models import Word2Vec, KeyedVectors
import gensim.downloader as api

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA = Path("data")
DATA.mkdir(exist_ok=True)
(DATA / "vectors").mkdir(exist_ok=True)
DEVICE = torch.device("cpu")  # SGNS con tablas sparse: CPU rinde mejor que MPS (gathers pequeños, overhead de kernels)
N_WORKERS = 8                 # hilos para gensim
CORPUS_TOKENS = 20_000_000    # prefijo de WikiText-103 usado como corpus principal (de ~100M)
MIN_COUNT = 5
print("torch", torch.__version__, "| device", DEVICE)
""")

# ================================================================== 1. Fase 1
md("""
## Fase 1 — Exploración y preprocesamiento

### 1.1 Carga de WikiText-103

El formato *raw* ya viene separado por espacios, con marcas propias: encabezados `= Título =`, y
`@-@`, `@,@`, `@.@` para guiones/comas/puntos dentro de palabras o números. Se limpian al normalizar.
""")
code("""
wiki = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
lines = [l for l in wiki["train"]["text"] if l.strip() and not l.lstrip().startswith("=")]  # fuera líneas vacías y encabezados
print(f"{len(wiki['train']):,} líneas totales -> {len(lines):,} párrafos")
print(lines[1][:300])
""")

md("""
### 1.2 Tokenizador propio vs spaCy vs NLTK

Pipeline propio: minúsculas → resolver marcas `@x@` → regex que separa palabras (con contracciones tipo
`don't`), números y signos de puntuación. Para embeddings se descarta la puntuación y los números se
colapsan a `<num>` (un vector por cada año/cifra solo diluye el vocabulario).
""")
code("""
WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|\\d+(?:[.,]\\d+)*|[^\\sa-z\\d]")
AT_RE = re.compile(r" @(.)@ ")

def tokenize(text, keep_punct=False):
    \"\"\"texto crudo -> lista de tokens. keep_punct=True mantiene signos (solo para comparar tokenizadores).\"\"\"
    text = AT_RE.sub(r"\\1", text.lower())  # "1 @,@ 000" -> "1,000", "well @-@ known" -> "well-known"
    toks = WORD_RE.findall(text)
    toks = ["<num>" if t[0].isdigit() else t for t in toks]
    return toks if keep_punct else [t for t in toks if t[0].isalpha() or t == "<num>"]

print(tokenize(lines[1][:200], keep_punct=True))
""")
code("""
import spacy
from nltk.tokenize import NLTKWordTokenizer

nlp = spacy.blank("en")  # solo el tokenizador de spaCy, sin modelo (el resto del pipeline no se usa aquí)
nltk_tok = NLTKWordTokenizer()
sample = lines[:3000]

def run(fn):
    t = time.time()
    out = [fn(s) for s in sample]
    return out, time.time() - t

res = {
    "propio (regex)": run(lambda s: tokenize(s, keep_punct=True)),
    "spaCy (blank en)": run(lambda s: [t.text.lower() for t in nlp.make_doc(s)]),
    "NLTK (Treebank)": run(lambda s: [t.lower() for t in nltk_tok.tokenize(AT_RE.sub(r"\\1", s))]),
}
rows = []
for name, (toks, secs) in res.items():
    flat = [t for d in toks for t in d]
    rows.append({"tokenizador": name, "tokens": len(flat), "vocab": len(set(flat)),
                 "seg/3000 párrafos": round(secs, 2)})
pd.DataFrame(rows)
""")
code("""
ex = "The U.S. president's well-known \\"state-of-the-art\\" plan costs $1,000.50 -- isn't cheap!"
for name, fn in {"propio": lambda s: tokenize(s, True),
                 "spaCy": lambda s: [t.text for t in nlp.make_doc(s)],
                 "NLTK": nltk_tok.tokenize}.items():
    print(f"{name:7s}", fn(ex))
""")
md("""
**Observaciones.** Las tres coinciden en lo esencial (palabras y puntuación separadas). Difieren en casos límite:
spaCy conserva `U.S.` y `1,000.50` como un token y separa `'s` y `n't`; Treebank además deja `well-known` y
`state-of-the-art` enteros; el regex propio parte `u.s.` en `u`,`s`, pega `president's` y colapsa números a `<num>`.
El regex es ~25x más rápido que spaCy y ~8x que NLTK. Como WikiText ya viene pre-tokenizado y para embeddings la
diferencia es mínima, se usa el propio. (Costo conocido: el clítico `'s` ya separado por WikiText queda como el token `s`.)
""")

md("### 1.3 Corpus de entrenamiento (texto → tokens → IDs)")
code("""
CACHE = DATA / "corpus.pkl"
if CACHE.exists():
    tokens = pickle.load(open(CACHE, "rb"))
else:
    tokens, n = [], 0
    for l in lines:
        t = tokenize(l)
        tokens.extend(t)
        if len(tokens) >= CORPUS_TOKENS:
            break
    tokens = tokens[:CORPUS_TOKENS]
    pickle.dump(tokens, open(CACHE, "wb"))
print(f"{len(tokens):,} tokens | {len(set(tokens)):,} tipos")
counts_all = collections.Counter(tokens)
""")

md("""
### 1.4 Ley de Zipf

Si la frecuencia sigue Zipf, `freq(r) ∝ 1 / r^α` con α≈1: una recta de pendiente ≈ −1 en escala log–log.
""")
code("""
freqs = np.array(sorted(counts_all.values(), reverse=True))
ranks = np.arange(1, len(freqs) + 1)
# la cola (frecuencia 1-2) se aplana por discretización; ajustar sobre el rango 10..10^4
sel = (ranks >= 10) & (ranks <= 10_000)
slope, intercept = np.polyfit(np.log(ranks[sel]), np.log(freqs[sel]), 1)
fig, ax = plt.subplots(figsize=(6, 4))
ax.loglog(ranks, freqs, label="datos")
ax.loglog(ranks[sel], np.exp(intercept) * ranks[sel] ** slope, "r--", label=f"ajuste (α={-slope:.2f})")
ax.set(xlabel="rango", ylabel="frecuencia", title="Ley de Zipf — WikiText-103 (20M tokens)")
ax.legend(); plt.tight_layout(); plt.savefig("zipf.png", dpi=150); plt.show()
print("top-10:", counts_all.most_common(10))
""")

md("### 1.5 Tamaño del vocabulario vs `min_count`")
code("""
total = sum(freqs)
rows = []
for mc in [1, 2, 5, 10, 20, 50, 100]:
    keep = freqs >= mc
    rows.append({"min_count": mc, "vocab": int(keep.sum()), "% tipos": round(100 * keep.mean(), 1),
                 "% tokens cubiertos": round(100 * freqs[keep].sum() / total, 2)})
mc_table = pd.DataFrame(rows)
mc_table
""")
md("""
La cola larga domina el vocabulario (~65% de los tipos aparece <5 veces) pero cubre una fracción
mínima de los tokens. `min_count=5` conserva ~99% del texto con una fracción del vocabulario: las palabras
más raras no tienen contextos suficientes para aprender un vector útil y solo agrandan la tabla de embeddings.
""")

md("""
### 1.6 Vocabulario, IDs y submuestreo de palabras frecuentes

Submuestreo (Mikolov et al., 2013): cada ocurrencia de la palabra `w` se conserva con probabilidad
`P(w) = min(1, sqrt(t / f(w)) + t / f(w))`, con `f` la frecuencia relativa. Las palabras muy frecuentes
(`the`, `of`) aportan poca información por ocurrencia y consumirían la mayor parte del cómputo.
""")
code("""
itos = [w for w, c in counts_all.most_common() if c >= MIN_COUNT]
stoi = {w: i for i, w in enumerate(itos)}
V = len(itos)
ids = np.array([stoi[t] for t in tokens if t in stoi], dtype=np.int32)
cnt = np.array([counts_all[w] for w in itos], dtype=np.float64)
freq = cnt / cnt.sum()
print(f"V={V:,} | corpus en IDs: {len(ids):,}")

SUBSAMPLE_T = 1e-4  # t=1e-5 es lo típico para corpus de miles de millones de tokens; 1e-4 para 20M
def keep_prob(freq, t=SUBSAMPLE_T):
    return np.minimum(1.0, np.sqrt(t / freq) + t / freq)

kp = keep_prob(freq)
exp_kept = (kp[ids]).mean()
print(f"fracción esperada de tokens conservados por época: {exp_kept:.1%}")
print(pd.DataFrame({"palabra": itos[:8], "freq": freq[:8].round(4), "P(conservar)": kp[:8].round(3)}).to_string(index=False))
""")
code("""
rng = np.random.default_rng(SEED)
sub = ids[rng.random(len(ids)) < kp[ids]]
c0 = np.bincount(ids, minlength=V); c1 = np.bincount(sub, minlength=V)
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.loglog(np.arange(1, 5001), c0[:5000], label="original")
ax.loglog(np.arange(1, 5001), c1[:5000], label="submuestreado")
ax.set(xlabel="rango", ylabel="frecuencia", title="Efecto del submuestreo (t=1e-4)")
ax.legend(); plt.tight_layout(); plt.savefig("subsampling.png", dpi=150); plt.show()
print(f"{len(ids):,} -> {len(sub):,} tokens ({len(sub)/len(ids):.1%}); 'the': {c0[stoi['the']]:,} -> {c1[stoi['the']]:,}")
""")

nb.cells = cells
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "notebook.ipynb")
print("notebook.ipynb:", len(cells), "cells")
