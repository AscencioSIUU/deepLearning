# Space Invaders — reglas del proyecto

Esta carpeta tiene dos cosas: el **Lab #5**, ya entregado y congelado en `lab5/`, y el
**Proyecto 2**, que es el trabajo activo en `agent/`.

## Reglas del Proyecto 2

El enunciado permite usar librerías: *"Puede implementarse desde cero o utilizar una
librería, siempre documentando y explicando las decisiones de diseño."* Se usa
stable-baselines3. Por eso las reglas de aquí son de **método**, no de tecnología.

1. **Una iteración cambia una sola cosa.** Si se cambian dos a la vez, la tabla de resultados
   que pide la sección 2.3 del enunciado deja de decir nada causal.
2. **Registrar antes de avanzar.** Toda iteración se evalúa y se añade a
   `agent/runs/results.csv` antes de empezar la siguiente. Ese CSV es la tabla del reporte.
3. **`agent/env.py` es la única fuente del preprocesamiento.** `train.py` y `evaluate.py` lo
   llaman; nunca redefinen wrappers. El enunciado exige que evaluación y entrenamiento usen
   el mismo preprocesamiento.
4. **No ajustar mirando la métrica de evaluación.** Los 5 episodios greedy con semillas 0-4
   se corren sólo al cerrar una iteración.
5. **Añadir un assert al `--check` cuando se encuentre un bug silencioso.** Ya pasó con el
   frameskip duplicado: el agente entrenaba igual, sólo que peor, sin ningún error.

## Invariantes que no se tocan

- `frameskip=1` en el entorno base. `ALE/SpaceInvaders-v5` trae 4 de fábrica y el
  `AtariWrapper` aplica otro 4: el skip efectivo sería 16. Verificado por `--check`.
- Evaluación con `clip_reward=False` y `terminal_on_life_loss=False`. La métrica es el score
  real de una partida completa de 3 vidas.
- `resolve_device()` en vez de `device="auto"`. SB3 sólo detecta CUDA; en Apple Silicon hay
  que pedir MPS explícitamente y vale 2.4x de velocidad.
- `buffer_size` no sube de 100 000 sin activar `optimize_memory_usage`: 1 M de transiciones
  con frames apilados son ~28 GB y la máquina tiene 24 GB.
- `env.close()` siempre. `VecVideoRecorder` escribe el mp4 al cerrar.

## `lab5/` está congelado

Ya se entregó. No se modifica ni se refactoriza. `lab5/ale_utils.py` vive ahí porque el
notebook hace `import ale_utils`. `agent/` no lo importa: hereda las decisiones (registrar
ALE, cerrar el entorno, sembrar el espacio de acciones, grabar con los wrappers de
Gymnasium), no el código, porque SB3 trabaja con `VecEnv` y `model.predict()`.

## Verificación

```sh
python -m agent.env --check                  # contrato de los entornos
python -m agent.train    --config i1_smoke   # ~4 min
python -m agent.evaluate --config i1_smoke --video
```
