# PPO + NatureCNN + annealing lineal de lr/clip_range

Modelo final usado en la competencia. No es DQN: es PPO (on-policy, actor-crítico), por lo
que no aplica replay buffer y por lo tanto no hay PER ni n-step returns — esas técnicas son
propias de la familia DQN. La composición real es la de arriba.

## Glosario — qué es cada pieza

| Término | Qué es |
|---|---|
| **PPO** (Proximal Policy Optimization) | Algoritmo de RL on-policy (Schulman et al., 2017). Alterna recolectar un lote corto de experiencia con la política actual y actualizarla por gradiente, limitando (`clip_range`) cuánto puede cambiar en cada update para no desestabilizar el entrenamiento. |
| **Stable-Baselines3 (SB3)** | Librería de Python sobre PyTorch con implementaciones de referencia de algoritmos de RL (PPO, DQN, A2C, entre otros). No es "stable-baseline2": SB3 es la reescritura en PyTorch de la Stable-Baselines original (TensorFlow). Aporta el algoritmo, el bucle de entrenamiento, callbacks de checkpoint y guardado/carga de modelos — se usó en vez de implementar PPO desde cero, como permite el enunciado. |
| **ActorCriticCnnPolicy** | Clase de política de SB3 para algoritmos actor-crítico (PPO, A2C) sobre observaciones de imagen. Empaqueta en un solo objeto: el extractor de features convolucional (NatureCNN), la cabeza de política (`action_net`, elige la acción) y la cabeza de valor (`value_net`, estima qué tan buena es la situación), y expone `.predict()` para usarlas juntas. |
| **Pesos del modelo** | `model.zip` en esta misma carpeta: los parámetros entrenados de la `ActorCriticCnnPolicy` (~1.69M, ver tabla de arquitectura abajo) más el espacio de observación/acción y los hiperparámetros del algoritmo. Se cargan con `PPO.load("model.zip")` — exactamente lo que hace `agent/evaluate.py` antes de correr los episodios. |

## Componentes

| Componente | Qué es | Por qué aquí |
|---|---|---|
| **PPO** | Algoritmo on-policy: recolecta un lote corto de experiencia con la política actual y la actualiza limitando (`clip_range`) cuánto puede moverse en cada paso, para no destruir lo ya aprendido. | Estable con redes convolucionales grandes y paraleliza directo en `n_envs` entornos — sin replay buffer no hay límite de RAM por transiciones acumuladas, a diferencia de DQN. |
| **NatureCNN** | 3 capas convolucionales + 1 densa (la arquitectura de DeepMind 2015 para Atari) que convierte 4 frames 84×84 en un vector de 512 features. | Extrae movimiento y posición de sprites a partir de píxeles crudos; es el extractor estándar validado en decenas de juegos Atari, no hace falta diseñar uno propio. |
| **Annealing lineal de lr/clip_range** | `learning_rate` y `clip_range` decaen linealmente de su valor inicial a 0 a lo largo del entrenamiento, en vez de quedar constantes. | En corridas largas (80M pasos) un `clip_range` fijo sigue permitiendo updates agresivos al final, cuando la política ya casi convergió (visto en i5 sin annealing: `clip_fraction` 0.263 al 83% del entrenamiento). Decaerlo estabiliza el cierre. |

Los tres encajan porque atacan capas distintas del mismo problema: NatureCNN resuelve *qué
observa* el agente, PPO resuelve *cómo actualiza* la política de forma segura, y el annealing
resuelve *cuánto* confiar en cada actualización a medida que avanza el entrenamiento.

## El modelo, en conjunto

Una red convolucional (NatureCNN) mira 4 frames apilados y produce un vector de 512
features, compartido por dos cabezas lineales: una elige la acción (política) y otra estima
qué tan buena es la situación (valor). PPO entrena ambas a la vez sobre lotes cortos de
experiencia recolectados en 12 copias del entorno en paralelo, con el tamaño de sus updates
decayendo a 0 conforme se acerca el final de los 80M pasos.

| Aspecto | Valor |
|---|---|
| Entrada | 4×84×84 uint8 (frames apilados, escala de grises) |
| Salida | 6 acciones discretas |
| Parámetros | ~1.69M |
| Entornos paralelos | 12 |
| Pasos de entrenamiento | 80,000,000 |
| Tiempo de entrenamiento | 20.5 h (MPS, ~1086 FPS) |

## Comando de ejecución

```bash
cd /Users/nesstor/Documents/git_u/2026/deepLearning/projects/space_invaders
/Users/nesstor/Documents/git_u/2026/deepLearning/.venv/bin/python -m agent.evaluate --config i12_ppo_80m --video
```

Carga `model.zip` de esta carpeta, corre 5 episodios con política greedy (semillas 0-4),
imprime los scores y el máximo (el que cuenta para el ranking), graba el video del episodio
semilla 0 en `videos/i12/` y añade una fila a `agent/runs/results.csv`.

## Qué se usó en el modelo

**Algoritmo**: PPO (`stable-baselines3`), política `ActorCriticCnnPolicy`.

**Preprocesamiento** (`agent/env.py`, idéntico en entrenamiento y evaluación salvo
recompensa/vidas): `ALE/SpaceInvaders-v5`, `frameskip=1` en el entorno base + `frame_skip=4`
del `AtariWrapper` (skip efectivo 4, no 16), max-pool de los 2 últimos frames, resize a 84×84
en escala de grises, 4 frames apilados. Acción: `Discrete(6)`, subconjunto mínimo, no
`full_action_space`.

**Arquitectura** (NatureCNN, la CNN estándar de DeepMind para Atari — verificada cargando el
modelo entrenado):

| Capa | Detalle |
|---|---|
| Conv1 | 4→32 canales, kernel 8×8, stride 4, ReLU |
| Conv2 | 32→64, kernel 4×4, stride 2, ReLU |
| Conv3 | 64→64, kernel 3×3, stride 1, ReLU |
| Flatten + Linear | 3136 → 512, ReLU |
| Cabezas | `action_net`: 512→6 (logits de política) · `value_net`: 512→1 (valor) |

Actor y crítico comparten la misma NatureCNN como extractor de features. ~1.69M parámetros.

**Hiperparámetros de entrenamiento** (`PPO_ATARI_TUNED` en `agent/configs.py`, receta
rl-zoo para Atari): `learning_rate` y `clip_range` con decaimiento lineal a 0
(2.5e-4→0, 0.1→0), `n_steps=128`, `batch_size=256`, `n_epochs=4`, `vf_coef=0.5`,
`ent_coef=0.01`, γ por defecto de SB3 (0.99). 12 entornos paralelos, 80,000,000 pasos totales
(73,652 s ≈ 20.5 h en MPS, ~1086 FPS).

**Resultado de evaluación** (5 episodios greedy, semillas 0-4, `clip_reward=False`,
`terminal_on_life_loss=False` — score real de una partida completa de 3 vidas):

| media | máximo (ranking) | mínimo | std |
|---|---|---|---|
| 36377.0 | 52575.0 | 7770.0 | 16048.9 |

## Video

[i12_ppo_80m_seed0-step-0-to-step-30000.mp4](../../../final/i12_ppo_80m/i12_ppo_80m_seed0-step-0-to-step-30000.mp4)
— episodio completo, semilla 0, política greedy. Misma config de evaluación que la competencia.

## Comparación entre iteraciones

Historial completo de `agent/runs/results.csv` (score máximo = el que cuenta para el ranking):

| Iteración | Algoritmo | Pasos | Score medio | Score máx | Cambio clave |
|---|---|---|---|---|---|
| i1_smoke | DQN | 100 K | 223 | 510 | DQN por defecto de SB3, valida el pipeline |
| i2_dqn_1m | DQN | 1 M | 539 | 1,000 | hiperparámetros de Atari del rl-zoo |
| i3_dqn_5m | DQN | 5 M | 1,132 | 1,325 | 5x más pasos que i2 |
| i4_qrdqn_5m | QRDQN | 5 M | 972 | 1,775 | distribucional en vez de DQN |
| i5_ppo_10m | PPO | 10 M | 873 | 1,425 | on-policy, 12 entornos paralelos |
| i6_ppo_tuned | PPO | 10 M | 910 | 1,630 | annealing lineal de lr/clip_range |
| i7_ppo_40m | PPO | 40 M | 5,507 | 13,125 | 4x más pasos que i6 |
| i8_dqn_10m | DQN | 10 M | 795 | 1,080 | 2x más pasos que i3 |
| i9_qrdqn_q50 | QRDQN | 5 M | 814 | 1,175 | menos cuantiles (200→50) que i4 |
| i10_dqn_target10k | DQN | 5 M | 1,034 | 1,180 | target network menos frecuente |
| i11_dqn_target10k_10m | DQN | 10 M | 916 | 1,230 | 2x más pasos que i10 |
| **i12_ppo_80m** | **PPO** | **80 M** | **36,377** | **52,575** | **2x más pasos que i7 — modelo final** |

PPO con presupuesto grande (i7, i12) saca una diferencia de orden de magnitud sobre todo lo
demás; ni el algoritmo (DQN/QRDQN) ni los ajustes finos de hiperparámetros movieron la aguja
tanto como simplemente entrenar PPO por más tiempo.

Nota: al momento de escribir esto `results.csv` tiene 4 filas duplicadas de `i12_ppo_80m`
(mismos valores, evaluaciones repetidas el mismo día) — conviene dejar solo una antes de
entregar el CSV como tabla del reporte.
