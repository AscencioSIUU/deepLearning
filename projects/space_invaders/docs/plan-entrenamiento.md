# Plan de entrenamiento — Proyecto 2

Documento vivo: se actualiza al cerrar cada iteración. Alimenta la sección 2.3 del trabajo
escrito ("Resultados de iteraciones").

## Qué pide el proyecto

Entrenar un agente de Reinforcement Learning que juegue `ALE/SpaceInvaders-v5`. **No es
entrenamiento asistido**: no hay dataset etiquetado ni demostraciones humanas. El agente
juega, recibe como recompensa el incremento de score que ALE lee de la RAM, y ajusta su
política. Nadie le dice qué acción es correcta.

- Presentación: 21 de octubre de 2026. Entrega escrita: 25 de octubre de 2026.
- Rúbrica: 8 puntos = 5 trabajo escrito + 3 ranking de la clase.
- Evaluación: 5 episodios con política greedy.

### La métrica está contradicha en el enunciado

La tabla de "Información clave" dice *"Métrica objetivo: Puntaje promedio"*. La sección 3
dice *"se tomará el mayor de esos 5 episodios"*. `evaluate.py` registra **las dos**, más
mínimo y desviación. El máximo premia la varianza; la media premia la consistencia. Si hay
que elegir un agente entre dos, este es el criterio a vigilar.

## Reglas de método

1. Una iteración cambia **una sola cosa** respecto a la anterior. Si se cambian dos, la tabla
   de resultados deja de decir nada causal.
2. Toda iteración se evalúa y se registra en `agent/runs/results.csv` **antes** de empezar la
   siguiente.
3. El preprocesamiento de evaluación es idéntico al de entrenamiento, salvo dos diferencias
   deliberadas (recompensa sin recortar, episodio de 3 vidas). `agent/env.py` es la única
   fuente de esa configuración.
4. Los 5 episodios de evaluación usan semillas fijas 0-4 y se corren sólo al cerrar una
   iteración. No se ajustan hiperparámetros mirando ese número.

## El ladder

| Iter | Cambio respecto a la anterior | Steps | Qué responde | Estado |
|---|---|---|---|---|
| I0 | agente aleatorio del Lab 5 | — | el cero de la escala | hecho: media 123.5, máx 235 |
| I1 | `i1_smoke`: DQN de SB3, corrida corta | 100 k | ¿el circuito entrena → guarda → carga → evalúa → graba? | hecho: media 223.0, máx 510 |
| I2 | `i2_dqn_1m`: hiperparámetros de Atari del rl-zoo | 1 M | ¿cuánto vale la receta conocida? | pendiente (~35 min) |
| I3 | `i3_dqn_5m`: mismos hiperparámetros, 5x pasos | 5 M | ¿dónde está la curva de retorno vs. presupuesto? | pendiente (~3 h) |
| I4 | `i4_qrdqn_5m`: QRDQN en vez de DQN | 5 M | ¿el algoritmo distribucional sube el score? | pendiente (~3 h) |
| I5 | `i5_ppo_10m`: PPO con 12 entornos paralelos | 10 M | ¿mejor score por hora de reloj? | pendiente |
| I6 | el ganador de I3-I5, entrenado largo + ajuste de exploración/lr | 10-20 M | agente candidato | pendiente |
| I7 | selección final, pesos congelados + video | — | entregable | pendiente |

El orden es deliberado: I1-I2 cuestan minutos y validan todo antes de comprometer las noches
de I3-I6. Con 36 días hay margen de sobra.

## Decisiones tomadas, con su evidencia

### MPS gana a CPU por 2.4x

Medido con 20 000 pasos de DQN en el M5 Pro:

| Device | Tiempo | FPS |
|---|---|---|
| cpu | 130.4 s | 153 |
| mps | 54.4 s | 368 |

SB3 con `device="auto"` **no** detecta MPS: sólo busca CUDA y cae a CPU. Hay que pedirlo
explícitamente, y de eso se encarga `resolve_device()` en `agent/configs.py`. A 470 FPS
(medido en la corrida real de I1), 1 M de pasos son ~35 min y 5 M son ~3 h.

### El frameskip estaba duplicado

`ALE/SpaceInvaders-v5` trae `frameskip=4` de fábrica, y el `AtariWrapper` de SB3 aplica otro
4 encima: skip efectivo de **16**. El agente actuaba 4 veces menos de lo debido y no podía
reaccionar. Se corrigió creando el entorno base con `frameskip=1`, delegando el skip al
wrapper, que además hace max-pool de los 2 últimos frames.

Efecto medido en I1, mismo entrenamiento de 100 k pasos:

| | Pasos por episodio | Score medio | Score máx |
|---|---|---|---|
| skip 16 (roto) | 83-240 | 211.0 | 410 |
| skip 4 (correcto) | 255-855 | 223.0 | 510 |

Es un bug silencioso: el agente entrena igual, sólo que peor, y no da ningún error.
`python -m agent.env --check` ahora lo verifica con un assert.

### El buffer de repetición cabe en RAM, pero justo

Un replay buffer de 1 M de transiciones con frames apilados ocuparía ~28 GB y la máquina
tiene 24 GB. Por eso `buffer_size=100_000`, que es además lo que usa el rl-zoo para Atari.
Para subirlo habría que activar `optimize_memory_usage=True`.

### Recompensa recortada al entrenar, sin recortar al evaluar

Entrenamiento: `clip_reward=True` y `terminal_on_life_loss=True`. El recorte a {-1, 0, +1}
permite usar hiperparámetros estándar; tratar cada vida como fin de episodio da una señal de
fracaso más inmediata.

Evaluación: ambos en `False`. La métrica del ranking es el score real de una partida
completa de 3 vidas. Evaluar con recorte mediría "número de aciertos", no puntaje.

## Qué queda por decidir

- Si I5 (PPO) gana a I3/I4 en score por hora de reloj, el agente final sale de ahí.
- Si el score se estanca, las palancas en orden de coste: más pasos, luego `learning_rate`
  con schedule, luego `exploration_final_eps` más bajo, y por último `buffer_size` mayor con
  `optimize_memory_usage`.
