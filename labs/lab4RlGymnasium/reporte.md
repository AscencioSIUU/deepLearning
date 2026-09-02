---
title: "Laboratorio #4 --- Fundamentos de Aprendizaje por Refuerzo y Gymnasium"
subtitle: "CC3092 Deep Learning y Sistemas Inteligentes"
author: "Ernesto Ascencio 23009"
geometry: margin=1.6cm
fontsize: 10pt
mainfont: "Arial"
---

**Repositorio:** [https://github.com/AscencioSIUU/deepLearning/tree/main/labs/lab4RlGymnasium](https://github.com/AscencioSIUU/deepLearning/tree/main/labs/lab4RlGymnasium)

## 1. Fundamentos del aprendizaje por refuerzo

**RL vs. supervisado / no supervisado.** El aprendizaje por refuerzo aprende *qué acción tomar*
para maximizar una recompensa acumulada, descubriéndola por prueba y error; no hay etiqueta de la
acción correcta (feedback *evaluativo*, no *instructivo*) y las decisiones determinan qué datos se
observan después. El supervisado generaliza desde ejemplos `(entrada, salida correcta)` dados por
un supervisor; el no supervisado busca estructura oculta en datos sin etiqueta. RL es un tercer
paradigma, con exploración vs. explotación, feedback retardado y datos no i.i.d. (Sutton & Barto,
2.ª ed., cap. 1).

**Componentes.** *Agente* (aprende y decide), *entorno* (recibe la acción y devuelve estado y
recompensa), *estado/observación* (situación actual; la observación puede ser parcial), *acción*
(decisión, discreta o continua), *recompensa* `R_t` (escalar que define el objetivo) y *política*
`π(a|s)` (estrategia estado→acción, lo que se aprende). El objetivo es maximizar el retorno
`G_t = Σ_k γ^k R_{t+k+1}` (Gymnasium docs; Sutton & Barto §3.1–3.3).

**MDP.** Formaliza la decisión secuencial bajo la propiedad de Markov (el futuro depende solo del
estado y acción actuales). Tupla `(S, A, P, R, γ)`: `S` estados; `A` acciones; `P(s'|s,a)` función
de transición (dinámica del entorno); `R(s,a,s')` recompensa esperada de la transición;
`γ∈[0,1]` factor de descuento que pondera el futuro y mantiene finito el retorno en tareas
continuas (Sutton & Barto, cap. 3).

**`V(s)` y `Q(s,a)`.** `V^π(s)=E_π[G_t|S_t=s]` es el retorno esperado empezando en `s` y siguiendo
`π`. `Q^π(s,a)=E_π[G_t|S_t=s,A_t=a]` es el retorno esperado tomando `a` en `s` y luego siguiendo
`π`. Se relacionan por `V^π(s)=Σ_a π(a|s)Q^π(s,a)`. Para la política óptima
`V*(s)=max_a Q*(s,a)` y `π*(s)=argmax_a Q*(s,a)`: conocer `Q*` basta para actuar óptimamente sin
modelo del entorno (Sutton & Barto §3.5–3.6).

**Ecuación de Bellman.** Expresa el valor de un estado en función del de sus sucesores: "valor
aquí = recompensa inmediata esperada + valor descontado del estado al que llego". Es una
condición de consistencia recursiva:
`V^π(s)=Σ_a π(a|s)Σ_{s'} P(s'|s,a)[R(s,a,s')+γV^π(s')]`; la versión de optimalidad reemplaza
`Σ_a π(a|s)` por `max_a`. Su rol: convierte una suma infinita sobre trayectorias en un sistema de
ecuaciones (una por estado); programación dinámica, Monte Carlo y diferencia temporal son formas
de resolverla o aproximarla (Sutton & Barto, cap. 3–6).

**Exploración vs. explotación.** Explotar da recompensa ahora; explorar puede revelar algo mejor,
a costa de recompensa a corto plazo. *ε-greedy*: acción greedy con probabilidad `1−ε`, aleatoria
uniforme con probabilidad `ε` (que suele decaer); explora "a ciegas". *Softmax/Boltzmann*:
`π(a|s)∝exp(Q(s,a)/τ)`, con temperatura `τ` que va de casi-uniforme (`τ` alta) a casi-greedy
(`τ` baja); explora proporcionalmente al valor estimado (Sutton & Barto, cap. 2).

**Familias de métodos.** *Episódicas* (con estado terminal, retorno finito) vs. *continuas* (sin
fin, requieren `γ<1`); en Gymnasium: `terminated` vs. `truncated`. *On-policy* (evalúa y mejora la
misma política que genera los datos, p. ej. SARSA) vs. *off-policy* (política objetivo distinta de
la de comportamiento, p. ej. Q-Learning). *Model-based* (aprende/usa `P` y `R` y planifica; más
eficiente en muestras, sensible a error de modelo) vs. *model-free* (aprende valores o política de
la experiencia directa; más simple, necesita más interacción) (Sutton & Barto §3.3–3.4, 5.5, 6.4, cap. 8).

**Q-Learning** (Watkins & Dayan, 1992). Método model-free, off-policy, de diferencia temporal:
`Q(s,a) ← Q(s,a) + α[ r + γ·max_{a'} Q(s',a') − Q(s,a) ]`. El corchete es el *TD error*: distancia
entre la estimación actual y un objetivo mejor informado (recompensa observada + mejor valor
futuro estimado). Es off-policy por el `max_{a'}` aunque `a` se haya elegido explorando. `α`
(tasa de aprendizaje) es el tamaño de paso hacia ese objetivo: alta → rápido pero oscila; baja →
lento pero estable. `γ` pondera el término futuro `max_{a'}Q(s',a')`: `γ→0` miope, `γ→1`
previsor pero de propagación más lenta.

## 2. La librería Gymnasium

**Qué es y relación con OpenAI Gym.** Gymnasium define una *API estándar* para entornos de RL y
provee entornos de referencia que la implementan. Resuelve la *interoperabilidad*: con una
interfaz común (`reset`, `step`, espacios autodescritos) un mismo agente corre en cualquier
entorno compatible y los resultados son comparables. Es el fork mantenido de OpenAI Gym: OpenAI
dejó de mantener Gym en 2021 y la Farama Foundation continuó el desarrollo como Gymnasium en 2022.
Reemplazo directo (`import gymnasium as gym`); el cambio de API principal es que `step()` devuelve
cinco valores, separando `terminated` de `truncated` (antes un único `done`), y `reset()` devuelve
`(obs, info)` y acepta `seed=`. (Gymnasium docs; Farama Foundation, 2022).

**Estructura de un `Env`.** `reset(seed, options) → (observation, info)` inicia un episodio.
`step(action) → (observation, reward, terminated, truncated, info)` avanza un paso. `render()`
produce una visualización según `render_mode`. `close()` libera recursos. La tupla de `step()`:
*observation* (elemento del `observation_space`); *reward* (`float`); *terminated* (`True` en
estado terminal del MDP — meta, poste caído; no hay valor futuro); *truncated* (`True` por corte
externo, típicamente el límite de pasos de `TimeLimit`; el estado no es terminal); *info* (`dict`
de diagnóstico, no para aprender). El episodio acaba con `terminated or truncated`.

**Spaces.** `Discrete(n, start=0)`: un entero en un rango finito — un valor categórico; acciones
izquierda/derecha, estados enumerados de una grilla (acción de CartPole `Discrete(2)`, obs de
FrozenLake `Discrete(16)`). `Box(low, high, shape, dtype)`: tensor de reales acotado elemento a
elemento (`±inf` permitido) — magnitudes continuas: observaciones físicas, imágenes
`Box(0,255,(H,W,3),uint8)`, torques (obs de CartPole `Box(4,)`). `MultiDiscrete(nvec)`: vector de
enteros, componente `i` en `{0..nvec[i]−1}` — producto de varios `Discrete` para acciones con
varias dimensiones categóricas simultáneas (gamepad `[5,2,2]`, acción factorizada de Atari).
También `MultiBinary`, `Tuple`, `Dict`.

**Catálogo — 4 entornos.**

| Entorno (familia) | Objetivo | Observación | Acción |
|---|---|---|---|
| CartPole-v1 (Classic Control) | equilibrar un poste sobre un carro; +1/paso hasta 500 | `Box(4,)`: pos. y vel. del carro, ángulo y vel. angular del poste | `Discrete(2)`: izq / der |
| MountainCar-v0 (Classic Control) | subir un carro sin potencia a la cima acumulando impulso; −1/paso (máx. 200) | `Box(2,)`: posición ∈[−1.2,0.6], velocidad ∈[−0.07,0.07] | `Discrete(3)`: acel. izq / nada / acel. der |
| FrozenLake-v1 (Toy Text) | cruzar una grilla 4×4 helada del inicio a la meta sin caer en un agujero; +1 solo al llegar | `Discrete(16)`: casilla actual | `Discrete(4)`: izq/abajo/der/arriba (resbala si `is_slippery`) |
| LunarLander-v3 (Box2D) | aterrizar un módulo lunar sobre la plataforma; premia acercarse/aterrizar, penaliza chocar y gastar combustible | `Box(8,)`: pos. (x,y), vel. (vx,vy), ángulo, vel. angular, 2 flags de contacto | `Discrete(4)` motores (o `Box(2,)` continuo) |

**Wrappers.** Un wrapper envuelve un `Env` y modifica su comportamiento sin tocar el código del
entorno, exponiendo la misma API (patrón decorador); se apilan — `gym.make("CartPole-v1")` ya
devuelve `TimeLimit(OrderEnforcing(PassiveEnvChecker(CartPoleEnv)))`. Ejemplos: `TimeLimit` pone
`truncated=True` al llegar a `max_episode_steps`; `RecordVideo` graba episodios a `.mp4`;
`RecordEpisodeStatistics` agrega retorno y longitud a `info`; `NormalizeObservation` /
`NormalizeReward` estandarizan con media y varianza corrientes (además de `RescaleAction`,
`ClipAction`, `FrameStackObservation`, `GrayscaleObservation`).
