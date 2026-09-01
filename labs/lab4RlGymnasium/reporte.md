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
