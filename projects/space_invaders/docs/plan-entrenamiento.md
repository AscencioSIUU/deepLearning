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

## Qué estamos maximizando, en concreto

No hay distancia ni progreso espacial: la nave solo se mueve en horizontal. El score es una
cosa y solo una: **cuántos aliens derriba el agente antes de perder sus 3 vidas.** Cada oleada
son 36 invasores; con la tabla de puntos estándar del Atari 2600 (5/10/15/20/25/30 por fila)
una oleada completa vale ~630 puntos. Traducción de los resultados: i3 (1132 de media) limpia
la primera oleada y buena parte de la segunda. Subir el score exige dos habilidades: **apuntar
mejor** y **esquivar más tiempo** — la mitad de morir viene de no esquivar los disparos
enemigos, por eso importa poder verlos en el video (ver sección de videos, `VIDEO_SCALE`).

### La métrica está contradicha en el enunciado

La tabla de "Información clave" dice *"Métrica objetivo: Puntaje promedio"*. La sección 3
dice *"se tomará el mayor de esos 5 episodios"*. `evaluate.py` registra **las dos**, más
mínimo, desviación y `max_esperado`. La media premia consistencia (5 de los 8 puntos de la
rúbrica, tabla del reporte); el máximo premia varianza (3 de los 8, ranking de la clase). La
mediana no aparece en el enunciado y no se usa: con 5 datos tira los extremos.

**Cómo se reconcilian.** Para N=5 muestras, el máximo esperado se aproxima con
`media + 1.16 * std` — no es un objetivo aparte, es la media más una prima por varianza.
De ahí la prioridad para todo el ladder:

1. **Subir la media** — sube también el máximo esperado. Objetivo de i6-i9.
2. **A igualdad de media (diferencia por debajo del ruido: SE de 90-210 con 5 episodios),
   preferir más varianza** — es lo único que ataca directamente el ranking.
3. **Nunca sacrificar media a propósito por varianza** — sería apostar los 5 puntos del
   reporte por los 3 del ranking.

`results.csv` trae la columna `max_esperado` calculada para tener la prima a la vista.

**Selección del modelo final:** la std con 5 muestras tiene ~35% de error, así que rankear por
`max_esperado` con 5 episodios no es confiable para *elegir* entre los 2 mejores candidatos.
Para esa decisión (no para hiperparámetros, eso seguiría violando la regla 4) se evalúan con
`--episodios 20`, que no se mezcla con el CSV oficial (`evaluate.py` lo deja fuera a propósito
cuando el número de episodios no es 5). La cifra que se reporta sigue siendo la de 0-4.

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

## El ladder — I0 a I5 (cerrado)

| Iter | Cambio respecto a la anterior | Steps | Media | Máx | Std | max_esperado | Tiempo |
|---|---|---|---|---|---|---|---|
| I0 | agente aleatorio del Lab 5 | — | 123.5 | 235 | — | — | — |
| I1 | `i1_smoke`: DQN de SB3, corrida corta | 100k | 223.0 | 510 | 198.1 | 452.8 | — |
| I2 | `i2_dqn_1m`: hiperparámetros de Atari del rl-zoo | 1M | 539.0 | 1000 | 250.3 | 829.3 | 43.5 min |
| I3 | `i3_dqn_5m`: mismos hiperparámetros, 5x pasos | 5M | **1132.0** | 1325 | 201.2 | 1365.4 | 228.5 min |
| I4 | `i4_qrdqn_5m`: QRDQN en vez de DQN | 5M | 972.0 | **1775** | 472.2 | **1519.8** | 670 min |
| I5 | `i5_ppo_10m`: PPO, 12 entornos paralelos | 10M | 873.0 | 1425 | 316.5 | 1240.1 | 139.1 min |

### Qué funcionó y qué no

- **i1→i2 (+142%)**: la receta conocida del rl-zoo vale más que cualquier ajuste propio.
- **i2→i3 (+110%)**: el lever confirmado más fuerte es "más pasos". Ojo — no fue un cambio
  puro: `exploration_fraction=0.1` es fracción del *total*, así que i3 también estiró el
  annealing de epsilon de 100k a 500k pasos. El +110% mezcla presupuesto con exploración.
- **i3→i4 (−14% en media)**: QRDQN pierde en media y es 3x más lento (124 vs 365 FPS) —el
  peor ROI del ladder— pero gana claramente en máximo (1775 vs 1325): es varianza, no mejora.
- **i3→i5 (−23% en media)**: PPO tampoco corrió la receta completa del zoo — le faltaban los
  schedules lineales de `learning_rate`/`clip_range` (estaban constantes) y usa `n_envs=12` en
  vez de 8 (33% menos pasos de gradiente: `n_updates 26040 = 6510×4` en el log). Pero es 3.3x
  más barato por paso (1198 FPS) y su curva de `ep_rew_mean` **seguía subiendo** al terminar
  (804→888, cerró en 861), mientras la de DQN en i3 ya estaba plana (703→708 desde 4.26M).
- **Por máximo el orden se invierte**: i4 > i5 > i3. Con SE de 90-210 (5 episodios), i3 e i4
  **no son estadísticamente separables** en media.

Aviso: `ep_rew_mean` de entrenamiento no es comparable entre familias — DQN entrena con
eps=0.01 (708 entrenamiento → 1132 evaluación greedy, +60%), PPO ya es casi determinista al
final (861 → 873, +1%). Sirve para ver si una misma rama sigue subiendo, no para rankear
ramas entre sí. Contexto adicional: el entorno usa sticky actions
(`repeat_action_probability=0.25`, default de v5), que hunde los scores respecto a los
números publicados con `NoFrameskip-v4`; no se toca, la competencia usa el mismo entorno.

## El ladder — I6 a I10 (plan)

Cada iteración cambia una sola cosa respecto a un predecesor nombrado (salvo I10, marcada
como fuera de la escalera causal). La columna "apunta a" conecta con la prioridad de arriba
(media primero, varianza en caso de empate).

| Iter | Config | Cambio único vs. | Apunta a | Coste | Estado |
|---|---|---|---|---|---|
| I6 | `i6_ppo_tuned` | i5: `lr`/`clip_range` → schedule lineal | media | ~2.3 h (real: 5.9 h, ver nota de sueño) | **hecho: media 910.0, máx 1630, std 443.9, max_esperado 1425.0** |
| I8 | `i8_dqn_10m` | i3: 2x pasos (10M) | media — ¿el plateau de DQN es real al presupuesto de la receta? | 466.9 min | **hecho: media 795.0 (peor que i3), ver hallazgo abajo** |
| I9 | `i9_qrdqn_q50` | i4: `n_quantiles` 200→50 | varianza — QRDQN a velocidad viable | 220.9 min | **hecho: media 814.0, máx 1175 — peor que i4 en todo, ver abajo** |
| I10 | `i10_dqn_target10k` | i3: `target_update_interval` 1000→10000 | media/estabilidad — ataca la oscilación encontrada en i8, no solo sus síntomas | 242.4 min | **hecho: media 1034.0, máx 1180, std 259.8 — ver hallazgo abajo** |
| I11 | `i11_dqn_target10k_10m` | i10: 2x pasos (10M) | media | 494.0 min | **hecho: media 916.0 (5 ep) / 962.8 (20 ep) — empatada con todo, ver hallazgo abajo** |
| I7 | `i7_ppo_40m` | i6: 4x pasos (40M) | media y máximo | 665.4 min | **hecho: media 5507.0 (5 ep) / 5059.0 (20 ep), máx 13125 — 5.7x el mejor resultado anterior, ver hallazgo abajo** |
| I12 | `i12_ppo_80m` | i7: 2x pasos (80M) | media/máximo | 1227.5 min | **hecho: media 36377.0 (5 ep) / 26105.2 (20 ep), máx 52605 — otro salto grande, ver hallazgo abajo** |
| I13 | `i13_ppo_160m` | i12: 2x pasos (160M) | media/máximo — el patrón de duplicación sigue dando saltos grandes | ~40 h | **se lanza ahora** |
| I14 | corrida de entrega | — (no aísla causa) | el modelo que se entrega — candidato líder hoy: i12 (media 26105.2 a 20 ep) | overnight | pendiente |

### i10: arregló la oscilación, empató en media — y por qué igual vale la pena escalarlo

`i10_dqn_target10k` (mismos hiperparámetros que i3, `target_update_interval` 1000→10000) dio
media 1034.0 — **estadísticamente empatado con i3 (1132.0)**, no una mejora clara en el
número de 5 episodios. Pero dos señales dicen que el cambio sí funcionó:

1. **La curva de entrenamiento subió limpio, sin la oscilación de i8**: de step 1M a 5M pasa
   de 576 a 893 de forma consistente (pico 946 en 4.75M, cierra estable ~890-896 en el último
   medio millón). Nada parecido al 1023→513→729 en ventanas de 500k que se vio en los
   checkpoints de i8. Actualizar el objetivo cada 10k pasos en vez de 1k le da tiempo al
   bootstrap a asentarse antes de moverse otra vez — la hipótesis de la tríada mortal se
   confirma indirectamente.
2. **El techo de entrenamiento subió +26%** (893 vs 706-708 de i3) aunque el greedy de 5
   episodios no lo refleje — el ruido de 5 episodios (std 260, SE~115) es del tamaño de la
   diferencia observada, así que "empatado en eval" y "mejor en entrenamiento" no se
   contradicen, son medidas con precisión distinta.

Con la inestabilidad resuelta, vale la pena repetir el intento de i8 (más presupuesto) sobre
esta base estable — es la comparación limpia que i8 no pudo hacer.

### i11: repitió i8 sobre la base estable — y reveló que el problema real era otro

`i11_dqn_target10k_10m` dio 916.0 de media — **peor que i10 (1034) con el doble de pasos**,
a pesar de que su curva de entrenamiento subió más (946-999 vs 893-946 de i10). Con esto, las
cuatro variantes de DQN (i3, i8, i10, i11) daban resultados de eval entre 795 y 1132 sin
ningún patrón limpio con presupuesto o estabilidad. Eso encendió una alarma: con 5 episodios
y std de 190-260, el **SE es de 85-115** — del mismo tamaño que las diferencias que se estaban
persiguiendo desde i8. Antes de lanzar una quinta variante de DQN a ciegas, tocaba medir con
suficiente precisión para saber si había señal.

### La comparación de 20 episodios — el hallazgo que reordena todo el proyecto

Se evaluaron los candidatos con `--episodios 20` (SE~70-100, no se registra en el CSV, ver
Parte 0):

| Config | Algo | Media | Máx | Std | max_esperado |
|---|---|---|---|---|---|
| i3_dqn_5m | DQN | 939.0 | 1400 | 342.2 | 1335.9 |
| i10_dqn_target10k | DQN | 956.8 | 1360 | 305.3 | 1310.9 |
| i11_dqn_target10k_10m | DQN | 962.8 | 1640 | 350.0 | 1368.8 |
| i4_qrdqn_5m | QRDQN | 940.8 | 1775 | 396.3 | 1400.5 |
| **i6_ppo_tuned** | PPO | **964.0** | **1960** | 446.7 | **1482.2** |

**Las cinco configs — tres familias de algoritmos distintas — convergen a la misma media
(939-964, un rango de 25 puntos, dentro del ruido).** Toda la campaña i8→i11 persiguió ruido
de 5 episodios: ninguna variante de DQN mejoró realmente sobre i3, `target_update_interval`
incluido. Hay un techo real de ~950 de media para esta familia de recetas del rl-zoo sobre
este entorno (con sticky actions y episodios de 3 vidas) que ningún cambio probado rompió.

**Pero el máximo sí tiene recorrido real, y ahí está la palanca que queda.** Creció con más
pasos en DQN (i3→i11: 1400→1640, el mismo patrón que ya se había visto con 5 episodios) y
**PPO (i6) lidera todo el tablero en máximo (1960) y max_esperado (1482.2)**, con el costo por
paso más bajo de los tres (PPO ~1200 FPS vs QRDQN ~130-380 vs DQN ~340-450). Con la media
tapada, subir el máximo es la única palanca que le queda al ranking (3 de los 8 puntos) sin
sacrificar los otros 5. Esto reordena la prioridad de la Parte 0 para lo que queda del
proyecto: **entre candidatos empatados en media, maximizar max_esperado — que es justo la
regla 2 que ya estaba escrita, solo que ahora hay evidencia de que la regla 1 (subir media)
ya tocó su techo.**

Con esto, **i7 (PPO 40M) se reactiva** — no para "cerrar el hueco con DQN en media" (ya no
hay hueco, hay empate), sino para ver si más pasos siguen empujando el máximo de PPO como
lo hicieron con DQN, aprovechando que es el algoritmo más barato del ladder.

### i7: el techo de ~950 no era un techo — era el presupuesto

`i7_ppo_40m` (mismos hiperparámetros que i6, 4x pasos) terminó en 665.4 min (1002 FPS,
sobrevivió corriendo con batería sin cargador conectado, ver nota de infraestructura abajo) y
dio **media 5507.0, máximo 13125.0** con 5 episodios — **5.7x la media y 6.7x el máximo de
cualquier config anterior.** La curva de entrenamiento (`ep_rew_mean`) confirma que no es un
accidente de 5 episodios: sube de forma sostenida y casi monótona los 40M pasos completos, sin
plateau, de 151 al empezar a un pico de 5320 en el paso 39.6M — seguía subiendo al cerrar.

Esto **revisa la conclusión de la comparación de 20 episodios**: no hay un techo de ~950 de
media para el entorno — hay un techo de ~950 **al presupuesto de 5-10M pasos** que probaron
i3, i4, i6, i10, i11. PPO simplemente necesitaba mucho más presupuesto para despegar; con 4x
los pasos de i6 no solo lo alcanzó, lo dejó atrás por un orden de magnitud. Como la varianza
del eval de 5 episodios también fue enorme (std 3970, rango 2675-13125 entre semillas), la
cifra se confirma con 20 episodios antes de decidir si vale la pena escalar más presupuesto
(ver resultado abajo).

**Nota de infraestructura — la Mac corrió sin cargador toda la corrida.** El `caffeinate -s`
no evita que macOS entre en ciclos de "Sleep Service Back to Sleep" con batería (visto:
22 ciclos de sueño/despertar en 30 min, batería cayendo ~0.7%/min). i7 sobrevivió igual
porque los checkpoints cada 500k pasos limitan la pérdida a un máximo de ~25 min de cómputo
si el proceso muere, pero el fps promedio se resintió (cayó de 1081 a ~740-850 durante el
tramo sin cargador). Para corridas largas futuras: **conectar el cargador antes de lanzar**,
no solo `caffeinate`. Se repitió con la evaluación de 20 episodios (se olvidó envolverla en
`caffeinate` al lanzarla) — corregido ahí también, pero la causa raíz sigue siendo la
batería, no el software.

**Confirmado con 20 episodios**: media 5059.0, máximo 13125.0, std 3478.2, max_esperado
9093.7 — consistente con la cifra oficial de 5 episodios (5507.0). No es un artefacto de
muestra chica; el salto es real.

### i12: doblar otra vez — el salto se repite, más grande

`i12_ppo_80m` (mismos hiperparámetros que i7, 2x pasos) terminó en 1227.5 min (1086 FPS,
~20.5h, con la Mac en corriente la mayor parte del tiempo — ver nota de batería abajo). La
curva de `ep_rew_mean` no solo no tuvo plateau, aceleró: de 4070 en el 50% del entrenamiento
salta a >15000 en el último cuarto, con incrementos bruscos entre lecturas (ej. 42.6M→45.3M:
6980→5960, luego 55.9M: 11200) — consistente con el agente encontrando una estrategia
cualitativamente distinta (episodios de hasta 27000 pasos, vs cientos-miles en i7), no solo
afinando la política existente.

Evaluación (5 episodios): **media 36377.0, máximo 52575.0, max_esperado 54993.8** — 6.6x la
media de i7. Confirmado con 20 episodios: **media 26105.2, máximo 52605.0, max_esperado
45854.7** — la cifra de 5 episodios fue optimista (sesgo de muestra chica hacia arriba, ya
visto con i7), pero el salto real sigue siendo ~5x sobre i7. La std es enorme en términos
absolutos (17025 con 20 ep) pero **proporcionalmente similar a i7** (~65% del la media en
ambos) — no se volvió más errático, solo escaló.

**Bug de infraestructura encontrado y corregido**: `video_length` seguía en 6000 (el default
original, puesto cuando "un episodio cabía de sobra"). Con episodios de hasta 27000 pasos, el
video de i12 se cortó a menos de un cuarto del episodio real — el mp4 no mentía sobre el
score, pero no mostraba la partida completa. Subido a 30000 en `agent/configs.py`. Regla 5
del proyecto: bug silencioso, sin error, solo un video incompleto.

Con dos duplicaciones seguidas mostrando el mismo patrón (i7→i12 fue incluso más marcado
que i6→i7), vale la pena seguir mientras el patrón aguante — **i13 dobla otra vez a 160M**.

### i9: negativo — 50 cuantiles fue demasiado poco

`i9_qrdqn_q50` (QRDQN, `n_quantiles` 200→50) sí fue 3x más rápido que i4 (220.9 min vs 670,
confirma que la pérdida cuantílica O(N²) era el cuello). Pero perdió en **todos** los ejes:
media 814 vs 972 (−16%), máximo 1175 vs 1775 (−34%), max_esperado 1123 vs 1520 (−26%). La
resolución distribucional que hace bueno a QRDQN para el máximo se perdió con menos cuantiles
— no hay premio de varianza que compense: el std bajó (266.5 vs 472.2) pero a costa de tirar
también la media. Esta vía queda cerrada: no vale la pena escalar QRDQN reducido, y `n_quantiles`
por debajo de 200 no es un buen trade-off aquí.

### i8: la media empeoró (¡al doble de presupuesto!) — y el porqué es más importante que el número

`i8_dqn_10m` dio **795.0 de media, peor que i3 (1132.0) con la mitad de los pasos.** Esto
parecía refutar de un plumazo la hipótesis de que "el plateau de i3 era falta de presupuesto".
Pero la curva de entrenamiento (`ep_rew_mean`, muestreada del log) cuenta otra historia:

```
step 3.7M -> 772   step 5.0M -> 762   step 6.0M -> 886 (pico)   step 8.2M -> 685   step 10M -> 773
```

**DQN alcanzó su pico en el paso 6.03M y se degradó después.** No es un plateau monótono
como el de i3 a 5M — es un pico y caída, un patrón conocido de inestabilidad de DQN
(sobreestimación de valores Q, agravada aquí por un `buffer_size=100_000` que en 10M pasos se
sobrescribe 100 veces). El pipeline solo guarda y evalúa los **pesos finales**
(`model.save()` al terminar `learn()`), así que estuvo a punto de descartar el mejor modelo
que produjo esta corrida.

**Verificado, no solo sospechado:** se evaluó a mano el checkpoint de 6M
(`agent/runs/i8_dqn_10m/checkpoints/i8_dqn_10m_6000000_steps.zip`, guardado gracias al
`save_freq` de infraestructura que ya estaba a 500k) con los mismos 5 episodios greedy:

| Pesos | Media | Máx | Std |
|---|---|---|---|
| final (10M) | 795.0 | 1080 | 191.2 |
| **checkpoint 6M** | **1023.0** | 1350 | 279.9 |

+29% solo por elegir mejor los pesos, sin re-entrenar nada. Y 1023 está dentro del margen de
ruido de i3 (1132, SE~90-125 con 5 episodios) — así que la conclusión real de i8 **no** es
"DQN a 10M es peor que a 5M". Es: **DQN no mejora monótonamente con más pasos, y el pipeline
necesita seleccionar el checkpoint, no asumir que el final es el mejor.**

Este resultado de 6M **no se registra como fila nueva en el CSV** — no es una iteración que
aísle una sola causa, es un diagnóstico sobre una ya cerrada. `evaluate.py` ahora acepta
`--checkpoint <ruta>` para repetir este chequeo sin scripts ad-hoc.

**Corrección importante al hallazgo anterior — el "pico" no es estable.** Se evaluaron tres
checkpoints más de la misma corrida (5.5M, 6.5M, 7M) para acotar el pico de 6.03M:

| Paso | Media | Máx | Std |
|---|---|---|---|
| 5.5M | 913.0 | 1640 | 478.0 |
| **6.0M** | **1023.0** | 1350 | 279.9 |
| 6.5M | 513.0 | 695 | 179.4 |
| 7.0M | 729.0 | 1120 | 264.5 |

De 1023 a 513 en solo 500k pasos, y de vuelta a 729 medio millón después. Esto **no** es un
pico único y estable que basta con "encontrar" — es oscilación (la tríada mortal de DQN:
bootstrap + aproximación de función + off-policy, con `target_update_interval=1000`
demasiado frecuente). Conclusión revisada: **no cherry-picking de checkpoints** mirando el
score de 5 episodios — eso es la regla 4 disfrazada (ajustar mirando la métrica) y con esta
varianza es indistinguible de tirar una moneda. `--checkpoint` se reserva para el protocolo de
20 episodios de i11 (selección final), no para elegir a ojo entre corridas cerradas.

La palanca correcta no es buscar mejor el checkpoint: es **atacar la causa** de la
oscilación. Ver i10 abajo.

### i6: resultado y por qué cambió el orden

`i6_ppo_tuned` mejoró la media solo **+4.2%** (873→910), en el borde inferior de lo esperado
(+5-15%), y **no** cerró el hueco con i3 (1132): sigue 19% por debajo. El máximo esperado sí
subió más (1240→1425, +14.9%), pero por el motivo equivocado — el std **empeoró** (316.5→443.9)
en vez de mejorar, así que la ganancia es varianza, no la estabilidad que se esperaba del
annealing.

Más importante para decidir i7 vs i8: la curva de entrenamiento de i6 (`ep_rew_mean`,
muestreada cada 40 iteraciones) sube de ~130 a un régimen de **700-880 que oscila sin subir
más** en el último tercio del entrenamiento — el mismo rango donde terminó i5 (804-888). La
premisa que puso a i7 antes que i8 en el plan original ("la curva de PPO todavía sube") ya no
se sostiene con el schedule corregido: PPO parece plateaudo en el mismo sitio con o sin
annealing. Nota aparte, no una señal de inestabilidad: `clip_fraction` subió a 0.25-0.40 al
final (más que en i5) simplemente porque `clip_range` decayó a ~0 y cualquier cambio de
probabilidad, por chico que sea, excede un margen casi nulo — es mecánico, no señal de
políticas agresivas.

Con eso, la apuesta de mayor valor esperado para subir la media deja de ser "más pasos sobre
PPO" y pasa a ser **i8 (DQN a 10M, el presupuesto para el que el rl-zoo diseñó la receta)**:
DQN sigue siendo el líder claro en media (1132 vs 910) y nunca se probó al doble de
presupuesto. i7 se pospone, no se descarta — si i8 también plateaudo, confirma que 5-10M ya
agota la receta estándar y hay que mirar hiperparámetros (`target_update_interval`) en vez de
presupuesto puro.

**Descartada `dqn_buffer300k`**: `optimize_memory_usage=True` exige también
`handle_timeout_termination=False` o `ReplayBuffer` lanza `ValueError` — son dos cambios
semánticos, no uno. Si hace falta atacar el plateau de DQN más adelante, la palanca limpia es
`target_update_interval` (1000 hoy, 10000 en el Nature DQN), no el buffer.

Riesgo de infraestructura: `--resume` no restaura el replay buffer de DQN (`Algo.load()` no
lo guarda), así que un crash en una corrida larga de DQN obliga a reiniciar desde cero — otro
motivo para dar los turnos overnight a PPO.

**La Mac se duerme y detiene el entrenamiento por completo.** Pasó en i6: `pmset -g log`
mostró ciclos de "Maintenance Sleep" (la tapa cerrada) durante casi 3h, en las que el proceso
acumuló solo 5 min de CPU real — 0 FPS mientras el sistema dormía, no throttling parcial.
Arreglo: `caffeinate -s -w <pid>` enganchado al proceso de entrenamiento, que libera la
asserción solo cuando el proceso termina. Obligatorio para i7/i8/i10 (overnight): lanzar
siempre como `python -m agent.train --config <cfg> & caffeinate -s -w $!`.

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

### Cómo leer las curvas de entrenamiento

SB3 aplica el `Monitor` **antes** del `AtariWrapper` (está documentado en `make_vec_env`:
*"the wrapper specified by this parameter will be applied after the Monitor wrapper"*). Eso
cambia el significado de dos métricas del log, y es fácil malinterpretarlas:

| Métrica del log | Qué es en realidad |
|---|---|
| `rollout/ep_rew_mean` | score real **sin recortar** de una partida completa de 3 vidas |
| `rollout/ep_len_mean` | pasos **previos** al frameskip; dividir entre 4 para pasos del agente |

Es una buena noticia: `ep_rew_mean` es directamente comparable con el score de evaluación, así
que se puede seguir el progreso en tensorboard sin parar a evaluar. Al arrancar I2 marcaba
~165, coherente con el agente aleatorio (123.5 de media en 10 episodios).

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

### Videos legibles

Los mp4 de evaluación se graban a `VIDEO_SCALE=4` (640x840) y `VIDEO_FPS=15` (`agent/env.py`).
A 160x210/30fps originales las balas eran de 1-2 px y el video corría al doble de velocidad
real (cada frame grabado son 4 frames de emulador a 60 Hz). `python -m agent.evaluate --config
<cfg> --solo-video` graba sin tocar `results.csv`, para re-grabar iteraciones ya cerradas.

## Estado a la fecha de entrega (2026-09-21)

La mejor iteración cerrada hasta hoy es **i12_ppo_80m** (score medio 36377, máximo 52575 en
5 episodios greedy) — es la que se presenta como resultado del proyecto.

**i13_ppo_160m** (mismos hiperparámetros que i12, 2x más pasos) quedó como trabajo en curso:
arrancó el 2026-09-21 y a ~1000-1080 fps (medido en i7/i12) tarda ~41-44 h en completar los
160M pasos, más de lo que da el plazo de esta entrega. Sigue corriendo en background sin
`caffeinate` (no hay chequeos automáticos de progreso); si cierra con mejor score que i12 se
añade como fila nueva a `results.csv` y se documenta en la sección de iteraciones como trabajo
posterior a la entrega, no como parte de ella.

## Qué queda por decidir

- Si i6 (PPO con schedules) supera a i5, la rama de escalado overnight es PPO (i7); si no, el
  cuello no era el annealing y hay que revisar otra cosa antes de comprometer una noche.
  i8 (DQN 10M) corre en paralelo para zanjar si el plateau de DQN es real.
- El modelo de entrega sale de comparar los 2 mejores candidatos con `--episodios 20` (no se
  ajustan hiperparámetros con ese número, solo se elige cuál pesos se congelan).
