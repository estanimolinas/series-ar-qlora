# Guía del entrenamiento QLoRA

Explica qué se entrenó, cómo funciona QLoRA y por qué se eligió cada parámetro. Complementa a `guia_metricas.md`
(que explica cómo se mide) y a `decisiones.md` (que resume las decisiones). Los números salen del código
(`src/series_qlora/entrenamiento.py`) y del registro de la corrida (notebook `06_qlora_entrenamiento`).

**Aclaración que conviene tener presente:** los hiperparámetros son los habituales para LoRA y QLoRA. Se entrenó una
sola vez y no se probaron alternativas. Esta guía explica por qué cada valor es razonable, no que sea el mejor posible.

## 1. En 30 segundos

Se tomó Qwen3-4B-Instruct (4.055 millones de parámetros), se congelaron sus pesos y se comprimieron a 4 bits para que
entre en una GPU de 15 GB. Encima se agregaron matrices pequeñas (un adaptador LoRA de 33 millones de parámetros,
0,81 % del total) y solo esas se entrenaron, durante una época (100 pasos, 75 minutos en una T4) con 1.586 conversaciones
de ejemplo. La pérdida se calculó solo sobre lo que escribe el asistente. Resultado: el modelo aprendió el
procedimiento (llamar a las herramientas en orden y terminar con `submit_answer`), no conocimiento nuevo sobre las series.

## 2. Qué es QLoRA

QLoRA combina dos ideas independientes.

### 2.1 LoRA: entrenar un adaptador en lugar del modelo

Entrenar todo el modelo exige guardar, por cada parámetro, el valor, el gradiente y los estados del optimizador. Con 4.000
millones de parámetros eso no cabe en 15 GB. LoRA congela la matriz original `W` de cada capa y aprende una corrección
de rango bajo:

```
W' = W + (alpha / r) · B · A        A: r × entrada      B: salida × r
```

- **`r` (rango)** es el ancho de las matrices `A` y `B`. Cuanto mayor, más capacidad de corrección y más parámetros.
- **`alpha`** escala la corrección. Con `alpha = r` (acá 16 y 16) el factor es 1 y no amplifica ni atenúa.
- **Arranque:** `B` empieza en ceros, así que antes de entrenar `W' = W`: el modelo parte idéntico al original.
- **Qué se entrena:** solo `A` y `B`. `W` no cambia.

**Cuánto es "33 millones".** Para Qwen3-4B (36 capas, dimensión 2.560, capa intermedia 9.728, atención con 4.096 salidas
de consulta y 1.024 de clave y valor), cada proyección aporta `r · (entrada + salida)` parámetros:

| Proyección | Parámetros con r = 16 |
|---|---|
| `q_proj` | 106.496 |
| `k_proj`, `v_proj` | 57.344 cada una |
| `o_proj` | 106.496 |
| `gate_proj`, `up_proj`, `down_proj` | 196.608 cada una |
| **Por capa** | **917.504** |

36 capas × 917.504 = **33.030.144**, que coincide con lo que imprimió Unsloth ("33.030.144 de 4.055.498.240, 0,81 %").
Un test (`tests/test_entrenamiento.py`) fija esta cuenta.

### 2.2 Cuantización a 4 bits: que el modelo congelado ocupe poco

Los pesos congelados `W` se guardan comprimidos a 4 bits (biblioteca `bitsandbytes`). Se descomprimen sobre la marcha
al calcular y el adaptador `A`, `B` se entrena en mayor precisión. En el registro: 219 capas del modelo base en 4 bits
y todos los parámetros entrenables pertenecen al adaptador (el código lo comprueba y falla si no).
Memoria: 3,7 GB con el modelo cargado y 4,6 GB de pico durante el entrenamiento, sobre los 14,6 GB de la T4.
El modelo tiene 252 proyecciones lineales (7 por capa); el resto de las 219 cuantizadas queda en mayor precisión por el
criterio de cuantización dinámica de Unsloth. No se verificó cuáles son.

Si el modelo base no estuviera en 4 bits, esto sería LoRA y no QLoRA: por eso el código lo verifica antes de entrenar.

## 3. Hiperparámetros: qué hace cada uno y por qué ese valor

| Parámetro | Valor | Qué hace | Por qué ese valor |
|---|---|---|---|
| Modelo base | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` | Punto de partida, ya en 4 bits | Chico para una T4; ya sabe seguir instrucciones y usar herramientas |
| `r` | 16 | Capacidad del adaptador | Valor de uso común; la tarea es aprender un procedimiento, no mucho conocimiento. No se probó 8 ni 32 |
| `lora_alpha` | 16 | Escala de la corrección | `alpha = r` deja el factor en 1 |
| `lora_dropout` | 0 | Apaga unidades al azar para regularizar | Con una sola época no hay repetición que regularizar; además es el valor que Unsloth optimiza |
| Módulos objetivo | `q, k, v, o, gate, up, down` | Qué capas reciben adaptador | Todas las proyecciones lineales: más capacidad con costo bajo (0,81 % de parámetros) |
| Tasa de aprendizaje | 2e-4 | Tamaño del paso de cada actualización | Valor típico de LoRA. Es mayor que en entrenamiento completo (~1e-5) porque se ajusta solo un adaptador pequeño |
| Planificador | lineal, con 10 pasos de calentamiento | Sube el paso en los primeros 10 pasos y luego lo baja hasta 0 | Evita pasos grandes al comienzo, cuando la pérdida es alta, y cierra con pasos chicos |
| Optimizador | `adamw_8bit` | AdamW con estados de 8 bits | Ahorra memoria: con 33 M de parámetros son unos 200 MB. Es la configuración habitual de Unsloth |
| Decaimiento de pesos | 0,01 | Penaliza pesos grandes | Valor por defecto habitual |
| Lote | 2 por paso × 8 de acumulación = 16 | Ejemplos por actualización | 2 es lo que entra en memoria; se acumulan 8 pasos para tener un gradiente más estable |
| Épocas | 1 (100 pasos) | Veces que se recorre el conjunto | 1.586 ejemplos ÷ 16 = 99,1, o sea 100 pasos. Cada época tarda 75 minutos en la T4; no se probó más de una |
| Precisión | `fp16` | Aritmética en 16 bits | La T4 no admite `bf16` (el código elige `bf16` solo si la GPU lo soporta) |
| Longitud máxima | 2.048 en entrenamiento, 4.096 al evaluar | Tokens por conversación | El ejemplo más largo tiene 1.750 tokens. En evaluación el agente acumula más contexto |
| Semilla | 3407 | Fija el azar | Una sola semilla: no hay medida de la variación entre semillas |
| Gradient checkpointing | `unsloth` | Recalcula activaciones en lugar de guardarlas | Ahorra memoria a costa de tiempo |

## 4. Qué se aprende y qué no: `train_on_responses_only`

Cada ejemplo es una conversación: la pregunta, lo que escribe el asistente (la llamada a `search_series`, la llamada a
`get_series`, la respuesta con `submit_answer`) y lo que devuelven las herramientas. La pérdida se calcula solo sobre las
partes del asistente. Las respuestas de las herramientas y la pregunta se ven pero no cuentan para la pérdida.

- **Por qué:** el modelo no tiene que aprender a producir lo que devuelve una herramienta, porque eso lo trae la API.
- **Cuánto pesa:** solo el 11,7 % de los tokens entra en la pérdida.
- **Comprobación:** ningún ejemplo entrena sobre un `<tool_response>` (0 de 1.586, verificado en el registro).
- **Cómo se marca:** el código indica dónde empiezan los turnos del usuario (`<|im_start|>user`) y del asistente
  (`<|im_start|>assistant`).

## 5. Los datos y el conjunto de control

Las 1.705 trayectorias se arman con la pregunta, la consulta de búsqueda que escribió Claude Haiku 5.5, y todo lo demás
sale de los datos y del código (ver `trayectorias.py`). De ellas se apartan **3 familias de datasets completas** como
**control**: 119 trayectorias (7,0 %). Quedan 1.586 para entrenar.

**Por qué por familias y no al azar.** Las series de una misma familia son casi idénticas. Un control al azar tendría
series casi iguales a las entrenadas y la pérdida de control parecería mejor de lo que es. Apartar la familia entera
mide la pérdida sobre algo que el modelo no vio. Es la misma razón por la que la partición de evaluación es por familias.

**Cómo se eligen.** Se barajan las familias con la semilla 3407 y se van apartando hasta reunir al menos el 5 % de las
trayectorias, salteando las que por sí solas pesan más del 8 %. El procedimiento es determinista.

## 6. Cómo leer la curva de pérdida (figura 4)

La **pérdida** es la entropía cruzada por token: mide cuánta probabilidad le asigna el modelo a cada token correcto del
asistente, cuando se le da la conversación hasta ahí. Cuanto menor, mejor. No es la exactitud del agente.

| Paso | Pérdida de entrenamiento | Pérdida de control |
|---|---|---|
| 10 | 1,097 | |
| 20 | 0,361 | |
| 50 | 0,127 | 0,129 |
| 100 | 0,087 | 0,100 |

- **La caída inicial** (de 1,10 a 0,21 en 30 pasos) es el modelo aprendiendo el formato de las llamadas a herramientas.
- **La pérdida de control** baja de 0,129 a 0,100 entre el paso 50 y el 100: no hay señal de sobreajuste en esos dos puntos.
- **Qué no permite concluir:** el control tiene 119 trayectorias de solo 3 familias y se mide dos veces. Con eso no
  se puede descartar un sobreajuste leve, ni comparar con precisión contra la de entrenamiento (que es un promedio de
  los últimos 10 pasos, no un valor puntual). La diferencia de 0,014 en el paso 100 es chica, pero no hay forma de
  decir si es ruido.
- **Pérdida baja no implica agente bueno.** Mide qué tan bien se imita el procedimiento; que el agente acierte depende de
  elegir la serie correcta, y eso lo mide la evaluación (README).

## 7. Qué no se probó

- Otros valores de `r`, `alpha`, tasa de aprendizaje o decaimiento.
- Más de una época, o un control con más puntos de medición.
- Más de una semilla de entrenamiento.
- Otros modelos base.
- Una pérdida sobre todo el texto en lugar de solo las respuestas del asistente.

Los hiperparámetros no se ajustaron mirando validación ni test: se hizo una sola corrida completa. Eso evita sobreajustar a las
evaluaciones, pero también significa que no hay evidencia de que sean los mejores.

## 8. Reproducir y versiones

- **Local, sin GPU:** `uv run python scripts/entrenar.py --datos data --modo datos` verifica la separación de control
  (1.586 / 119, 100 pasos).
- **En Colab (T4):** `scripts/entrenar.py --modo completo`, o el notebook `06_qlora_entrenamiento`. Dependencias en
  `requirements-colab.txt`.
- **Versiones del registro:** Unsloth 2026.11.3, Transformers 5.17.0, PyTorch 2.14.1+cu130, Triton 3.8.0, CUDA 13.0,
  GPU Tesla T4. En el registro guardado de la evaluación en test figura Unsloth 2026.11.4.
- **`trl` y `bitsandbytes`:** no se imprimen en ningún registro. Se fijaron (`trl` 1.15.0, `bitsandbytes` 0.50.2, `peft` 0.21.2, `accelerate` 1.15.0, `datasets` 4.8.5)
  con las versiones que instala hoy `unsloth==2026.11.3`, no con las de la sesión original, así que pueden diferir levemente.

## 9. Preguntas que pueden hacer

**¿Por qué QLoRA y no ajustar todo el modelo?** Porque no entra en una T4. Con QLoRA se entrena el 0,81 % de los
parámetros y el pico de memoria es 4,6 GB.

**¿Por qué r = 16?** Es un valor habitual. No se probaron otros: la tarea (aprender un procedimiento) no parece
requerir mucha capacidad, pero eso es una suposición razonable y no un resultado.

**¿Por qué una época?** Cada época tarda 75 minutos en la T4 y el control no mostró señal de sobreajuste. No se probó
entrenar más.

**¿Se sobreajustó?** El control baja de 0,129 a 0,100 y no hay señal de sobreajuste, pero con 119 ejemplos de 3 familias
y dos mediciones es evidencia débil.

**¿Qué aprendió el modelo?** El procedimiento: el 99,7 % de las preguntas termina con `submit_answer` (contra 58 % del
base). No mejoró la elección de la serie (ver "Qué mejoró y qué no" en el README).

**¿Por qué una sola semilla?** Por tiempo de cómputo. Es una limitación declarada: no hay estimación de la variación
entre semillas, y los intervalos de confianza del README no la incluyen.

**¿El test se usó para elegir algo?** No. Hubo una sola corrida de entrenamiento completa, con los valores habituales, y test se evaluó una vez por modelo.
