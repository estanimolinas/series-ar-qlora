# Guía de métricas y estadística del proyecto

Explica, en palabras simples, cada herramienta estadística que usa el proyecto, por qué se eligió y qué no
permite concluir. Los números de ejemplo son los de la corrida de test (585 preguntas).

## 1. Qué se mide: exactitud de extremo a extremo

Cada pregunta termina en correcta o incorrecta. Es correcta solo si el agente (1) terminó con `submit_answer`,
(2) consultó la serie correcta, (3) dio el valor correcto y (4) dio la fecha correcta. La **exactitud** es la
fracción de preguntas correctas: 426 de 585 = 72,8 %.

Ese 72,8 % es lo que pasó en *estas* 585 preguntas. Lo que interesa es qué pasaría con otras preguntas parecidas.
Para eso sirve el intervalo de confianza.

## 2. Intervalo de confianza para una proporción

Si se repitiera la evaluación con otras 585 preguntas parecidas, el porcentaje no daría exactamente 72,8 %.
Un **intervalo de confianza del 95 %** es un rango construido con un método que, aplicado muchas veces, contiene el valor
real en el 95 % de los casos. Se lee así: "con estos datos, la exactitud verdadera es compatible con un valor entre 69 % y 76 %".
No significa que haya 95 % de probabilidad de que el valor esté dentro de ese rango específico.

Regla práctica: si los intervalos de dos modelos **no se tocan**, la diferencia no se explica por azar de muestreo
(sujeto a los supuestos de la sección 4).

## 3. Por qué el intervalo de Wilson y no la fórmula simple

La fórmula que más se enseña (llamada de Wald) es `p ± 1,96 · √(p(1−p)/n)`. Funciona con muestras grandes y
proporciones medias, y falla fuera de ese caso. Wilson está construido para no fallar ahí:

| Resultado | Wald (simple) | Wilson |
|---|---|---|
| 2 de 10 | −4,8 % a 44,8 % (porcentaje negativo: imposible) | 5,7 % a 51,0 % |
| 9 de 10 | 71,4 % a 108,6 % (más de 100 %: imposible) | 59,6 % a 98,2 % |
| 0 de 10 | 0 % a 0 % (afirma certeza con 10 casos) | 0 % a 27,8 % |

Con 585 preguntas las dos fórmulas casi coinciden (base: Wilson 24,9–32,2 %, Wald 24,7–32,0 %). Se usa Wilson igualmente porque
el proyecto también calcula intervalos por tipo de pregunta (grupos de ~100–200 preguntas) y casos con muy pocos aciertos
o muy pocos errores, donde la diferencia sí importa. Es una elección segura que no hace falta revisar caso por caso.

**Sobre el −2,8e-17 que aparece en el registro de errores:** el límite inferior de Wilson para 0 aciertos es exactamente
0. La computadora, al hacer la cuenta con decimales binarios, devolvió −0,000000000000000028 (un error de redondeo del orden de la
precisión de la máquina). Es irrelevante para el resultado, pero un intervalo con límite negativo es inválido para una proporción,
así que `evaluador.wilson` recorta el resultado a [0, 1] y hay un test que lo cubre.

## 4. Lo que Wilson no resuelve: las preguntas no son independientes

Wilson supone que cada pregunta es un intento independiente. Acá no lo son: las 585 preguntas de test salen de unas
23 **familias** de datasets (series casi iguales que comparten estructura). Si el modelo falla en una familia, tiende a fallar en
varias de sus preguntas a la vez. Es como encuestar a 585 personas que viven en 23 casas: no se tiene la información de 585 personas independientes.
Consecuencia: el intervalo de Wilson es **más angosto de lo que debería**.

### Bootstrap agrupado por familias

El bootstrap estima la incertidumbre sin fórmulas, remuestreando los propios datos:

1. Se toma la lista de 23 familias.
2. Se sortean 23 familias **con reposición** (una familia puede salir dos veces y otra ninguna).
3. Con las preguntas de esas familias se calcula la diferencia de exactitud (entrenado − base).
4. Se repite 10.000 veces.
5. Se ordenan las 10.000 diferencias y se toman los percentiles 2,5 y 97,5: ese es el intervalo del 95 %.

Se remuestrean familias enteras (no preguntas sueltas) justamente para respetar la dependencia dentro de la familia.
Límite honesto: con solo 23 familias, el propio bootstrap es aproximado; el intervalo es una guía, no una garantía.

## 5. Comparación emparejada y prueba exacta

Los dos modelos respondieron las mismas preguntas, así que se pueden emparejar. Cada pregunta cae en una de cuatro
casillas: ambas correctas, ninguna, solo el entrenado, solo el base. Las dos primeras no dicen nada sobre quién es mejor;
**solo importan las preguntas en que difieren**. Si no hubiera diferencia real, cada pregunta discordante sería "a favor del
entrenado" o "a favor del base" con igual probabilidad, como una moneda. La **prueba binomial exacta** calcula qué tan raro sería
el reparto observado si la moneda fuera justa. Es la misma idea de la prueba de McNemar, en su versión exacta.

También es optimista, porque trata cada pregunta discordante como independiente (ver sección 4). Por eso el informe
cita el bootstrap por familias como intervalo principal.

## 6. Cómo se validó el evaluador

Antes de medir modelos se validó que el evaluador mide lo que dice medir:

- **Agente oráculo:** un "agente" que conoce la respuesta y llama a las herramientas correctamente. El evaluador tiene que darle 100 %. Si no, el evaluador tiene un error.
- **Controles negativos:** agentes que responden mal a propósito (serie equivocada, valor al azar). El evaluador tiene que darles ~0 %.
- Los tests de `tests/test_evaluador.py` y `tests/test_agente.py` repiten estas comprobaciones sobre datos sintéticos.

Sin esto, una comparación entre modelos no tiene valor: no se sabría si la diferencia viene de los modelos o del evaluador.

## 7. Por qué la partición es por familias y test se usa una sola vez

- **Partición por familias:** series de una misma familia son casi duplicadas. Si unas van a entrenamiento y otras a test, el modelo puede
  "recordar" en lugar de generalizar. Poner familias enteras en un solo conjunto evita esa fuga.
- **Test una sola vez:** si se ajusta el prompt o el modelo mirando test, test deja de ser una prueba independiente. Los ajustes se hacen con
  train/validación y test se corre una vez, sin cambios posteriores.

## 8. Qué NO prueban estos números

- Solo hay un entrenamiento (una semilla): no se conoce la variación entre entrenamientos.
- El modelo base lleva un prompt y el entrenado no: la diferencia mezcla el efecto del modelo y el del prompt (control pendiente).
- La mejora observada es sobre todo de **procedimiento** (terminar con `submit_answer`, parámetros bien formados, conversión de porcentaje).
  La elección de la serie correcta rinde igual en ambos (76 % vs. 75 %).
- Las preguntas son sintéticas y de la misma distribución que el entrenamiento: no se mide el rendimiento con preguntas escritas por personas.
