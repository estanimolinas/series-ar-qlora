# Hoja de ruta

Estado al 16 de noviembre de 2026. Cada épica se divide en historias (qué necesita quién y para qué) y tareas.
Estados: **hecho**, **pendiente**, **GPU** (requiere Colab con T4), **API** (llama a la API de Anthropic, con costo),
**decisión** (depende de una decisión o de leer términos antes de actuar).

## E1. Reproducibilidad

| Historia | Estado | Tareas |
|---|---|---|
| Como lector, quiero reproducir los resultados sin GPU ni gasto, para comprobar que el código hace lo que dice | **hecho** | Entorno con `uv` y `uv.lock`; 126 tests; `particionar.py --verificar`; `construir_trayectorias.py --comparar`; `entrenar.py --modo datos` |
| Como lector, quiero el mismo entorno de Colab que se usó, para repetir el entrenamiento | **hecho**, con una salvedad | Versiones fijadas en `requirements-colab.txt`. `trl`, `bitsandbytes` y afines salen de una sesión nueva con `unsloth==2026.11.3`, no de la original. Salvedad: aún no se reentrenó con este archivo para confirmar que reproduce el adaptador |
| Como mantenedor, quiero que cada cambio corra ruff y los tests, para no romper lo verificado | **hecho** | `.github/workflows/ci.yml` (Python 3.10, 3.12 y 3.13). Se activa al publicar el repositorio |
| Como lector, quiero ver cómo se generaron las preguntas y qué filtros pasaron, para juzgar su calidad | **hecho**, con una salvedad | `especificaciones.py`, `preguntas.py` y `cache_api.py` con tests; `scripts/preparar_preguntas.py` (4 pasos, costo medido declarado). Las reglas reproducen las marcas de los conjuntos finales de train y test. Salvedad: las especificaciones de train se generaron con una versión previa y luego se corrigieron; correr el script sobre train da especificaciones equivalentes en criterio, no idénticas. La evaluación del buscador de `00b` sigue sin migrar |
| Como lector, quiero que el buscador devuelva el mismo orden en cualquier máquina, para que los resultados no dependan del equipo | **hecho** en test | Regla `round(s, 9), rowid`; las 1.170 búsquedas de test se reproducen idénticas. En validación falta repetirlo: requiere los episodios de validación de Drive |

## E2. Rigor de la evaluación

Es la épica que más fortalece la conclusión del proyecto. Hoy la comparación es asimétrica (el base lleva un prompt de
sistema, el ajustado no) y hay una sola semilla.

| Historia | Estado | Tareas |
|---|---|---|
| Como revisor, quiero una comparación con un prompt más fuerte para el base, para saber cuánto de la mejora es ajuste y cuánto es prompt | pendiente (**GPU**) | Iterar el prompt sobre un máximo de 60 preguntas de entrenamiento y 3 versiones; una corrida en validación y una en test; informar las tres versiones |
| Como revisor, quiero la comparación con un modelo mayor como agente, para tener una referencia de techo | pendiente (**API**) | Haiku 5.5 como agente sobre las mismas preguntas; declarar el costo |
| Como revisor, quiero saber cuánto varía el resultado entre semillas, para no atribuir al ajuste lo que es azar | pendiente (**GPU**) | Entrenar al menos 2 semillas más (75 minutos cada una) y evaluar en validación; informar la dispersión |
| Como revisor, quiero un evaluador que no penalice a un agente que consulta varias series, para que la métrica sea justa | pendiente | Revisar `evaluador.py`: hoy `serie_ok` mira solo la última consulta a `get_series` |
| Como revisor, quiero ver el resultado con preguntas coloquiales, para saber si generaliza más allá del lenguaje técnico | pendiente (**GPU**), opcional | Evaluar el conjunto coloquial de test, sin tocar lo ya medido |

## E3. Calidad de la búsqueda

El buscador tiene un techo: en 8 a 10 % de las preguntas la serie correcta no aparece entre los 10 candidatos.

| Historia | Estado | Tareas |
|---|---|---|
| Como usuario del agente, quiero que la serie correcta aparezca entre los candidatos más a menudo, para que el agente pueda acertar | pendiente | Retomar el experimento de consulta construida desde la pregunta (notebook `07`, celda 6) y medirlo; probar búsqueda híbrida con embeddings |

## E4. Publicación

| Historia | Estado | Tareas |
|---|---|---|
| Como visitante, quiero saber bajo qué licencia puedo usar el código | **hecho** | Licencia MIT |
| Como mantenedor, quiero publicar solo lo que puedo publicar, para no incumplir licencias ni términos | **decisión** | Leer la licencia del portal de datos abiertos y los Términos Comerciales de Anthropic; decidir si se publican el catálogo, las preguntas y los resultados por pregunta (`data/README.md`) |
| Como lector, quiero probar el modelo ajustado, para ver el resultado sin entrenar | **decisión** | Publicar solo el adaptador en Hugging Face, después de lo anterior y de verificar la licencia del modelo base |
| Como lector, quiero ver la observabilidad del agente, para entender cómo se traza | pendiente | Capturas de Langfuse en `docs/`; generar claves nuevas de Langfuse y de Anthropic antes de publicar |
| Como mantenedor, quiero ordenar el trabajo pendiente en el repositorio | pendiente | Crear issues desde esta hoja de ruta, con etiquetas por épica |

## E5. Perfil y siguiente proyecto

| Historia | Estado | Tareas |
|---|---|---|
| Como visitante del perfil, quiero ver de un vistazo qué hace esta persona | pendiente | Reescribir el README del perfil con foco en ingeniería de IA (evaluación de LLM, agentes, pipelines de datos públicos) |
| Como autor, quiero un segundo proyecto de ingeniería de datos | pendiente | Definir alcance de la versión 0.1 |
