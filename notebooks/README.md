# Notebooks

Notebooks de Colab del proyecto, en el orden en que se ejecutan. Están sin salidas guardadas, y cada uno empieza con una celda de Markdown que indica qué contiene, el runtime, las entradas y salidas, los secretos que necesita, el costo y las advertencias. Los datos viven en `MyDrive/qlora-series-ar/data/` (ver `data/README.md`).

La lógica reutilizable está en `src/series_qlora/` y `scripts/`, con tests. Los notebooks son el registro de cómo se obtuvieron los datos y los resultados. No existe un notebook 03.

| Notebook | Qué hace | Runtime | Equivalente en el repositorio |
|---|---|---|---|
| `00_api_datosgovarg` | Explora y verifica la API de Series de Tiempo; arma el catálogo verificado | CPU | |
| `00b_test_haiku` | Partición inicial, tipo de cambio, especificaciones y preguntas con Claude Haiku 5.5, validadores, correcciones y evaluación del buscador | CPU, API de Anthropic | `cache_api.py`, `especificaciones.py`, `preguntas.py`, `scripts/preparar_preguntas.py` (preguntas); `particion.py`, `herramientas.py` (el resto). La evaluación del buscador sigue solo en el notebook |
| `01_buscador_y_particion` | Índice de búsqueda y partición por familias de datasets | CPU | `herramientas.py`, `particion.py`, `scripts/particionar.py` |
| `02_trayectorias` | Trayectorias de entrenamiento (Haiku escribe solo la consulta) | CPU, API de Anthropic | `trayectorias.py`, `scripts/construir_trayectorias.py` |
| `04_agente_y_evaluacion` | Herramientas del agente y evaluador, validados con un agente oráculo | CPU | `herramientas.py`, `evaluador.py`, `agente.py` |
| `05_agente_baseline` | Modelo base sobre validación | GPU T4 | `agente.py`, `corrida.py`, `scripts/evaluar.py` |
| `06_qlora_entrenamiento` | Entrenamiento QLoRA (75 min) | GPU T4 | `entrenamiento.py`, `scripts/entrenar.py` |
| `07_evaluacion_modelo_entrenado` | Modelo entrenado sobre validación y análisis de la búsqueda | GPU T4 | `agente.py`, `corrida.py`, `scripts/evaluar.py` |
| `08_evaluacion_test` | Evaluación en test, una vez por modelo | GPU T4 | `scripts/evaluar.py` |
| `09_comparacion_y_graficos` | Comparación en test y cinco figuras | CPU | `graficos.py` |

Los secretos de Colab que usan: `ANTHROPIC_API_KEY` (00b, 02) y `LANGFUSE_PUBLIC_KEY` y `LANGFUSE_SECRET_KEY` (04, 05, 07, 08). Ninguna clave está escrita en los notebooks.
