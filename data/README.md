# Datos

Esta carpeta está fuera del repositorio (`.gitignore`): solo se versionan este archivo y `.gitkeep`. Los archivos se generan con los notebooks de `notebooks/` y viven en Drive (`MyDrive/qlora-series-ar/data/`). Para trabajar en local se copian aquí.

## Por qué no se publican

- **Catálogo y series:** vienen de la API de Series de Tiempo de datos.gob.ar. Falta leer la licencia del portal y citarla antes de redistribuirlos.
- **Preguntas y resultados por pregunta:** las preguntas las generó Claude Haiku 5.5. Falta revisar los Términos Comerciales de Anthropic antes de publicarlas (ver `docs/uso_api_anthropic.md`).
- Los cálculos agregados (tablas y figuras) sí están en el repositorio: `README.md`, `results/` y `figures/`.

Estado de cada decisión: `docs/datos_y_licencias.md`.

## Archivos

| Archivo | Qué es | Lo produce |
|---|---|---|
| `catalogo_busqueda.csv` | Catálogo de 8.226 series con título, descripción, dataset, tema, unidades, frecuencia, estado y partición | `00b`, `01` |
| `catalogo_busqueda_antes_tc.csv` | El mismo catálogo sin el dataset de tipo de cambio (8.209 series). La partición publicada se calculó sobre este | `00b` |
| `specs_train_v1.csv`, `train_final2.csv` | Especificaciones y preguntas de entrenamiento | `00b` |
| `consultas_busqueda_train_parcial.csv` | Consulta de búsqueda que escribió Haiku para cada pregunta de entrenamiento | `02` |
| `trayectorias_train_v1.jsonl`, `tools_v1.json` | Conversaciones de entrenamiento y esquema de herramientas usados para entrenar el modelo publicado | `02` |
| `trayectorias_train.jsonl` | Reconstrucción local del conjunto anterior. Es idéntica a `trayectorias_train_v1.jsonl` | `scripts/construir_trayectorias.py` |
| `trayectorias_train_v2.jsonl` | Experimento con otra regla de consulta. No se usó para entrenar | `07`, celda 6 |
| `eval_test_final2.csv` | Preguntas de test con su serie objetivo y su familia | `00b` |
| `baseline_*` y `finetuned_*` (`.csv`) | Resultado por pregunta de cada corrida, en validación y en test | `05`, `07`, `08` |
| `*_episodios.jsonl`, `*_resultados.jsonl` | Conversaciones completas y resultados detallados de las corridas de test | `08` |

## Verificación

Con estos archivos en esta carpeta:

```bash
uv run python scripts/particionar.py --datos data --catalogo catalogo_busqueda_antes_tc.csv --verificar
uv run python scripts/construir_trayectorias.py --datos data --solo-armar --comparar trayectorias_train_v1.jsonl
uv run python scripts/entrenar.py --datos data --modo datos
```
