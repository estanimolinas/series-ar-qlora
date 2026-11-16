"""Constantes y rutas. Nada de rutas personales: los datos se ubican con la variable SERIES_QLORA_DATA."""
from __future__ import annotations

import os
from pathlib import Path

API_BASE = "https://apis.datos.gob.ar/series/api/series"
HEADERS = {"User-Agent": "Mozilla/5.0"}

BASE_MODEL_ID = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"

MAX_PASOS = 6          # máximo de turnos del agente por pregunta
MAX_NUEVOS = 300       # máximo de tokens generados por turno
MAX_SEQ_INFERENCIA = 4096
MAX_SEQ_ENTRENAMIENTO = 2048

# Partición por familias de datasets
UMBRAL_FAMILIA = 0.85          # similitud de títulos desde la cual dos datasets son de la misma familia
TOPE_FAMILIA = 150             # tope de series por familia al calcular el balance entre splits
PROPORCIONES = (0.8, 0.1, 0.1)  # train, val, test
SPLITS = ("train", "val", "test")
N_SEMILLAS = 5000
# Datasets que la similitud de títulos no une pero son la misma publicación partida en varias tablas
UNIONES_MANUALES = (
    ("193", "194", "195", "196", "197", "198", "199", "464"),   # IPC por provincia, CABA y Gran Resistencia
    ("448", "449", "450"),                                      # precios mayoristas
    ("183", "184", "185"),                                      # balance cambiario: cobros y pagos
    ("75", "76", "77", "78", "79"),                             # intercambio comercial argentino
    ("152", "153", "154", "157"),                               # asalariados registrados, sector privado
)
ESTADOS_USABLES = ("normal", "api_adelantada")

PARAMS_TOOL = frozenset({"ids", "start_date", "end_date", "representation_mode", "limit", "sort"})


def directorio_datos() -> Path:
    """Carpeta con catálogo, índices, cachés y resultados.

    En Colab: /content/drive/MyDrive/qlora-series-ar/data. En local: ./data por defecto.
    """
    return Path(os.environ.get("SERIES_QLORA_DATA", "data"))
