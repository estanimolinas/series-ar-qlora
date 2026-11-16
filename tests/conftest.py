"""Fixtures con datos 100 % sintéticos (series inventadas): los tests no usan ni producen datos reales."""
import json

import pandas as pd
import pytest

from series_qlora.herramientas import Herramientas, construir_indice, crear_cache


@pytest.fixture
def catalogo():
    cols = ["serie_id", "serie_titulo", "distribucion_titulo", "frecuencia", "serie_unidades",
            "dataset_titulo", "dataset_tema", "serie_descripcion"]
    filas = [
        ("ser_ipc_nivel", "Índice de precios al consumidor, nivel general", "IPC mensual", "mensual", "índice",
         "Índice de precios", "Precios", None),
        ("ser_ipc_alim", "Índice de precios al consumidor, alimentos y bebidas", "IPC mensual", "mensual", "índice",
         "Índice de precios", "Precios", None),
        ("ser_emae", "Actividad económica, nivel general", "EMAE", "mensual", "índice",
         "Estimador de actividad", "Actividad", "Medición mensual de la actividad"),
        ("ser_pbi_anual", "Producto interno bruto a precios constantes", "PIB anual", "anual", None,
         "Cuentas nacionales", "Cuentas nacionales", None),
        ("ser_exportaciones", "Exportaciones de bienes en millones de dólares", "Comercio exterior", "trimestral",
         "USD", "Balanza de pagos", "Sector externo", None),
    ]
    return pd.DataFrame(filas, columns=cols).set_index("serie_id")


class ApiFalsa:
    """API inyectable: respuestas guionadas por clave de parámetros; cuenta las llamadas."""

    def __init__(self, respuestas=None, status=200):
        self.respuestas, self.status, self.llamadas = respuestas or {}, status, []

    def __call__(self, params):
        self.llamadas.append(params)
        clave_ids = params.get("ids")
        return self.status, self.respuestas.get(clave_ids, {"data": [["2020-01-01", 1.5]]})


@pytest.fixture
def api():
    return ApiFalsa()


@pytest.fixture
def herramientas(tmp_path, catalogo, api):
    crear_cache(tmp_path / "cache.db")
    construir_indice(tmp_path / "indice.db", catalogo)
    h = Herramientas(tmp_path / "cache.db", tmp_path / "indice.db", catalogo, api=api)
    yield h
    h.cerrar()


def llamada(nombre, **args):
    """Texto que emite el modelo para una llamada a herramienta (formato Hermes de Qwen3)."""
    return f"<tool_call>\n{json.dumps({'name': nombre, 'arguments': args}, ensure_ascii=False)}\n</tool_call><|im_end|>"
